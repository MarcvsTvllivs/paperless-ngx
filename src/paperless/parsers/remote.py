"""
Built-in remote-OCR document parser.

Handles documents by sending them to a configured remote OCR engine
(currently Azure AI Vision / Document Intelligence) and retrieving both
the extracted text and a searchable PDF with an embedded text layer.

When no engine is configured, ``score()`` returns ``None`` so the parser
is effectively invisible to the registry — the tesseract parser handles
these MIME types instead.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Self

from django.conf import settings

from paperless.version import __full_version_str__

if TYPE_CHECKING:
    import datetime
    from types import TracebackType

    from paperless.parsers import MetadataEntry
    from paperless.parsers import ParserContext

logger = logging.getLogger("paperless.parsing.remote")

_SUPPORTED_MIME_TYPES: dict[str, str] = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


# LOCAL PATCH (2026-07-30). Operator escape hatch for the skip-if-text gate in
# RemoteOcrParser.score(): while this file exists AND the deadline inside it is
# in the future, every PDF goes to the remote engine, text layer or not. For
# re-OCRing documents whose existing text layer is bad.
#
#   date -u -d '+12 hours' +%Y-%m-%dT%H:%M:%S+00:00 > /opt/paperless/remote-ocr-force-all
#
# No service restart is needed either way, the file is read per document. The
# deadline lives in the file rather than in a timer so the window cannot outlive
# itself if the container reboots. A missing, empty, or unparseable file means
# the gate is active, which is the safe direction (nothing extra billed, nothing
# extra leaves the network).
_FORCE_ALL_SENTINEL = Path("/opt/paperless/remote-ocr-force-all")


def _force_all_active() -> bool:
    """Return True while the force-all window is open.

    Returns
    -------
    bool
        True only when the sentinel file exists and holds an ISO 8601
        timestamp that has not passed yet.
    """
    import datetime

    try:
        raw = _FORCE_ALL_SENTINEL.read_text().strip()
    except OSError:
        return False

    try:
        expires = datetime.datetime.fromisoformat(raw)
    except ValueError:
        logger.warning(
            "Remote OCR force-all sentinel holds an unparseable deadline (%r), "
            "treating the skip-if-text gate as active",
            raw,
        )
        return False

    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=datetime.timezone.utc)

    if datetime.datetime.now(datetime.timezone.utc) >= expires:
        logger.info(
            "Remote OCR force-all window expired at %s, skip-if-text gate is back on",
            expires.isoformat(),
        )
        return False

    return True


# Operators that draw text, and the two rendering modes that draw nothing
# visible (3 = invisible, 7 = add to clip path only). An OCR layer is written
# in mode 3 by every producer worth naming: ocrmypdf, Azure, Adobe, scanner
# firmware. Visible text means the page's text IS its content, i.e. born
# digital, and must never be stripped.
_TEXT_SHOWING_OPS = {"Tj", "TJ", "'", '"'}
_INVISIBLE_MODES = {3, 7}


def _instructions_draw_visible_text(instructions, resources, depth: int = 0) -> bool:
    """Walk content-stream instructions looking for visibly drawn text."""
    import pikepdf

    mode = 0
    for instruction in instructions:
        op = str(instruction.operator)

        if op == "Tr" and instruction.operands:
            try:
                mode = int(instruction.operands[0])
            except (TypeError, ValueError):
                mode = 0

        elif op in _TEXT_SHOWING_OPS and mode not in _INVISIBLE_MODES:
            return True

        elif op == "Do" and depth < 4 and instruction.operands:
            # Text can live inside a form XObject. Missing that would call a
            # born-digital page "no visible text" and strip it, so recurse.
            try:
                xobjects = resources.get("/XObject", {})
                xobj = xobjects[str(instruction.operands[0])]
                if str(xobj.get("/Subtype", "")) != "/Form":
                    continue
                if _instructions_draw_visible_text(
                    pikepdf.parse_content_stream(xobj),
                    xobj.get("/Resources", pikepdf.Dictionary()),
                    depth + 1,
                ):
                    return True
            except Exception:  # noqa: BLE001 - an unreadable XObject is not proof of absence
                logger.debug("Could not inspect a form XObject, assuming visible text")
                return True

    return False


def _pages_with_visible_text(path: Path) -> tuple[set[int], int]:
    """Return the 1-based pages that draw visible text, and the page count.

    This is the whole born-digital test, and it is a determination rather than
    a guess. A scanned page carries its content as an image with the OCR text
    written invisibly on top (mode 3); a born-digital page draws its text
    visibly. ``gs -dFILTERTEXT`` removes both kinds, so a page with visible
    text must never be stripped: doing so blanks it (doc 7828, 6,023 chars in,
    2 out).

    The earlier heuristic asked whether the page held an image >= 1000 px,
    which got two Kalläne invoices wrong: born-digital pages with a large
    letterhead. Rendering mode gets them right.

    Anything unreadable counts as visible text, because the failure that
    matters is destroying a real page, not skipping one.

    Parameters
    ----------
    path:
        Absolute path to the PDF.

    Returns
    -------
    tuple[set[int], int]
        Pages drawing visible text, and the total page count.
    """
    import pikepdf

    visible: set[int] = set()
    try:
        with pikepdf.open(path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, 1):
                try:
                    if _instructions_draw_visible_text(
                        pikepdf.parse_content_stream(page),
                        page.get("/Resources", pikepdf.Dictionary()),
                    ):
                        visible.add(i)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Could not parse page %d of %s, treating it as born digital",
                        i,
                        path.name,
                    )
                    visible.add(i)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not open %s to classify its pages: %s", path.name, e)
        return set(), 0

    return visible, total


def _extract_pages(source: Path, pages: set[int], out: Path) -> Path | None:
    """Write a PDF holding only *pages* (1-based) of *source*, in order."""
    import pikepdf

    try:
        with pikepdf.open(source) as src:
            sub = pikepdf.Pdf.new()
            for n in sorted(pages):
                sub.pages.append(src.pages[n - 1])
            sub.save(out)
    except Exception as e:
        logger.warning("Could not extract pages from %s: %s", source.name, e)
        return None
    return out


def _splice_pages(
    original: Path,
    replacement: Path,
    pages: set[int],
    out: Path,
) -> Path | None:
    """Put the pages of *replacement* back into *original* at *pages*.

    The engine is handed only the scanned pages of a mixed document, so its
    output has to be spliced back in position, leaving the vector-text pages
    exactly as they were. Their text is already correct and is the one thing
    OCR must not touch.
    """
    import pikepdf

    try:
        with pikepdf.open(original) as src, pikepdf.open(replacement) as rep:
            targets = sorted(pages)
            if len(rep.pages) != len(targets):
                logger.error(
                    "Refusing to splice %s: engine returned %d pages for %d sent",
                    original.name,
                    len(rep.pages),
                    len(targets),
                )
                return None
            for i, n in enumerate(targets):
                src.pages[n - 1] = rep.pages[i]
            src.save(out)
    except Exception as e:
        logger.warning("Could not splice pages into %s: %s", original.name, e)
        return None
    return out


def _strip_text_layer(path: Path, tempdir: Path) -> Path | None:
    """Return a copy of *path* with its text layer removed, or None.

    Azure's searchable PDF keeps whatever text the input already carried and
    adds its own on top: measured 5.9x duplication on doc 7828. Re-OCRing a
    document with a bad text layer therefore has to remove that layer first,
    or the bad text survives in the archive copy next to the good text.

    Parameters
    ----------
    path:
        Absolute path to the PDF to strip.
    tempdir:
        Directory to write the stripped copy into.

    Returns
    -------
    Path | None
        The stripped copy, or None if Ghostscript failed.
    """
    out = tempdir / "stripped.pdf"
    try:
        subprocess.run(
            [
                "gs",
                "-q",
                "-o",
                str(out),
                "-sDEVICE=pdfwrite",
                "-dFILTERTEXT",
                str(path),
            ],
            capture_output=True,
            timeout=900,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Could not strip the text layer from %s: %s", path.name, e)
        return None

    if not out.exists() or out.stat().st_size == 0:
        logger.warning("Text-layer strip produced nothing for %s", path.name)
        return None
    return out


class RemoteEngineConfig:
    """Holds and validates the remote OCR engine configuration."""

    def __init__(
        self,
        engine: str | None,
        api_key: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.engine = engine
        self.api_key = api_key
        self.endpoint = endpoint

    def engine_is_valid(self) -> bool:
        """Return True when the engine is known and fully configured."""
        return (
            self.engine in ("azureai",)
            and self.api_key is not None
            and not (self.engine == "azureai" and self.endpoint is None)
        )


class RemoteDocumentParser:
    """Parse documents via a remote OCR API (currently Azure AI Vision).

    This parser sends documents to a remote engine that returns both
    extracted text and a searchable PDF with an embedded text layer.
    It does not depend on Tesseract or ocrmypdf.

    Class attributes
    ----------------
    name : str
        Human-readable parser name.
    version : str
        Semantic version string, kept in sync with Paperless-ngx releases.
    author : str
        Maintainer name.
    url : str
        Issue tracker / source URL.
    """

    name: str = "Paperless-ngx Remote OCR Parser"
    version: str = __full_version_str__
    author: str = "Paperless-ngx Contributors"
    url: str = "https://github.com/paperless-ngx/paperless-ngx"

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def supported_mime_types(cls) -> dict[str, str]:
        """Return the MIME types this parser can handle.

        The full set is always returned regardless of whether a remote
        engine is configured.  The ``score()`` method handles the
        "am I active?" logic by returning ``None`` when not configured.

        Returns
        -------
        dict[str, str]
            Mapping of MIME type to preferred file extension.
        """
        return _SUPPORTED_MIME_TYPES

    @classmethod
    def score(
        cls,
        mime_type: str,
        filename: str,
        path: Path | None = None,
    ) -> int | None:
        """Return the priority score for handling this file, or None.

        Returns ``None`` when no valid remote engine is configured,
        making the parser invisible to the registry for this file.
        When configured, returns 20 — higher than the Tesseract parser's
        default of 10 — so the remote engine takes priority.

        Parameters
        ----------
        mime_type:
            Detected MIME type of the file.
        filename:
            Original filename including extension.
        path:
            Optional filesystem path. Inspected for PDFs: one that already
            carries a text layer is left to the local parser.

        Returns
        -------
        int | None
            20 when the remote engine is configured and the MIME type is
            supported, otherwise None.
        """
        config = RemoteEngineConfig(
            engine=settings.REMOTE_OCR_ENGINE,
            api_key=settings.REMOTE_OCR_API_KEY,
            endpoint=settings.REMOTE_OCR_ENDPOINT,
        )
        if not config.engine_is_valid():
            return None
        if mime_type not in _SUPPORTED_MIME_TYPES:
            return None
        # LOCAL PATCH (2026-07-30) — upstream discussion #13264, drop when 3.1
        # lands skip-if-text upstream. 3.0.3 sends every PDF to the remote
        # engine, so born-digital PDFs get a second text layer stacked on the
        # one they already have, get billed per page and leave the network for
        # nothing. Same predicate the local pipeline uses in
        # documents.consumer.should_produce_archive.
        #
        # The gate is suspended while the force-all sentinel is live, for
        # re-OCRing scans whose existing text layer is bad. It carries its own
        # deadline so it cannot outlive its window, not even across a reboot:
        # see _force_all_active().
        #
        # Force-all never applies to vector-text PDFs. There is no bad OCR in
        # those to replace, their embedded text beats any OCR of a rendered
        # page, and the strip step in parse() would blank them.
        if mime_type == "application/pdf" and path is not None:
            # 3.0.5 moved this predicate out of documents.consumer into
            # pdf_born_digital_text(), which normalises the extracted text
            # through post_process_text() before measuring it (GH #13387):
            # raw pdftotext output can be non-empty whitespace and form-feed
            # padding with no real content behind it. Calling the wrapper
            # rather than re-deriving it keeps this gate identical to the
            # archive-generation decision, which is the whole point of it,
            # and matters here because padded OCR layers are common in this
            # archive (one document: 3,131 raw characters, 1,232 real).
            from paperless.parsers.utils import pdf_born_digital_text

            _text, has_text = pdf_born_digital_text(path, log=logger)
            if has_text:
                if not _force_all_active():
                    logger.debug(
                        "Remote OCR skipped: %s already has a text layer",
                        path.name,
                    )
                    return None
                visible, total = _pages_with_visible_text(path)
                if total and len(visible) >= total:
                    logger.info(
                        "Remote OCR skipped even under force-all: every page of "
                        "%s draws visible text, so it is born digital and OCR "
                        "cannot improve it",
                        path.name,
                    )
                    return None
        return 20

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def can_produce_archive(self) -> bool:
        """Whether this parser can produce a searchable PDF archive copy.

        Returns
        -------
        bool
            Always True — the remote engine always returns a PDF with an
            embedded text layer that serves as the archive copy.
        """
        return True

    @property
    def requires_pdf_rendition(self) -> bool:
        """Whether the parser must produce a PDF for the frontend to display.

        Returns
        -------
        bool
            Always False — all supported originals are displayable by
            the browser (PDF) or handled via the archive copy (images).
        """
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, logging_group: object = None) -> None:
        settings.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        self._tempdir = Path(
            tempfile.mkdtemp(prefix="paperless-", dir=settings.SCRATCH_DIR),
        )
        self._logging_group = logging_group
        self._text: str | None = None
        self._archive_path: Path | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger.debug("Cleaning up temporary directory %s", self._tempdir)
        shutil.rmtree(self._tempdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Core parsing interface
    # ------------------------------------------------------------------

    def configure(self, context: ParserContext) -> None:
        pass

    def parse(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool = True,
    ) -> None:
        """Send the document to the remote engine and store results.

        Parameters
        ----------
        document_path:
            Absolute path to the document file to parse.
        mime_type:
            Detected MIME type of the document.
        produce_archive:
            When False, only the text is kept and the engine's searchable
            PDF is not downloaded or stored (LOCAL PATCH — 3.0.3 upstream
            ignores this flag and always stores the archive copy).
        """
        config = RemoteEngineConfig(
            engine=settings.REMOTE_OCR_ENGINE,
            api_key=settings.REMOTE_OCR_API_KEY,
            endpoint=settings.REMOTE_OCR_ENDPOINT,
        )

        if not config.engine_is_valid():
            logger.warning(
                "No valid remote parser engine is configured, content will be empty.",
            )
            self._text = ""
            return

        if config.engine == "azureai":
            source = document_path
            original_text = ""

            # Re-OCR under force-all means REPLACING the old text layer, not
            # stacking a second one on it. Azure preserves whatever the input
            # carried, so the old layer is stripped before the file is sent,
            # and the archive copy is then produced unconditionally: embedding
            # the new layer in the PDF is the entire point of the exercise.
            splice: set[int] | None = None
            if mime_type == "application/pdf" and _force_all_active():
                from paperless.parsers.utils import extract_pdf_text

                original_text = extract_pdf_text(document_path) or ""
                if original_text:
                    visible, total = _pages_with_visible_text(document_path)
                    scanned = {p for p in range(1, total + 1) if p not in visible}
                    to_strip = document_path
                    if visible and scanned:
                        # Mixed document. Only the pages without visible text
                        # may be stripped and re-OCRed; the born-digital pages
                        # are handed straight back at the end, untouched.
                        subset = _extract_pages(
                            document_path,
                            scanned,
                            self._tempdir / "scanned-pages.pdf",
                        )
                        if subset is None:
                            self._text = original_text
                            return
                        to_strip = subset
                        splice = scanned
                        logger.info(
                            "%s is mixed: re-OCRing %d of %d pages, splicing them back",
                            document_path.name,
                            len(scanned),
                            total,
                        )

                    stripped = _strip_text_layer(to_strip, self._tempdir)
                    if stripped is None:
                        logger.error(
                            "Refusing to re-OCR %s: its old text layer could not "
                            "be stripped, and sending it as-is would leave the bad "
                            "text in the archive alongside the new text",
                            document_path.name,
                        )
                        self._text = original_text
                        return
                    source = stripped
                    produce_archive = True

            self._text = self._azure_ai_vision_parse(
                source,
                config,
                produce_archive=produce_archive,
                splice_into=(document_path, splice) if splice else None,
            )

            if splice and self._archive_path is not None:
                # Content must describe the whole document, not just the pages
                # that were sent, so read it back off the spliced archive.
                from paperless.parsers.utils import extract_pdf_text

                self._text = extract_pdf_text(self._archive_path) or self._text

            # Guard against having stripped a page the raster check misjudged:
            # if the engine found far less text than the layer we removed held,
            # keep the original text and leave the existing archive alone.
            #
            # Compare non-whitespace characters only. Text layers written by
            # OCR are heavily padded with alignment spaces (doc 7006: 3,131 raw
            # against 1,232 of actual content), so a raw length comparison
            # rejects perfectly good results.
            # The threshold is deliberately low. This guard exists to catch a
            # blanked page, where the engine comes back with essentially
            # nothing, and it must NOT fire merely because the new text is
            # shorter: a junk OCR layer is often LONGER than a correct one.
            # Two measured points separate the cases cleanly: the Kalläne
            # invoices (vector text wrongly stripped) returned 13.7% of the old
            # layer, while Passport 2018 spreads returned 47% and was a genuine
            # improvement over tesseract's noise from the security printing.
            if original_text:
                before = len("".join(original_text.split()))
                after = len("".join((self._text or "").split()))
                if after < before * 0.25:
                    logger.error(
                        "Re-OCR of %s returned %d visible chars against %d in the "
                        "layer it replaced; keeping the original text and archive copy",
                        document_path.name,
                        after,
                        before,
                    )
                    self._text = original_text
                    self._archive_path = None

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_text(self) -> str:
        """Return the plain-text content extracted during parse."""
        return self._text or ""

    def get_date(self) -> datetime.datetime | None:
        """Return the document date detected during parse.

        Returns
        -------
        datetime.datetime | None
            Always None — the remote parser does not detect dates.
        """
        return None

    def get_archive_path(self) -> Path | None:
        """Return the path to the generated archive PDF, or None."""
        return self._archive_path

    # ------------------------------------------------------------------
    # Thumbnail and metadata
    # ------------------------------------------------------------------

    def get_thumbnail(self, document_path: Path, mime_type: str) -> Path:
        """Generate a thumbnail image for the document.

        Uses the archive PDF produced by the remote engine when available,
        otherwise falls back to the original document path (PDF inputs).

        Parameters
        ----------
        document_path:
            Absolute path to the source document.
        mime_type:
            Detected MIME type of the document.

        Returns
        -------
        Path
            Path to the generated WebP thumbnail inside the temp directory.
        """
        # make_thumbnail_from_pdf lives in documents.parsers for now;
        # it will move to paperless.parsers.utils when the tesseract
        # parser is migrated in a later phase.
        from documents.parsers import make_thumbnail_from_pdf

        return make_thumbnail_from_pdf(
            self._archive_path or document_path,
            self._tempdir,
            self._logging_group,
        )

    def get_page_count(
        self,
        document_path: Path,
        mime_type: str,
    ) -> int | None:
        """Return the number of pages in a PDF document.

        Parameters
        ----------
        document_path:
            Absolute path to the source document.
        mime_type:
            Detected MIME type of the document.

        Returns
        -------
        int | None
            Page count for PDF inputs, or ``None`` for other MIME types.
        """
        if mime_type != "application/pdf":
            return None

        from paperless.parsers.utils import get_page_count_for_pdf

        return get_page_count_for_pdf(document_path, log=logger)

    def extract_metadata(
        self,
        document_path: Path,
        mime_type: str,
    ) -> list[MetadataEntry]:
        """Extract format-specific metadata from the document.

        Delegates to the shared pikepdf-based extractor for PDF files.
        Returns ``[]`` for all other MIME types.

        Parameters
        ----------
        document_path:
            Absolute path to the file to extract metadata from.
        mime_type:
            MIME type of the file.  May be ``"application/pdf"`` when
            called for the archive version of an image original.

        Returns
        -------
        list[MetadataEntry]
            Zero or more metadata entries.
        """
        if mime_type != "application/pdf":
            return []

        from paperless.parsers.utils import extract_pdf_metadata

        return extract_pdf_metadata(document_path, log=logger)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _azure_ai_vision_parse(
        self,
        file: Path,
        config: RemoteEngineConfig,
        *,
        produce_archive: bool = True,
        splice_into: tuple[Path, set[int]] | None = None,
    ) -> str | None:
        """Send ``file`` to Azure AI Document Intelligence and return text.

        Downloads the searchable PDF output from Azure, recompresses it to
        PDF/A and stores it at ``self._archive_path``.  Returns the
        extracted text content, or ``None`` on failure (the error is
        logged).

        Parameters
        ----------
        file:
            Absolute path to the document to analyse.
        config:
            Validated remote engine configuration.
        produce_archive:
            When False, skip the searchable-PDF download entirely.

        Returns
        -------
        str | None
            Extracted text, or None if the Azure call failed.
        """
        if TYPE_CHECKING:
            # Callers must have already validated config via engine_is_valid():
            # engine_is_valid() asserts api_key is not None and (for azureai)
            # endpoint is not None, so these casts are provably safe.
            assert config.endpoint is not None
            assert config.api_key is not None

        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        from azure.ai.documentintelligence.models import AnalyzeOutputOption
        from azure.ai.documentintelligence.models import DocumentContentFormat
        from azure.core.credentials import AzureKeyCredential

        client = DocumentIntelligenceClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.api_key),
        )

        try:
            with file.open("rb") as f:
                analyze_request = AnalyzeDocumentRequest(bytes_source=f.read())
                poller = client.begin_analyze_document(
                    model_id="prebuilt-read",
                    body=analyze_request,
                    output_content_format=DocumentContentFormat.TEXT,
                    output=[AnalyzeOutputOption.PDF],
                    content_type="application/json",
                )

            poller.wait()
            result_id = poller.details["operation_id"]
            result = poller.result()

            if produce_archive:
                raw = self._tempdir / "azure.pdf"
                with raw.open("wb") as f:
                    for chunk in client.get_analyze_result_pdf(
                        model_id="prebuilt-read",
                        result_id=result_id,
                    ):
                        f.write(chunk)

                if splice_into is not None:
                    original, pages = splice_into
                    merged = _splice_pages(
                        original,
                        raw,
                        pages,
                        self._tempdir / "spliced.pdf",
                    )
                    if merged is None:
                        # Splicing failed: an archive holding only the scanned
                        # pages would silently lose the rest of the document.
                        self._archive_path = None
                        return result.content
                    raw = merged

                self._archive_path = self._optimize_archive(raw)

            return result.content

        except Exception as e:
            logger.exception("Azure AI Vision parsing failed: %s", e)

        finally:
            client.close()

        return None

    def _optimize_archive(self, raw: Path) -> Path:
        """Recompress the engine's PDF the way the local pipeline would.

        LOCAL PATCH (2026-07-30) — upstream discussion #13264, drop when 3.1
        fixes this.  3.0.3 stores the engine's PDF verbatim, so it bypasses
        ``PAPERLESS_OCR_OUTPUT_TYPE`` and every optimisation the tesseract
        parser applies: the archive copy lands at 100%+ of the original
        instead of the ~35% ocrmypdf reaches on scans here.  ``skip_text``
        keeps the engine's text layer and stops ocrmypdf re-OCRing pages
        that already have one.  Falls back to the unoptimised PDF rather
        than losing the archive copy.

        Parameters
        ----------
        raw:
            Path to the PDF as downloaded from the remote engine.

        Returns
        -------
        Path
            Path to the optimised PDF, or ``raw`` if optimisation failed.
        """
        import ocrmypdf

        from paperless.config import OcrConfig

        config = OcrConfig()
        out = self._tempdir / "archive.pdf"
        args = {
            "skip_text": True,
            "output_type": config.output_type,
            "language": config.language,
            "progress_bar": False,
            "use_threads": True,
            "jobs": settings.THREADS_PER_WORKER,
        }
        if "pdfa" in config.output_type:
            args["color_conversion_strategy"] = config.color_conversion_strategy

        try:
            ocrmypdf.ocr(raw, out, **args)
        except Exception:
            logger.warning(
                "Could not optimise the remote archive copy, storing it as-is",
                exc_info=True,
            )
            return raw

        logger.info(
            "Optimised remote archive copy: %d -> %d bytes",
            raw.stat().st_size,
            out.stat().st_size,
        )
        return self._stamp_provenance(out)

    def _stamp_provenance(self, path: Path) -> Path:
        """Record in the PDF that its text layer came from the remote engine.

        Azure stamps nothing identifying on its output: it passes the input's
        metadata straight through (measured on doc 7006, which came back
        claiming ``OmniPage CSDK 21``, the scanner software). Meanwhile the
        ocrmypdf pass writes its own ``CreatorTool``, so an Azure-OCRed archive
        ends up labelled *OCRmyPDF + Tesseract OCR* — false, and impossible to
        tell apart from a genuinely local one.

        Only the standard ``xmp:CreatorTool`` field is touched, which is what
        paperless shows under archived-document metadata. Custom fields would
        need an extension schema to keep PDF/A valid; a standard one does not.
        Title, description and the rest are left exactly as they came.
        """
        import ocrmypdf
        import pikepdf

        engine = settings.REMOTE_OCR_ENGINE or "remote"
        stamp = (
            f"Azure AI Document Intelligence (prebuilt-read, {engine}) "
            f"via OCRmyPDF {ocrmypdf.__version__}"
        )
        try:
            with pikepdf.open(path, allow_overwriting_input=True) as pdf:
                with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
                    meta["xmp:CreatorTool"] = stamp
                pdf.save(path)
        except Exception:
            logger.warning(
                "Could not stamp remote-OCR provenance on the archive copy",
                exc_info=True,
            )
        return path
