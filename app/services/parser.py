from io import BytesIO

from docx import Document
from pypdf import PdfReader

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class ResumeParseError(Exception):
    pass


def extension_of(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def parse_resume(filename: str | None, content: bytes) -> str:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ResumeParseError("File exceeds the 2 MB limit")

    extension = extension_of(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise ResumeParseError("Upload a .pdf, .docx, or .txt resume")

    if extension == ".txt":
        text = _parse_txt(content)
    elif extension == ".pdf":
        text = _parse_pdf(content)
    else:
        text = _parse_docx(content)

    cleaned = text.strip()
    if not cleaned:
        raise ResumeParseError("No text could be extracted from the resume")
    return cleaned


def _parse_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _parse_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise ResumeParseError("Could not read the PDF resume") from exc
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _parse_docx(content: bytes) -> str:
    try:
        document = Document(BytesIO(content))
    except Exception as exc:
        raise ResumeParseError("Could not read the DOCX resume") from exc
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
