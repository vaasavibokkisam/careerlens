import PyPDF2
import io


def parse_resume(uploaded_file) -> str:
    """Extract text from an uploaded PDF resume."""
    pdf_bytes = uploaded_file.read()
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text.strip())

    full_text = "\n\n".join(text_parts)

    # Basic cleanup
    full_text = " ".join(full_text.split())
    return full_text
