import io


def text_from_pdf(pdf: str | bytes | io.BytesIO):
    """
    Extract text from a PDF. Does NOT use OCR, extracts the literal text.
    The source can be:
    - A file path (str)
    - Bytes of a PDF file
    - A BytesIO object containing a PDF file
    """
    try:
        import pymupdf  # pyright: ignore
    except ImportError:
        raise ImportError(
            "pymupdf is required to extract text from PDFs. Install lm_deluge[pdf] or lm_deluge[full]."
        )
    if isinstance(pdf, str):
        # It's a file path
        doc = pymupdf.open(pdf)
    elif isinstance(pdf, (bytes, io.BytesIO)):
        # It's bytes or a BytesIO object
        if isinstance(pdf, bytes):
            pdf = io.BytesIO(pdf)
        doc = pymupdf.open(stream=pdf, filetype="pdf")
    else:
        raise ValueError("Unsupported pdf_source type. Must be str, bytes, or BytesIO.")

    text_content = []
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        for block in blocks:
            # block[4] contains the text content
            text_content.append(block[4].strip())
            text_content.append("\n")  # Add extra newlines between blocks

    # Join all text content with newlines
    full_text = "\n".join(text_content).strip()
    # Replace multiple consecutive spaces with a single space
    full_text = " ".join(full_text.split())
    # Clean up any resulting double spaces or newlines
    full_text = " ".join([x for x in full_text.split(" ") if x])
    full_text = "\n".join([x for x in full_text.split("\n") if x])

    return full_text
