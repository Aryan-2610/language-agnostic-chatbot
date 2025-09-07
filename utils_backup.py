import os
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

# Extract text from normal PDFs
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except:
        return ""  # If fails, fallback to OCR

# OCR for scanned/image-based PDFs
def extract_text_with_ocr(file_path):
    images = convert_from_path(file_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="eng")
    return text.strip()

# Fallback function (try normal, else OCR)
def extract_text(file_path):
    text = extract_text_from_pdf(file_path)
    if not text:  # If empty, use OCR
        text = extract_text_with_ocr(file_path)
    return text
