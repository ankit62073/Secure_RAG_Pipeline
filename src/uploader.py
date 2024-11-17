import shutil
from pathlib import Path
from typing import List
import fitz  # PyMuPDF

from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.config import Config

from logger.logging import logging

class InvalidPDFException(Exception):
    pass

def is_pdf_with_text(file_path: Path) -> bool:
    """Check if a PDF file contains any text."""
    with fitz.open(file_path) as doc:
        for page in doc:
            text = page.get_text()
            if text.strip():  # Check if there's any non-empty text
                return True
    return False

def upload_files(files: List[UploadedFile], remove_old_files: bool = True) -> List[Path]:
    if remove_old_files:
        shutil.rmtree(Config.Path.DATABASE_DIR, ignore_errors=True)
        shutil.rmtree(Config.Path.DOCUMENTS_DIR, ignore_errors=True)
    
    Config.Path.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    file_paths = []

    for file in files:
        try:
            if not file.name.lower().endswith('.pdf'):
                raise ValueError(f"File '{file.name}' is not a PDF.")
            
            file_path = Config.Path.DOCUMENTS_DIR / file.name
            with file_path.open("wb") as f:
                f.write(file.getvalue())

            if not is_pdf_with_text(file_path):
                raise InvalidPDFException(f"The PDF '{file.name}' does not contain any text.")

            file_paths.append(file_path)

        except (ValueError, InvalidPDFException) as e:
            # Handle specific exceptions
            logging.error(f"Error processing file '{file.name}': {e}")

        except Exception as e:
            # Handle any other unexpected exceptions
            logging.exception(f"An unexpected error occurred with file '{file.name}': {e}")
    
    return file_paths
