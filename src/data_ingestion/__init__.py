"""Stage 1 — Data Ingestion: PDF download, validation, local persistence."""
from .downloader import download_pdf

__all__ = ["download_pdf"]
