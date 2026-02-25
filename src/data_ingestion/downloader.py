"""
Stage 1 — Data Ingestion: downloader.py
Async PDF download via httpx with configurable timeout from config.yaml.
"""
import os
import shutil
import asyncio

import httpx
import yaml


def _get_timeout() -> int:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return int(cfg.get("concurrency", {}).get("http_download_timeout", 30))
    except Exception:
        return 30


async def download_pdf(pdf_url: str, dest_path: str, tmpdir: str, filename: str) -> None:
    """
    Async download of a PDF from ``pdf_url`` and persist to ``dest_path``.
    Also copies the file into ``tmpdir/filename`` for the processing pipeline.

    Parameters
    ----------
    pdf_url   : Remote URL of the PDF.
    dest_path : Permanent storage path  (e.g. stored_pdfs/input_<ts>.pdf).
    tmpdir    : Temp directory fed into the pipeline for processing.
    filename  : Filename inside tmpdir.

    Raises
    ------
    httpx.HTTPStatusError  if the server returns a non-2xx response.
    """
    timeout = _get_timeout()

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(pdf_url)
        response.raise_for_status()

    content = response.content

    def _write():
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(content)
        shutil.copy2(dest_path, os.path.join(tmpdir, filename))

    await asyncio.to_thread(_write)
