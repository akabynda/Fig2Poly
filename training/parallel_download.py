"""Bounded, resumable HTTP range download for immutable released dataset assets."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable

import requests


class RangeDownloadError(RuntimeError):
    pass


def parallel_download(
    url: str,
    destination: Path,
    size: int,
    workers: int = 8,
    chunk_size: int = 16 * 1024 * 1024,
    attempts: int = 3,
    log: Callable[[str], None] | None = None,
    request_get: Callable | None = None,
    retry_delay: float = 1.0,
) -> Path:
    """Save exact validated ranges, then atomically publish the assembled archive.

    The ledger binds retained parts to the URL, expected size and range layout.
    Failed/interrupted downloads retain their parts for a subsequent invocation.
    Existing sequential ``.part`` files are left untouched.
    """
    if workers < 1 or size < 1 or chunk_size < 1 or attempts < 1:
        raise ValueError("workers, size, chunk_size and attempts must be positive")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        return destination
    parts = destination.with_name(destination.name + ".ranges")
    parts.mkdir(exist_ok=True)
    ledger = parts / "manifest.json"
    settings = {"format": "http-ranges-v1", "url": url, "size": size, "chunk_size": chunk_size}
    if ledger.is_file():
        if json.loads(ledger.read_text(encoding="utf-8")) != settings:
            raise RangeDownloadError(f"Range state belongs to another asset or layout: {parts}")
    else:
        if any(parts.iterdir()):
            raise RangeDownloadError(f"Unrecognized nonempty range directory: {parts}")
        temporary = parts / "manifest.json.tmp"
        temporary.write_text(json.dumps(settings), encoding="utf-8")
        os.replace(temporary, ledger)
    ranges = [(index, start, min(start + chunk_size, size) - 1)
              for index, start in enumerate(range(0, size, chunk_size))]
    stop = threading.Event()
    get = request_get or requests.get

    def download_range(item: tuple[int, int, int]) -> Path:
        index, start, end = item
        target = parts / f"{index:08d}.chunk"
        partial = parts / f"{index:08d}.part"
        expected = end - start + 1
        if target.is_file():
            if target.stat().st_size != expected:
                raise RangeDownloadError(f"Completed range has incorrect length: {target}")
            return target
        for attempt in range(attempts):
            if stop.is_set():
                raise RangeDownloadError("Another range failed; keeping partial downloads")
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise RangeDownloadError(f"Partial range exceeds its expected length: {partial}")
            if offset == expected:
                # A full .part lacks the successful end-of-response confirmation
                # signaled by .chunk. It may follow an oversized response or an
                # interruption just before validation: fetch that range again.
                with partial.open("wb"):
                    pass
                offset = 0
            requested_start = start + offset
            try:
                with get(
                    url,
                    headers={"Range": f"bytes={requested_start}-{end}",
                             "Accept-Encoding": "identity",
                             "User-Agent": "Fig2Poly-public-benchmark-downloader/1.0"},
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    response.raise_for_status()
                    content_range = f"bytes {requested_start}-{end}/{size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != content_range:
                        raise RangeDownloadError(
                            f"Server did not honor exact range {content_range}: "
                            f"HTTP {response.status_code}, {response.headers.get('Content-Range')}"
                        )
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise RangeDownloadError("Encoded HTTP range cannot be compared to archive byte offsets")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != expected - offset:
                        raise RangeDownloadError(f"Range Content-Length mismatch: {length} != {expected - offset}")
                    received = offset
                    with partial.open("ab") as output:
                        for data in response.iter_content(256 * 1024):
                            if stop.is_set():
                                raise RangeDownloadError("Another range failed; keeping partial downloads")
                            if not data:
                                continue
                            if received + len(data) > expected:
                                raise RangeDownloadError(f"Range body exceeds expected length: {index}")
                            output.write(data)
                            received += len(data)
                    if received != expected:
                        raise RangeDownloadError(f"Truncated range {index}: {received}/{expected} bytes")
                os.replace(partial, target)
                return target
            except (requests.RequestException, OSError, RangeDownloadError, ValueError) as error:
                if attempt + 1 == attempts:
                    raise RangeDownloadError(
                        f"Range {index} ({start}-{end}) failed after {attempts} attempts: {error}"
                    ) from error
                if log is not None:
                    log(f"retry range {index}, attempt {attempt + 2}/{attempts}: {error}")
                if retry_delay:
                    stop.wait(min((attempt + 1) * retry_delay, 2.0))
        raise AssertionError("Unreachable")

    completed = 0
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_range, item) for item in ranges]
        try:
            for future in as_completed(futures):
                future.result()
                completed += 1
                now = time.monotonic()
                if log is not None and (now - last_report >= 15 or completed == len(ranges)):
                    log(f"range download {destination.name}: {completed}/{len(ranges)} chunks complete")
                    last_report = now
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise

    assembly = destination.with_name(destination.name + ".parallel-assembling")
    with assembly.open("wb") as output:
        for index, start, end in ranges:
            chunk = parts / f"{index:08d}.chunk"
            if chunk.stat().st_size != end - start + 1:
                raise RangeDownloadError(f"Range changed before assembly: {chunk}")
            with chunk.open("rb") as stream:
                shutil.copyfileobj(stream, output, 1024 * 1024)
    if assembly.stat().st_size != size:
        raise RangeDownloadError(f"Assembled archive has incorrect length: {assembly}")
    os.replace(assembly, destination)
    # Delete only the exact range files created by this downloader, after success.
    for index, _, _ in ranges:
        (parts / f"{index:08d}.chunk").unlink()
    ledger.unlink()
    try:
        parts.rmdir()
    except OSError:
        pass  # Never remove unrelated files that appeared in the range directory.
    return destination
