from pathlib import Path
import re

import pytest

from training.parallel_download import RangeDownloadError, parallel_download


class Response:
    def __init__(self, body, start, end, total, *, status=206, content_range=None,
                 length=None, fragments=None):
        self.status_code = status
        self.headers = {"Content-Range": content_range or f"bytes {start}-{end}/{total}",
                        "Content-Length": str(length if length is not None else end - start + 1)}
        self.fragments = fragments if fragments is not None else [body]

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _):
        yield from self.fragments


class FakeHTTP:
    def __init__(self, payload, fault=None):
        self.payload = payload
        self.fault = fault
        self.calls = []

    def __call__(self, url, *, headers, stream, timeout):
        assert headers["Accept-Encoding"] == "identity"
        assert stream and timeout == (30, 120)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
        start, end = map(int, match.groups())
        self.calls.append((start, end))
        body = self.payload[start:end + 1]
        if self.fault == "truncated":
            return Response(body[:4], start, end, len(self.payload))
        if self.fault == "wrong-range":
            return Response(body, start, end, len(self.payload), content_range=f"bytes {start + 1}-{end}/{len(self.payload)}")
        if self.fault == "status-200":
            return Response(body, start, end, len(self.payload), status=200)
        if self.fault == "wrong-length":
            return Response(body, start, end, len(self.payload), length=len(body) + 1)
        if self.fault == "overlong":
            return Response(body, start, end, len(self.payload), fragments=[body, b"!"])
        return Response(body, start, end, len(self.payload))


def test_parallel_ranges_assemble_in_order_and_keep_sequential_partial(tmp_path: Path) -> None:
    payload = bytes(range(101))
    http = FakeHTTP(payload)
    target = tmp_path / "asset.tar.gz"
    old_partial = tmp_path / "asset.tar.gz.part"
    old_partial.write_bytes(b"previous sequential progress")
    result = parallel_download("https://example.test/asset", target, len(payload),
                               workers=4, chunk_size=16, request_get=http, retry_delay=0)
    assert result.read_bytes() == payload
    assert sorted(http.calls) == [(0, 15), (16, 31), (32, 47), (48, 63),
                                  (64, 79), (80, 95), (96, 100)]
    assert not (tmp_path / "asset.tar.gz.ranges").exists()
    assert old_partial.read_bytes() == b"previous sequential progress"
    # A complete archive is reused without another HTTP request.
    http.calls.clear()
    assert parallel_download("https://example.test/asset", target, len(payload), request_get=http) == target
    assert not http.calls


def test_truncated_range_resumes_across_invocations(tmp_path: Path) -> None:
    target = tmp_path / "asset.bin"
    payload = b"abcdefgh"
    failing = FakeHTTP(payload, "truncated")
    with pytest.raises(RangeDownloadError, match="Truncated range"):
        parallel_download("https://example.test/asset", target, len(payload),
                          workers=1, chunk_size=8, attempts=1, request_get=failing, retry_delay=0)
    assert not target.exists()
    assert (tmp_path / "asset.bin.ranges/00000000.part").read_bytes() == b"abcd"
    resumed = FakeHTTP(payload)
    parallel_download("https://example.test/asset", target, len(payload), workers=2,
                      chunk_size=8, request_get=resumed, retry_delay=0)
    assert resumed.calls == [(4, 7)]
    assert target.read_bytes() == payload


@pytest.mark.parametrize("fault", ["wrong-range", "status-200", "wrong-length", "overlong"])
def test_invalid_http_ranges_never_publish_and_retries_are_bounded(tmp_path: Path, fault: str) -> None:
    target = tmp_path / "asset.bin"
    target.write_bytes(b"old")
    http = FakeHTTP(b"abcdefgh", fault)
    with pytest.raises(RangeDownloadError, match="failed after 3 attempts"):
        parallel_download("https://example.test/asset", target, 8, workers=2,
                          chunk_size=8, attempts=3, request_get=http, retry_delay=0)
    assert len(http.calls) == 3
    assert target.read_bytes() == b"old"
    assert not (tmp_path / "asset.bin.ranges/00000000.chunk").exists()


def test_resume_rejects_parts_from_another_asset(tmp_path: Path) -> None:
    target = tmp_path / "asset.bin"
    http = FakeHTTP(b"abcdefgh", "truncated")
    with pytest.raises(RangeDownloadError):
        parallel_download("https://example.test/old", target, 8, workers=1,
                          chunk_size=8, attempts=1, request_get=http, retry_delay=0)
    with pytest.raises(RangeDownloadError, match="another asset or layout"):
        parallel_download("https://example.test/new", target, 8, workers=1,
                          chunk_size=8, request_get=http, retry_delay=0)
