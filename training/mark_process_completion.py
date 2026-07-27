from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import time


SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102


def process_alive(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()
    while process_alive(args.pid):
        time.sleep(20)
    marker = Path(args.marker).resolve()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"process {args.pid} completed\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
