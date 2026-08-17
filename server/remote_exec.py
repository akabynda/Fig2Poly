"""Run a command on the aichem login node through the ITMO gateway.

Credentials are read from the repository's ignored .env file and are never
printed. The caller supplies the command to execute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import paramiko


def load_credentials(path: Path) -> tuple[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values["LOGIN"], values["PASSWORD"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="+")
    parser.add_argument("--env", type=Path, default=Path(__file__).parents[1] / ".env")
    parser.add_argument("--gateway", default="aicltr.itmo.ru")
    parser.add_argument("--target", default="aichem")
    args = parser.parse_args()

    login, password = load_credentials(args.env)
    gateway = paramiko.SSHClient()
    gateway.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gateway.connect(args.gateway, username=login, password=password, timeout=30)
    channel = gateway.get_transport().open_channel(
        "direct-tcpip", (args.target, 22), ("127.0.0.1", 0)
    )
    target = paramiko.SSHClient()
    target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    target.connect(args.target, username=login, password=password, sock=channel, timeout=30)
    _, stdout, stderr = target.exec_command(" ".join(args.command))
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out:
        print(out, end="")
    if err:
        print(err, end="", file=__import__("sys").stderr)
    status = stdout.channel.recv_exit_status()
    target.close()
    gateway.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
