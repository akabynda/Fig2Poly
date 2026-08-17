"""Copy files to or from aichem through the ITMO SSH gateway."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import paramiko

from remote_exec import load_credentials


def connect(env: Path, gateway_host: str, target_host: str) -> tuple[paramiko.SSHClient, paramiko.SSHClient]:
    login, password = load_credentials(env)
    gateway = paramiko.SSHClient()
    gateway.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gateway.connect(gateway_host, username=login, password=password, timeout=30)
    channel = gateway.get_transport().open_channel(
        "direct-tcpip", (target_host, 22), ("127.0.0.1", 0)
    )
    target = paramiko.SSHClient()
    target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    target.connect(target_host, username=login, password=password, sock=channel, timeout=30)
    return gateway, target


def mkdir_parents(sftp: paramiko.SFTPClient, remote_file: str) -> None:
    parts = PurePosixPath(remote_file).parent.parts
    current = ""
    for part in parts:
        current = "/" if part == "/" else f"{current.rstrip('/')}/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=("upload", "download"))
    parser.add_argument("local", type=Path)
    parser.add_argument("remote")
    parser.add_argument("--env", type=Path, default=Path(__file__).parents[1] / ".env")
    parser.add_argument("--gateway", default="aicltr.itmo.ru")
    parser.add_argument("--target", default="aichem")
    args = parser.parse_args()
    gateway, target = connect(args.env, args.gateway, args.target)
    sftp = target.open_sftp()
    if args.direction == "upload":
        mkdir_parents(sftp, args.remote)
        sftp.put(str(args.local.resolve()), args.remote)
    else:
        args.local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(args.remote, str(args.local.resolve()))
    sftp.close()
    target.close()
    gateway.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
