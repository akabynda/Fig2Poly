from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import zipfile

import requests


@dataclass(frozen=True)
class Asset:
    dataset: str
    split: str
    filename: str
    url: str
    size: int


ADOBE_BASE = "https://github.com/adobe-research/CHART-Synthetic/releases/download/v1.0"
GDRIVE_BASE = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="
ASSETS = [
    Asset(
        "ub_pmc22",
        "train",
        "ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0.zip",
        "https://www.dropbox.com/scl/fi/2c2yjerpiv5778j0xyi4u/"
        "ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0.zip"
        "?rlkey=hyrmt6wyac4s1zt5vm55htc70&dl=1",
        1_065_394_038,
    ),
    Asset(
        "ub_pmc22",
        "test",
        "ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1.zip",
        "https://www.dropbox.com/scl/fi/yz32crmv92hsi571vw3a5/"
        "ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1.zip"
        "?rlkey=hxmysq1hm8szyhjic1ts5v1pa&dl=1",
        782_063_690,
    ),
    Asset("lineex", "val", "val.zip", GDRIVE_BASE + "1FhYMa4i7IQhuwCjZUkolB8IPbeJxK7d1", 492_782_492),
    Asset("lineex", "test", "test.zip", GDRIVE_BASE + "1zkWSzbOFIPORtblCeAokPCQDdKTqHvm0", 988_531_668),
    Asset("lineex", "train", "train.zip", GDRIVE_BASE + "1fORnx1MH0V_BLiBAoGEuZtWSJbdk5DGI", 21_465_482_263),
    Asset("adobe_synth19", "train", "train_json_gt.tar.gz", f"{ADOBE_BASE}/train_json_gt.tar.gz", 493_594_325),
    Asset("adobe_synth19", "test", "test_tasks.tar.gz", f"{ADOBE_BASE}/test_tasks.tar.gz", 303_097_860),
    Asset("adobe_synth19", "all", "images_a.tar.gz", f"{ADOBE_BASE}/images_a.tar.gz", 1_142_211_746),
    Asset("adobe_synth19", "all", "images_b.tar.gz", f"{ADOBE_BASE}/images_b.tar.gz", 1_142_701_857),
    Asset("adobe_synth19", "all", "images_c.tar.gz", f"{ADOBE_BASE}/images_c.tar.gz", 1_133_460_378),
    Asset("adobe_synth19", "all", "images_d.tar.gz", f"{ADOBE_BASE}/images_d.tar.gz", 1_137_815_072),
    Asset("adobe_synth19", "all", "images_e.tar.gz", f"{ADOBE_BASE}/images_e.tar.gz", 1_133_201_075),
    Asset("adobe_synth19", "all", "images_f.tar.gz", f"{ADOBE_BASE}/images_f.tar.gz", 1_136_737_519),
    Asset("adobe_synth19", "all", "images_g.tar.gz", f"{ADOBE_BASE}/images_g.tar.gz", 1_133_364_635),
    Asset("adobe_synth19", "all", "images_h.tar.gz", f"{ADOBE_BASE}/images_h.tar.gz", 1_131_473_210),
    Asset("adobe_synth19", "all", "images_i.tar.gz", f"{ADOBE_BASE}/images_i.tar.gz", 1_136_732_041),
    Asset("adobe_synth19", "all", "images_j.tar.gz", f"{ADOBE_BASE}/images_j.tar.gz", 1_024_471_154),
]


def log(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: Asset, destination: Path, log_path: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file() and destination.stat().st_size == asset.size:
        log(log_path, f"reuse {asset.dataset}/{asset.filename} ({asset.size} bytes)")
        return destination
    if destination.exists():
        destination.unlink()

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Fig2Poly-public-benchmark-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with requests.get(asset.url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        mode = "ab" if offset and response.status_code == 206 else "wb"
        received = offset
        last_report = time.time()
        with partial.open(mode) as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                received += len(chunk)
                if time.time() - last_report >= 15:
                    log(
                        log_path,
                        f"download {asset.dataset}/{asset.filename}: "
                        f"{received / 2**30:.2f}/{asset.size / 2**30:.2f} GiB",
                    )
                    last_report = time.time()
    if partial.stat().st_size != asset.size:
        raise RuntimeError(
            f"Size mismatch for {asset.filename}: {partial.stat().st_size} != {asset.size}"
        )
    os.replace(partial, destination)
    return destination


def safe_extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as stream:
            stream.extractall(output, filter="data")
    elif archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as stream:
            base = output.resolve()
            for member in stream.infolist():
                target = (base / member.filename).resolve()
                if base != target and base not in target.parents:
                    raise RuntimeError(f"Unsafe archive member: {member.filename}")
            stream.extractall(output)
    else:
        raise ValueError(f"Unsupported archive: {archive}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/public")
    parser.add_argument("--datasets", default="ub_pmc22,lineex,adobe_synth19")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--delete-archives", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "download.log"
    selected = {item.strip() for item in args.datasets.split(",") if item.strip()}
    assets = [asset for asset in ASSETS if asset.dataset in selected]
    manifest = {"started": time.time(), "assets": []}
    for asset in assets:
        archive = root / "_archives" / asset.dataset / asset.filename
        log(log_path, f"start {asset.dataset}/{asset.filename}")
        downloaded = download(asset, archive, log_path)
        digest = sha256(downloaded)
        entry = {**asdict(asset), "sha256": digest, "archive": str(downloaded)}
        if args.extract:
            extract_root = root / "raw" / asset.dataset
            safe_extract(downloaded, extract_root)
            entry["extracted_to"] = str(extract_root)
            if args.delete_archives:
                downloaded.unlink()
                entry["archive_deleted_after_extract"] = True
        manifest["assets"].append(entry)
        manifest["updated"] = time.time()
        (root / "download_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(log_path, f"complete {asset.dataset}/{asset.filename} sha256={digest}")
    manifest["completed"] = time.time()
    (root / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
