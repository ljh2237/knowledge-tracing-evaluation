"""Build a portable source ZIP with verified file fingerprints, without venv."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts import VERSION  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT.parent / "交付版本")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    output = args.destination / f"鲁凯宁_知识追踪测试框架_v{VERSION}.zip"
    excluded = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs", ".git", "dist"}
    files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and not excluded.intersection(path.relative_to(ROOT).parts)
                   and path.name not in {".coverage", "release_manifest.json"})
    manifest = {
        "version": VERSION, "root_directory": ROOT.name,
        "files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "excluded": sorted(excluded),
        "runtime_note": "需要 Python 3.12+；虚拟环境不随 ZIP 分发，请按 README 安装锁定依赖。",
    }
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, f"{ROOT.name}/{path.relative_to(ROOT)}")
        archive.writestr(f"{ROOT.name}/release_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, fingerprint in manifest["files"].items():
            assert hashlib.sha256(archive.read(f"{ROOT.name}/{name}")).hexdigest() == fingerprint
    fingerprint = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{fingerprint}  {output.name}\n", encoding="utf-8")
    print(f"发布包已校验：{output}；{len(files)} 个文件；{output.stat().st_size} 字节")


if __name__ == "__main__":
    main()
