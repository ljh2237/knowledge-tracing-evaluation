"""构建刘静涵模型评估模块的集成发布包，并重新解压验证。

与负责人 A 的 ``scripts/package_release.py`` 使用同一套做法（ZIP + SHA256 清单 + 打包后
校验），但有两点不同：

1. 清单写入独立的 ``release_manifest_evaluation.json``，**不覆盖**负责人 A 原有的
   ``release_manifest.json``，两套校验信息互不干扰。
2. 打包后会重新解压到临时目录，实际执行 ``main.py`` 与 ``evaluate.py``，
   确认主要功能在解压后的副本中仍可运行，而不是只检查压缩包完整。

排除项：虚拟环境、缓存、临时文件、运行结果目录（``outputs/``）。
``evidence/evaluation/`` 属于验收证据，随包交付。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts import VERSION  # noqa: E402

EVALUATION_VERSION = "1.0.0"
ARCHIVE_ROOT = "knowledge_tracing_test"

EXCLUDED_PARTS = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "outputs", ".git", "dist", ".idea", ".vscode",
}
EXCLUDED_NAMES = {".coverage", "release_manifest.json", "release_manifest_evaluation.json"}


def collect_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.name not in EXCLUDED_NAMES
        and path.suffix not in {".pyc", ".pyo"}
    )


def build_manifest(files: list[Path]) -> dict:
    return {
        "package": f"刘静涵_知识追踪模型评估模块_v{EVALUATION_VERSION}",
        "evaluation_version": EVALUATION_VERSION,
        "framework_version": VERSION,
        "root_directory": ARCHIVE_ROOT,
        "files": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "excluded_directories": sorted(EXCLUDED_PARTS),
        "excluded_files": sorted(EXCLUDED_NAMES),
        "runtime_note": (
            "需要 Python 3.12+（与原框架 pyproject.toml 一致，本次在 3.11.9 上验证）。"
            "虚拟环境不随 ZIP 分发，请按 README 安装锁定依赖；"
            "评估模块额外需要 scikit-learn 与 matplotlib（requirements-lock-evaluation.txt）。"
        ),
        "scope_note": (
            "evidence/ 与 evidence/evaluation/ 中的预测与指标全部来自负责人 A 的合成数据与"
            "模拟模型（is_mock=True），只能证明程序与接口正确；"
            "真实 DKT/AKT 精度、真实遗忘机制效果与鸿蒙端侧性能尚未验证。"
        ),
    }


def run(command: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *command], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )


def verify_extracted(extracted: Path, checks: list[dict]) -> None:
    """在解压后的副本中实际运行主程序与评估程序。"""
    project = extracted / ARCHIVE_ROOT
    if not (project / "main.py").is_file():
        raise RuntimeError(f"解压后缺少 main.py: {project}")

    result = run(["main.py", "--output-dir", "outputs/release_check", "--overwrite"], project)
    if result.returncode != 0:
        raise RuntimeError(f"解压后 main.py 运行失败：{result.stdout}{result.stderr}")
    predictions = project / "outputs" / "release_check" / "predictions.csv"
    if not predictions.is_file():
        raise RuntimeError("解压后 main.py 未生成 predictions.csv")
    row_count = sum(1 for _ in predictions.open(encoding="utf-8")) - 1
    if row_count != 288:
        raise RuntimeError(f"解压后 main.py 的预测行数为 {row_count}，应为 288")
    checks.append({"name": "解压后运行 main.py", "exit_code": result.returncode,
                   "observed": f"{row_count} 条预测", "passed": True})

    result = run(
        ["evaluate.py", "metrics", "--predictions", "evidence/mock_run",
         "--output-dir", "outputs/release_check_eval", "--overwrite"],
        project,
    )
    if result.returncode != 0:
        raise RuntimeError(f"解压后 evaluate.py 运行失败：{result.stdout}{result.stderr}")
    metrics_json = project / "outputs" / "release_check_eval" / "metrics" / "metrics.json"
    if not metrics_json.is_file():
        raise RuntimeError("解压后 evaluate.py 未生成 metrics.json")
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    if payload["inputs"][0]["sample_count"] != 288:
        raise RuntimeError("解压后 evaluate.py 的样本数与预期不符")
    checks.append({"name": "解压后运行 evaluate.py", "exit_code": result.returncode,
                   "observed": f"{len(payload['rows'])} 行指标结果", "passed": True})

    result = run(["evaluate.py", "report", "--evaluation-dir", "outputs/release_check_eval",
                  "--output", "outputs/release_check_eval/reports/r.md", "--overwrite"], project)
    if result.returncode != 0 or not (
        project / "outputs" / "release_check_eval" / "reports" / "r.md"
    ).is_file():
        raise RuntimeError(f"解压后报告生成失败：{result.stdout}{result.stderr}")
    checks.append({"name": "解压后运行 evaluate.py report", "exit_code": result.returncode,
                   "observed": "报告已生成", "passed": True})

    for relative in (
        "docs/EVALUATION_TEST_REPORT.md",
        "docs/EVALUATION_INTERFACE.md",
        "docs/EVALUATION_GUIDE.md",
        "docs/EVALUATION_DELIVERY_CHECKLIST.md",
        "evidence/evaluation/acceptance_summary.json",
        "evidence/evaluation/metrics/metrics.csv",
        "evidence/evaluation/plots/auc_comparison.png",
        "requirements-lock-evaluation.txt",
    ):
        if not (project / relative).is_file():
            raise RuntimeError(f"解压后缺少交付文件：{relative}")
    checks.append({"name": "交付文件完整性", "exit_code": 0,
                   "observed": "文档、证据、依赖清单齐备", "passed": True})


def main() -> int:
    parser = argparse.ArgumentParser(description="构建并验证刘静涵模型评估模块发布包")
    parser.add_argument("--destination", type=Path, default=ROOT.parent / "交付版本")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    output = args.destination / f"刘静涵_知识追踪模型评估模块_v{EVALUATION_VERSION}.zip"
    files = collect_files()
    if not files:
        raise RuntimeError("没有可打包的文件")
    manifest = build_manifest(files)

    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, f"{ARCHIVE_ROOT}/{path.relative_to(ROOT).as_posix()}")
        archive.writestr(
            f"{ARCHIVE_ROOT}/release_manifest_evaluation.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

    checks: list[dict] = []
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP 内容校验失败")
        for name, fingerprint in manifest["files"].items():
            actual = hashlib.sha256(archive.read(f"{ARCHIVE_ROOT}/{name}")).hexdigest()
            if actual != fingerprint:
                raise RuntimeError(f"打包后 SHA256 不一致：{name}")
        checks.append({"name": "ZIP 完整性与文件指纹", "exit_code": 0,
                       "observed": f"{len(manifest['files'])} 个文件全部匹配", "passed": True})

        with tempfile.TemporaryDirectory(prefix="kt-release-verify-") as temporary:
            archive.extractall(temporary)
            verify_extracted(Path(temporary), checks)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    report = output.with_name("release_verification_evaluation.json")
    report.write_text(json.dumps({
        "package": output.name,
        "archive_sha256": digest,
        "file_count": len(manifest["files"]),
        "size_bytes": output.stat().st_size,
        "checks": checks,
        "all_passed": all(item["passed"] for item in checks),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"发布包已校验：{output}")
    print(f"  {len(manifest['files'])} 个文件，{output.stat().st_size} 字节")
    print(f"  解压验证 {len(checks)} 项全部通过：{[item['name'] for item in checks]}")
    print(f"  SHA256：{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
