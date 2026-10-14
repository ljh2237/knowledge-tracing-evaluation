"""Reproduce integration evidence; every subprocess exit status is checked."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_sample_data import generate_sample_data  # noqa: E402
from io_utils import sha256_file, write_csv, write_json  # noqa: E402

import pandas as pd  # noqa: E402


def main():
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    checks = []
    logs = []

    def command(name, argv, expected_code=0):
        result = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True, timeout=180)
        logs.append(f"## {name}\n{' '.join(argv)}\n退出码 {result.returncode}\n{result.stdout}{result.stderr}")
        if result.returncode != expected_code:
            raise RuntimeError(f"{name} 失败: {result.stdout}{result.stderr}")
        checks.append({"name": name, "exit_code": result.returncode, "expected_exit_code": expected_code, "passed": True})

    command("模拟主程序", ["main.py", "--output-dir", str(evidence / "mock_run"), "--overwrite"])
    command("插件模型", ["main.py", "--model", "plugin", "--factory", "examples.constant_plugin:create_model",
                         "--model-config", "examples/plugin_config.json", "--vocab", str(evidence / "mock_run/skill_mapping.json"),
                         "--output-dir", str(evidence / "plugin_run"), "--overwrite"])
    with tempfile.TemporaryDirectory(prefix="kt-missing-model-") as temporary:
        absent = Path(temporary) / "should_not_exist"
        command("未接真实模型明确失败", ["main.py", "--model", "plugin", "--factory", "examples.training_model_template:create_model",
                                    "--vocab", str(evidence / "mock_run/skill_mapping.json"), "--output-dir", str(absent)], 2)
        assert not absent.exists()

    command("ASSISTments 转换", ["dataset_converters.py", "--dataset", "assistments", "--input", "examples/assistments_fixture.csv",
                               "--output", str(evidence / "assistments.csv")])
    command("ASSISTments 推理", ["main.py", "--input", str(evidence / "assistments.csv"),
                               "--output-dir", str(evidence / "assistments_run"), "--overwrite"])
    command("EdNet 转换", ["dataset_converters.py", "--dataset", "ednet", "--input", "examples/u1.csv",
                           "--questions", "examples/questions_fixture.csv", "--output", str(evidence / "ednet.csv")])
    command("EdNet 推理", ["main.py", "--input", str(evidence / "ednet.csv"),
                          "--output-dir", str(evidence / "ednet_run"), "--overwrite"])
    invalid = pd.read_csv(ROOT / "sample_data.csv", dtype=str, keep_default_na=False)
    invalid.loc[0, "correct"] = ""
    invalid.loc[1, "timestamp"] = "bad-time"
    invalid.loc[2, "response_time_ms"] = "-1"
    write_csv(evidence / "invalid_input.csv", invalid)
    with tempfile.TemporaryDirectory(prefix="kt-invalid-") as temporary:
        strict_out = Path(temporary) / "strict"
        command("异常输入严格失败", ["main.py", "--input", str(evidence / "invalid_input.csv"),
                                   "--output-dir", str(strict_out)], 2)
        assert not strict_out.exists()
    command("异常输入拒收并继续", ["main.py", "--input", str(evidence / "invalid_input.csv"), "--invalid-policy", "drop",
                                  "--output-dir", str(evidence / "drop_run"), "--overwrite"])
    generate_sample_data(evidence / "stress_input.csv", students=100, events=100, seed=77)
    command("一万条模拟记录", ["main.py", "--input", str(evidence / "stress_input.csv"),
                               "--output-dir", str(evidence / "stress_run"), "--overwrite"])
    summaries = {name: json.loads((evidence / name / "run_summary.json").read_text(encoding="utf-8"))
                 for name in ["mock_run", "plugin_run", "assistments_run", "ednet_run", "drop_run", "stress_run"]}
    expected_counts = {"mock_run": 288, "plugin_run": 288, "assistments_run": 4,
                       "ednet_run": 3, "drop_run": 285, "stress_run": 10000}
    for name, expected_count in expected_counts.items():
        assert summaries[name]["prediction_count"] == expected_count
        assert summaries[name]["is_mock"] is True
        for file_name, fingerprint in summaries[name]["artifacts"].items():
            assert sha256_file(evidence / name / file_name) == fingerprint
    assert summaries["drop_run"]["data"]["rejected_rows"] == 3
    write_json(evidence / "acceptance.json", {"checks": checks, "runs": summaries, "all_passed": True,
                                             "data_type": "synthetic_and_handcrafted_fixtures"})
    (evidence / "acceptance.log").write_text("\n\n".join(logs), encoding="utf-8")
    print(f"集成验收通过：{len(checks)} 个命令检查；结果位于 {evidence}")


if __name__ == "__main__":
    main()
