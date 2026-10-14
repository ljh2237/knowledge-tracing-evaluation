import json
import subprocess
import sys
from pathlib import Path

import pytest

from main import main

ROOT = Path(__file__).resolve().parents[1]


def test_main_mock_and_model_config(tmp_path):
    output = tmp_path / "mock"
    assert main(["--output-dir", str(output)]) == 0
    assert main(["--output-dir", str(output), "--overwrite"]) == 0
    plugin = tmp_path / "plugin"
    assert main(["--model", "plugin", "--factory", "examples.constant_plugin:create_model",
                 "--vocab", str(output / "skill_mapping.json"), "--model-config",
                 str(ROOT / "examples" / "plugin_config.json"), "--output-dir", str(plugin)]) == 0
    # run_summary.json 以 UTF-8 写出；必须显式指定编码，否则在中文 Windows
    # 等非 UTF-8 默认编码的平台上解码失败。此修改与原有行为等价。
    summary = json.loads((plugin / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["is_mock"] is True


@pytest.mark.parametrize("args", [
    ["--model", "plugin"], ["--factory", "x:y"], ["--model-config", "missing.json"],
    ["--input", "missing.csv"], ["--batch-size", "0"], ["--threshold", "nan"],
])
def test_main_returns_nonzero_on_failures(tmp_path, args):
    assert main(["--output-dir", str(tmp_path / "out"), *args]) == 2
    assert not (tmp_path / "out").exists()


def test_cli_runs_from_other_working_directory(tmp_path):
    command = [sys.executable, str(ROOT / "main.py"), "--output-dir", str(tmp_path / "out")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "predictions.csv").exists()


def test_plugin_template_failure_creates_no_output(tmp_path):
    assert main(["--output-dir", str(tmp_path / "first")]) == 0
    assert main(["--model", "plugin", "--factory", "examples.training_model_template:create_model",
                 "--vocab", str(tmp_path / "first" / "skill_mapping.json"),
                 "--output-dir", str(tmp_path / "real")]) == 2
    assert not (tmp_path / "real").exists()
