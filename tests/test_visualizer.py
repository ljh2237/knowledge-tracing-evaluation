"""可视化模块测试：图表生成、数据驱动与无法计算时的跳过行为。"""
from __future__ import annotations

import json
import struct

import numpy as np
import pandas as pd
import pytest

from evaluation.common import EvaluationError, load_predictions
from evaluation.visualizer import (
    DEMO_TEXT,
    PlotInputs,
    configure_style,
    generate_plots,
    load_artifacts,
    plot_metric_comparison,
    plot_model_time_response,
    plot_prediction_distribution,
    plot_roc_curves,
    plot_throughput,
    plot_time_response,
    require_plot_inputs,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_size(path):
    """读取 PNG 的 IHDR 尺寸，确认输出是真实可用的图片而不是空文件。"""
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def prediction_file(path, labels, probabilities, *, label="model", is_mock="True"):
    frame = pd.DataFrame({
        "record_id": [f"r{index}" for index in range(len(labels))],
        "student_id": "S1", "question_id": "Q1", "skill_id": "K01", "time_source": "timestamp",
        "y_true": labels, "y_pred": probabilities,
        "predicted_label": [int(value >= 0.5) for value in probabilities],
        "history_length": 1, "cold_start": "False", "model_name": label, "is_mock": is_mock,
    })
    frame.to_csv(path, index=False)
    return path


def metrics_payload(labels, auc_values, *, is_mock=True):
    return {
        "rows": [
            {"group": "all", "label": label, "roc_auc": auc, "accuracy": auc, "log_loss": 1 - auc,
             "is_mock": is_mock, "auc_status": "computed", "auc_reason": None}
            for label, auc in zip(labels, auc_values, strict=False)
        ]
    }


@pytest.fixture(autouse=True)
def prefer_english_labels(monkeypatch):
    """强制英文标签，让断言不依赖运行环境是否安装中文字体。"""
    monkeypatch.setattr("evaluation.visualizer._STATE", {"lang": "en"})
    yield
    configure_style.cache_clear()


def test_configure_style_returns_supported_language():
    configure_style.cache_clear()
    assert configure_style() in {"zh", "en"}


def test_metric_comparison_plot_is_created(tmp_path):
    output = tmp_path / "auc.png"
    result = plot_metric_comparison(
        [{"label": f"m{index}", "value": 0.5 + index / 10, "is_mock": True}
         for index in range(3)],
        "roc_auc", output, title_key="auc_title", y_key="y_auc", source="test",
    )
    assert result.status == "created"
    assert result.demo is True
    assert png_size(output)[0] > 400


def test_metric_comparison_is_skipped_without_values(tmp_path):
    result = plot_metric_comparison(
        [{"label": "m", "value": None, "is_mock": True, "auc_status": "undefined_single_class",
          "auc_reason": "只有正样本"}],
        "roc_auc", tmp_path / "x.png", title_key="auc_title", y_key="y_auc", source="test",
    )
    assert result.status == "skipped"
    assert "只有正样本" in result.reason
    assert not (tmp_path / "x.png").exists()


def test_roc_curves_skip_single_class_models(tmp_path):
    good = load_predictions(prediction_file(
        tmp_path / "good.csv", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    ))
    single = load_predictions(prediction_file(
        tmp_path / "single.csv", [1, 1, 1, 1], [0.1, 0.2, 0.8, 0.9], label="single_class"
    ))
    output = tmp_path / "roc.png"
    result = plot_roc_curves([good, single], output)
    assert result.status == "created"
    assert png_size(output)[0] > 400


def test_roc_curves_skipped_when_no_model_is_plottable(tmp_path):
    single = load_predictions(prediction_file(
        tmp_path / "single.csv", [1, 1, 1, 1], [0.1, 0.2, 0.8, 0.9]
    ))
    result = plot_roc_curves([single], tmp_path / "roc.png")
    assert result.status == "skipped"
    assert "单一类别" in result.reason


def test_roc_curves_skipped_without_sources(tmp_path):
    result = plot_roc_curves([], tmp_path / "roc.png")
    assert result.status == "skipped"
    assert "未提供预测文件" in result.reason


def test_prediction_distribution_plot_is_created(tmp_path):
    labels, probabilities = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    first = load_predictions(prediction_file(tmp_path / "a.csv", labels, probabilities, label="a"))
    second = load_predictions(
        prediction_file(tmp_path / "b.csv", labels, [0.4, 0.3, 0.6, 0.7], label="b")
    )
    output = tmp_path / "dist.png"
    result = plot_prediction_distribution([first, second], output)
    assert result.status == "created"
    assert result.demo is True
    assert png_size(output)[0] > 600  # 双面板图，宽度应明显大于单面板


def test_prediction_distribution_handles_single_class_labels(tmp_path):
    """某一类标签缺失时不应报错，只跳过对应的一组直方图。"""
    data = load_predictions(prediction_file(tmp_path / "s.csv", [1, 1, 1], [0.2, 0.6, 0.9]))
    output = tmp_path / "dist.png"
    assert plot_prediction_distribution([data], output).status == "created"
    assert png_size(output)[0] > 600


def test_prediction_distribution_skipped_without_sources(tmp_path):
    result = plot_prediction_distribution([], tmp_path / "dist.png")
    assert result.status == "skipped"
    assert "未提供预测文件" in result.reason


def test_prediction_distribution_does_not_hardcode_values(tmp_path):
    """不同输入必须产出不同像素，证明分布来自数据。"""
    labels = [0, 0, 1, 1]
    digests = []
    for index, probabilities in enumerate(([0.1, 0.2, 0.8, 0.9], [0.45, 0.5, 0.55, 0.6])):
        data = load_predictions(
            prediction_file(tmp_path / f"d{index}.csv", labels, probabilities, label=f"m{index}")
        )
        output = tmp_path / f"dist{index}.png"
        plot_prediction_distribution([data], output)
        digests.append(np.frombuffer(output.read_bytes(), dtype=np.uint8).sum())
    assert digests[0] != digests[1]


def test_time_response_plot_uses_bucket_rows(tmp_path):
    rows = [
        {"bucket": "[0s, 1m)", "mean_y_pred": 0.6, "std_y_pred": 0.05, "sample_count": 12,
         "is_mock": True},
        {"bucket": "[1m, 1h)", "mean_y_pred": 0.55, "std_y_pred": 0.04, "sample_count": 30,
         "is_mock": True},
    ]
    output = tmp_path / "time.png"
    result = plot_time_response(rows, output)
    assert result.status == "created"
    assert png_size(output)[1] > 300


def test_time_response_plot_skipped_without_rows(tmp_path):
    assert plot_time_response([], tmp_path / "t.png").status == "skipped"


def test_model_time_response_requires_two_models(tmp_path):
    single = [{"label": "only", "available": True, "model": {"is_mock": True},
               "per_interval": [{"interval_days": 0.0, "mean_probability": 0.5,
                                 "std_probability": 0.0}]}]
    result = plot_model_time_response(single, tmp_path / "c.png")
    assert result.status == "skipped"
    assert "至少两个模型" in result.reason

    pair = single + [{"label": "second", "available": True, "model": {"is_mock": True},
                      "per_interval": [{"interval_days": 0.0, "mean_probability": 0.4,
                                        "std_probability": 0.01}]}]
    output = tmp_path / "c.png"
    assert plot_model_time_response(pair, output).status == "created"
    assert png_size(output)[0] > 400


def test_throughput_plot_is_created(tmp_path):
    rows = [{"batch_size": size, "throughput_samples_per_second": 1000 + size,
             "is_mock": True} for size in (1, 8, 32)]
    output = tmp_path / "tp.png"
    assert plot_throughput(rows, output).status == "created"
    assert png_size(output)[0] > 400


def test_throughput_plot_skipped_without_rows(tmp_path):
    assert plot_throughput([], tmp_path / "tp.png").status == "skipped"


def test_generate_plots_creates_all_nine_and_records_manifest(tmp_path):
    good = load_predictions(prediction_file(
        tmp_path / "good.csv", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], label="alpha"
    ))
    other = load_predictions(prediction_file(
        tmp_path / "other.csv", [0, 0, 1, 1], [0.4, 0.3, 0.6, 0.7], label="beta"
    ))
    inputs = PlotInputs(
        metrics=metrics_payload(["alpha", "beta"], [0.9, 0.8]),
        forgetting={
            "descriptive": {"available": True, "bucket_rows": [
                {"bucket": "[0s, 1m)", "mean_y_pred": 0.5, "std_y_pred": 0.1,
                 "sample_count": 4, "is_mock": True},
            ]},
            "controlled_models": [
                {"label": "mock", "available": True, "model": {"is_mock": True},
                 "per_interval": [{"interval_days": 0.0, "mean_probability": 0.5,
                                   "std_probability": 0.0}]},
                {"label": "fixture", "available": True, "model": {"is_mock": True},
                 "per_interval": [{"interval_days": 0.0, "mean_probability": 0.6,
                                   "std_probability": 0.0}]},
            ],
        },
        performance={"rows": [
            {"batch_size": size, "mean_batch_latency_ms": 1.0 * size, "p50_batch_latency_ms": 1.0,
             "p95_batch_latency_ms": 2.0, "mean_per_sample_latency_ms": 0.5,
             "throughput_samples_per_second": 500.0, "is_mock": True}
            for size in (1, 8)
        ]},
        prediction_sources=(good, other),
    )
    payload = generate_plots(inputs, tmp_path / "out", overwrite=True)
    assert len(payload["created"]) == 9
    assert payload["skipped"] == []
    manifest = json.loads(
        (tmp_path / "out" / "plots" / "plots_manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["created"]) == set(payload["created"])
    for item in manifest["plots"]:
        assert (tmp_path / "out" / "plots" / item["file"]).is_file()


def test_generate_plots_skips_everything_without_data(tmp_path):
    payload = generate_plots(PlotInputs(), tmp_path / "out", overwrite=True)
    assert payload["created"] == []
    assert len(payload["skipped"]) == 9
    assert all(item["reason"] for item in payload["skipped"])


def test_generate_plots_refuses_overwrite_without_flag(tmp_path):
    generate_plots(PlotInputs(), tmp_path / "out", overwrite=True)
    with pytest.raises(EvaluationError, match="--overwrite"):
        generate_plots(PlotInputs(), tmp_path / "out", overwrite=False)


def test_require_plot_inputs_rejects_empty_inputs():
    with pytest.raises(EvaluationError, match="没有可供绘图的数据"):
        require_plot_inputs(PlotInputs())
    require_plot_inputs(PlotInputs(metrics=metrics_payload(["a"], [0.5])))


def test_demo_watermark_flag_follows_is_mock(tmp_path):
    """模拟数据必须打水印、真实数据不打；同一份数据仅 is_mock 不同即产生不同图片。"""
    labels, probabilities = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    mock = load_predictions(prediction_file(tmp_path / "mock.csv", labels, probabilities))
    real = load_predictions(prediction_file(
        tmp_path / "real.csv", labels, probabilities, is_mock="False"
    ))
    mock_output, real_output = tmp_path / "mock.png", tmp_path / "real.png"
    mock_result = plot_roc_curves([mock], mock_output)
    real_result = plot_roc_curves([real], real_output)
    assert mock_result.demo is True and real_result.demo is False
    assert mock_output.read_bytes() != real_output.read_bytes()
    assert DEMO_TEXT  # 水印文案与图片一同由 visualizer 提供


def test_load_artifacts_reports_missing_files(tmp_path):
    loaded = load_artifacts(tmp_path / "nonexistent")
    assert loaded == {"metrics": None, "forgetting": None, "comparison": None, "performance": None}


def test_load_artifacts_reads_written_json(tmp_path):
    target = tmp_path / "metrics"
    target.mkdir()
    (target / "metrics.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    loaded = load_artifacts(tmp_path)
    assert loaded["metrics"] == {"rows": []}
    assert loaded["performance"] is None


def test_plots_do_not_hardcode_values(tmp_path):
    """同一函数在不同输入下必须产出不同像素，证明图来自数据而非写死的图形。"""
    values = [(0.1, 0.2, 0.8, 0.9), (0.9, 0.8, 0.2, 0.1)]
    digests = []
    for index, probabilities in enumerate(values):
        source = load_predictions(prediction_file(
            tmp_path / f"p{index}.csv", [0, 0, 1, 1], list(probabilities), label=f"m{index}"
        ))
        output = tmp_path / f"roc{index}.png"
        plot_roc_curves([source], output)
        digests.append(np.frombuffer(output.read_bytes(), dtype=np.uint8).sum())
    assert digests[0] != digests[1]
