# 鲁凯宁任务交付核对

## 七项任务对应成果

| 图片任务 | 工程成果 | 验收依据 |
| --- | --- | --- |
| 环境搭建 | requirements.txt、requirements-lock.txt、pyproject.toml、macOS/Linux/Windows 安装脚本 | Python 3.12.14 环境实际运行；锁定依赖安装成功 |
| 测试数据准备 | sample_data.csv、generate_sample_data.py、dataset_converters.py、examples/ | 288 条可复现模拟数据；ASSISTments/EdNet 人工夹具转换与端到端推理 |
| 数据预处理 | data_loader.py、docs/DATA_FORMAT.md | 缺失值、格式、时区、排序、多知识点、固定词表、拒收记录测试 |
| 模拟模型 | mock_model.py | 仅使用可见历史的确定性概率；冷启动与批量一致性测试 |
| 模型对接接口 | contracts.py、model_adapter.py、examples/、docs/MODEL_INTERFACE.md | 插件加载、乱序 ID 对齐、数量/ID/概率错误测试，未接真实模型时明确报错 |
| 主测试程序 | main.py、pipeline.py、io_utils.py | python main.py 读取默认数据并生成统一 CSV 与运行摘要 |
| 基础功能测试 | tests/test_pipeline.py、tests/test_datasets.py、tests/test_cli.py、scripts/verify.sh | 自动测试、覆盖率与集成验收，具体数值见 TEST_REPORT.md |

## 最终成果对应

- 可直接运行的 Python 主程序：`main.py`。
- 规范的测试数据与读取程序：`sample_data.csv`、`data_loader.py`、`docs/DATA_FORMAT.md`。
- 临时模拟模型：`mock_model.py`，明确声明 is_mock=True。
- 可替换真实模型的统一接口：`ModelAdapter.predict()`、`PredictModel` 协议、外部模型工厂入口。
- 基础功能测试与运行说明：`tests/`、`README.md`、`docs/TEST_REPORT.md`。
- 真实数据处理接口：ASSISTments 和 EdNet KT1 转换器及人工夹具，不包含未提供的真实数据。
- 刘静涵的评估交接格式：含 y_true/y_pred、稳定 record_id、时间、模型类型的 predictions.csv。

## 交付边界

负责人 A 的测试框架范围已实现并完成当前环境验收。真实 DKT/AKT 权重、遗忘模型验证、负责人 B 的评估模块、LLM 路径生成、知识图谱数据库和鸿蒙端部署不在本次任务图片为负责人 A 指定的成果中，也没有相应输入资产。接入真实模型时按 MODEL_INTERFACE.md 提供训练产物即可继续使用本框架。
