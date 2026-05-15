# CDC Priority RL

面向 CDC/行业事件流的优先级分类与调度优化毕业设计项目。

项目当前包含两条完整实验线：

- 公开数据集实验线
- 达梦行业数据集实验线

整体采用两阶段流程：

1. 优先级分类：为事件生成 `low / medium / high` 三档优先级
2. 调度优化：比较规则调度与强化学习调度在高优保障、整体延迟与公平性上的表现

## 运行环境

当前默认解释器：

- `D:\anaconda\envs\dl\python.exe`

建议后续所有复现命令都统一用这个解释器执行。

## 项目结构

- `configs/`: YAML 配置文件
- `cdc_priority/data/`: 数据集加载、预处理、打标、切分
- `cdc_priority/classifier/`: 分类模型、baseline、评估
- `cdc_priority/scheduler/`: 调度环境、规则策略、RL agent、评估
- `cdc_priority/pipeline/`: 端到端流水线编排
- `data/`: 原始数据、中间数据、行业数据
- `outputs/`: 训练结果、报告、图表、汇总文件

## 当前默认配置

### 公开数据集

- 分类配置：[classifier.yaml](/D:/Code/Python/configs/classifier.yaml:1)
- 调度配置：[scheduler.yaml](/D:/Code/Python/configs/scheduler.yaml:1)
- 数据集配置：[dataset.yaml](/D:/Code/Python/configs/dataset.yaml:1)

### 达梦行业数据集

- 分类配置：[classifier_industry_dm_alarm.yaml](/D:/Code/Python/configs/classifier_industry_dm_alarm.yaml:1)
- 调度配置：[scheduler_industry_dm_alarm.yaml](/D:/Code/Python/configs/scheduler_industry_dm_alarm.yaml:1)
- 数据集配置：[dataset_industry_dm_alarm.yaml](/D:/Code/Python/configs/dataset_industry_dm_alarm.yaml:1)
- 行业数据导出脚本：[export_dm_industry_dataset.py](/D:/Code/Python/export_dm_industry_dataset.py:1)

行业数据当前默认设定：

- `burst_window = 60s`
- `arrival_step_time_unit_seconds = 30`

## 快速开始

### 1. 公开数据集单模块实验

构建分类数据集：

```powershell
D:\anaconda\envs\dl\python.exe build_dataset.py
```

构建调度数据集：

```powershell
D:\anaconda\envs\dl\python.exe build_scheduler_dataset.py
```

训练分类器：

```powershell
D:\anaconda\envs\dl\python.exe train_classifier.py --config D:\Code\Python\configs\classifier.yaml --run-name public-manual
```

训练调度器：

```powershell
D:\anaconda\envs\dl\python.exe train_scheduler.py --config D:\Code\Python\configs\scheduler.yaml --run-name public-manual
```

### 2. 达梦行业数据集单模块实验

先导出行业事件数据（默认 `60s` burst 聚合）：

```powershell
D:\anaconda\envs\dl\python.exe export_dm_industry_dataset.py --password Hzy12345678 --burst-window-seconds 60
```

再构建行业调度中间数据：

```powershell
@'
from pathlib import Path
from cdc_priority.data.dataset_builder import build_and_export_scheduler_dataset_from_config
build_and_export_scheduler_dataset_from_config(
    Path(r'D:\Code\Python\configs\dataset_industry_dm_alarm.yaml'),
    Path(r'D:\Code\Python\data\scheduler_processed_industry_dm_alarm'),
    timestamp_column='timestamp',
)
print('done')
'@ | D:\anaconda\envs\dl\python.exe -
```

训练行业分类器：

```powershell
D:\anaconda\envs\dl\python.exe train_classifier.py --config D:\Code\Python\configs\classifier_industry_dm_alarm.yaml --run-name dm-manual
```

训练行业调度器：

```powershell
D:\anaconda\envs\dl\python.exe train_scheduler.py --config D:\Code\Python\configs\scheduler_industry_dm_alarm.yaml --run-name dm-manual
```

## 端到端复现命令

### 公开数据集端到端主实验

```powershell
D:\anaconda\envs\dl\python.exe run_pipeline.py --classifier-config D:\Code\Python\configs\classifier.yaml --scheduler-config D:\Code\Python\configs\scheduler.yaml --run-name final-public-pipeline
```

主产物：

- [pipeline_report.json](/D:/Code/Python/outputs/pipeline/final-public-pipeline/pipeline_report.json:1)
- [classifier_report.json](/D:/Code/Python/outputs/classifier/final-public-pipeline/classifier_report.json:1)
- [scheduler_report.json](/D:/Code/Python/outputs/scheduler/final-public-pipeline/scheduler_report.json:1)

### 达梦行业数据集端到端主实验

```powershell
D:\anaconda\envs\dl\python.exe run_pipeline.py --classifier-config D:\Code\Python\configs\classifier_industry_dm_alarm.yaml --scheduler-config D:\Code\Python\configs\scheduler_industry_dm_alarm.yaml --run-name final-dm-pipeline
```

主产物：

- [pipeline_report.json](/D:/Code/Python/outputs/pipeline/final-dm-pipeline/pipeline_report.json:1)
- [classifier_report.json](/D:/Code/Python/outputs/classifier/final-dm-pipeline/classifier_report.json:1)
- [scheduler_report.json](/D:/Code/Python/outputs/scheduler/final-dm-pipeline/scheduler_report.json:1)

## 最终主结果

### 分类结果

公开数据集：

- `embedding_mlp`: `accuracy = 0.9947`
- `lightgbm`: `accuracy = 0.9960`

行业数据集：

- `embedding_mlp`: `accuracy = 0.9884`
- `lightgbm`: `accuracy = 0.9900`

结论：

- 当前分类任务更像表格学习问题
- `LightGBM` 是当前最优分类 baseline / 推荐方案

### 调度结果

公开数据集：

- `strict_priority` 高优保障最强
- `dqn` 在高优保障与整体延迟之间更折中

行业数据集：

- `strict_priority` 高优匹配最强
- `dqn` 在公平性和整体队列压力控制上更平衡

结论：

- 规则策略在高优绝对保障上仍然非常强
- `DQN` 更适合作为综合平衡策略

## 重复实验与敏感性分析

重复实验汇总文件：

- [day3_classifier_repeats.csv](/D:/Code/Python/outputs/final_summaries/day3_classifier_repeats.csv:1)
- [day3_scheduler_repeats.csv](/D:/Code/Python/outputs/final_summaries/day3_scheduler_repeats.csv:1)
- [day3_repeat_summary.json](/D:/Code/Python/outputs/final_summaries/day3_repeat_summary.json:1)

baseline 快照：

- [day3_baseline_snapshot.csv](/D:/Code/Python/outputs/final_summaries/day3_baseline_snapshot.csv:1)

行业 burst 聚合敏感性：

- [day4_burst_sensitivity.csv](/D:/Code/Python/outputs/final_summaries/day4_burst_sensitivity.csv:1)

## 论文图表与产物路径对照

建议论文/汇报优先引用以下文件：

### 主结果

- 公开线 pipeline：[pipeline_report.json](/D:/Code/Python/outputs/pipeline/final-public-pipeline/pipeline_report.json:1)
- 行业线 pipeline：[pipeline_report.json](/D:/Code/Python/outputs/pipeline/final-dm-pipeline/pipeline_report.json:1)

### 分类图表

- 公开线混淆矩阵：[confusion_matrix.png](/D:/Code/Python/outputs/classifier/final-public-pipeline/confusion_matrix.png:1)
- 行业线混淆矩阵：[confusion_matrix.png](/D:/Code/Python/outputs/classifier/final-dm-pipeline/confusion_matrix.png:1)

### 调度图表

- 公开线策略对比图：[policy_comparison.png](/D:/Code/Python/outputs/scheduler/final-public-pipeline/policy_comparison.png:1)
- 行业线策略对比图：[policy_comparison.png](/D:/Code/Python/outputs/scheduler/final-dm-pipeline/policy_comparison.png:1)

### 稳定性与敏感性

- 重复实验摘要：[day3_repeat_summary.json](/D:/Code/Python/outputs/final_summaries/day3_repeat_summary.json:1)
- baseline 快照：[day3_baseline_snapshot.csv](/D:/Code/Python/outputs/final_summaries/day3_baseline_snapshot.csv:1)
- burst 敏感性：[day4_burst_sensitivity.csv](/D:/Code/Python/outputs/final_summaries/day4_burst_sensitivity.csv:1)

## 测试命令

```powershell
D:\anaconda\envs\dl\python.exe -m pytest tests/test_data.py -q -p no:cacheprovider
D:\anaconda\envs\dl\python.exe -m pytest tests/test_classifier.py -q -p no:cacheprovider
D:\anaconda\envs\dl\python.exe -m pytest tests/test_scheduler.py -q -p no:cacheprovider
D:\anaconda\envs\dl\python.exe -m pytest tests/test_pipeline.py -q -p no:cacheprovider
D:\anaconda\envs\dl\python.exe -m pytest tests/test_dm_industry_export.py -q -p no:cacheprovider
```

最终验收建议执行：

```powershell
D:\anaconda\envs\dl\python.exe -m pytest tests/test_pipeline.py tests/test_dm_industry_export.py tests/test_scheduler.py tests/test_classifier.py tests/test_data.py -q -p no:cacheprovider
```

## 相关材料

- 详细报告：[PROJECT_PROGRESS_REPORT_20260506.md](/D:/Code/Python/PROJECT_PROGRESS_REPORT_20260506.md:1)
- 一页摘要：[PROJECT_PROGRESS_SUMMARY_20260510.md](/D:/Code/Python/PROJECT_PROGRESS_SUMMARY_20260510.md:1)
- 交付清单：[FINAL_DELIVERY_CHECKLIST.md](/D:/Code/Python/FINAL_DELIVERY_CHECKLIST.md:1)
- 答辩提纲：[DEFENSE_OUTLINE.md](/D:/Code/Python/DEFENSE_OUTLINE.md:1)
- 答辩问答：[DEFENSE_QA.md](/D:/Code/Python/DEFENSE_QA.md:1)
