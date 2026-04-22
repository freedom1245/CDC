# CDC Priority RL

Project skeleton for a graduation design focused on CDC event priority
classification and reinforcement-learning-based synchronization scheduling.

## Entry Points

- `python train_classifier.py`
- `python train_scheduler.py`
- `python run_pipeline.py`

## Structure

- `configs/`: YAML configuration files
- `cdc_priority/data/`: dataset loading, preprocessing, labeling, splitting
- `cdc_priority/classifier/`: tabular classification models and evaluation
- `cdc_priority/scheduler/`: scheduling policies, RL environment, fairness logic
- `cdc_priority/pipeline/`: end-to-end orchestration
- `outputs/`: generated models, metrics, and figures

## Notes

The existing `thesios_classifier` package is kept intact so the current training
workflow can continue while this new architecture is filled in incrementally.
