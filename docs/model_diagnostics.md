# Model-performance diagnostics and the class-imbalance root cause

## Summary

The low macro-F1 values are not caused by a broken evaluator or a dropped metric. The actual root cause is the label distribution and the small number of classes with support in the held-out split.

The frozen MIT-BIH split keeps all 15 ECG beat labels in the training set, but the validation and held-out test sets contain only a subset of those labels. The majority class (`N`) dominates, while rare classes such as `f`, `J`, `Q`, `S`, and `e` have very few examples or no support in evaluation.

This means:

- Accuracy remains a poor summary statistic because the classifier is dominated by the normal-beat majority class.
- Macro-F1 is low because the model must score rare classes equally, even when they are scarcely represented.
- A perfectly valid centralized CNN can still show modest macro-F1 under extreme imbalance, especially when training is not explicitly class-weighted.

## Evidence from the repository

The saved final comparison table in [`../results/final/final_utility.csv`](../results/final/final_utility.csv) shows the actual behavior:

- centralized CNN accuracy: 0.7688
- centralized CNN macro-F1: 0.1399
- centralized CNN weighted-F1: 0.7208
- FedAvg accuracy: 0.7069
- FedAvg macro-F1: 0.0848

The framework also records validation curves and class distributions on the frozen MIT-BIH split. The per-class support is highly skewed. This is consistent with the historical finding that only 8 of the 15 classes are present in the held-out test set and the rarest classes have almost no support.

## Why the target is not a valid code requirement

The original supervisor note mentions a target of 90% accuracy, but the project data itself does not support a guaranteed 90% classification result under the current 15-class, highly imbalanced MIT-BIH protocol. The repo should therefore treat 90% as an aspirational target, not as a reproducible requirement unless the model, split, label taxonomy, and evaluation protocol are changed and re-validated without test-set tuning.

## Controlled improvement implemented

A separate versioned training configuration is added at [`../configs/training/centralized_cnn_weighted_v1.json`](../configs/training/centralized_cnn_weighted_v1.json). It uses inverse-frequency class weights during training to reduce majority-class domination without changing the frozen historical split or test data.

This is a controlled, validation-only improvement path. It does not overwrite the historical centralized prototype or the frozen results; it is a new training variant with a distinct output directory.

## Remaining scientific interpretation

The model may improve modestly under class weighting or alternative class-balanced training, but the dataset itself remains a severe multiclass imbalance problem. The research conclusion should therefore be framed as:

- strong normal-class accuracy is attainable,
- macro-F1 remains weak because rare classes are scarce and the task is intrinsically imbalanced,
- the reported utility should be described as a realistic prototype result, not as clinical-grade performance.
