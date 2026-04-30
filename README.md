# ECG Image Digitization Project

This workspace contains a complete runnable experiment for the PhysioNet ECG image digitization theme.

## Run

```powershell
python .\ecg_digitization_project.py --quick
```

For a longer run:

```powershell
python .\ecg_digitization_project.py --digitizer-epochs 12 --classifier-epochs 10
```

## What It Produces

The script writes all deliverables to `outputs/`:

- `outputs/results_report.md`: implementation details, hyperparameters, results, and analysis.
- `outputs/metrics/regression_metrics.csv`: MSE, RMSE, MAE, MAPE, R2 for digitization.
- `outputs/metrics/classification_metrics.csv`: accuracy, precision, recall, F1 for downstream classification.
- `outputs/plots/loss_curve.png`: digitizer loss curve.
- `outputs/plots/classification_loss_curve.png`: classifier loss curve.
- `outputs/plots/model_comparison.png`: CNN digitizer compared with base models.
- `outputs/plots/physical_vs_digital_ecg.png`: scanned physical ECG image vs generated strip vs digital signal.
- `outputs/plots/waveform_overlay.png`: predicted waveform overlaid on digital CSV waveform.
- `outputs/plots/scatter_residuals.png`: scatter and residual analysis.
- `outputs/plots/amplitude_bin_confusion_matrix.png`: confusion matrix.
- `outputs/plots/per_lead_rmse.png`: lead-wise RMSE.
- `outputs/plots/error_heatmap.png`: record/lead error heatmap.
- table plots for hyperparameters and metrics.

## Model Summary

The primary digitization task is treated as regression from an ECG-paper image strip to a numeric waveform. A compact CNN/U-Net-style digitizer is compared with:

- a mean-signal baseline
- a darkest-pixel trace extraction baseline

A ResNet-style image branch plus LSTM signal branch is included for downstream amplitude-bin classification, matching the multimodal fusion requirement. It is intentionally reported as downstream classification, not as the main digitization model.
