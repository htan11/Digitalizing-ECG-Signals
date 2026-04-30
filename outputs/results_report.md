# PhysioNet ECG Image Digitization: Results and Analysis

## Implementation Details

- Problem: reconstruct ECG time series from ECG paper/image inputs.
- Data used: `58` record folders and `696` lead-level samples from `dl-lms-data/sample_data`.
- Digitization model: compact CNN/U-Net-style image encoder trained to regress a normalized `256` point waveform from an ECG-paper strip image.
- Baselines: mean-signal baseline and a classical darkest-pixel trace extractor.
- Downstream classifier: amplitude-bin classification using an LSTM over the digitized signal and a multimodal ResNet-style image branch fused with the LSTM branch.
- Important note: the ResNet+LSTM fusion is evaluated as downstream classification. It is not used as the primary digitizer, because image-to-signal regression is the direct task for ECG digitization.

## Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Seed | 42 |
| Signal length | 256 |
| Image size | 128 x 512 |
| Digitizer epochs | 3 |
| Classifier epochs | 3 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Validation size | 0.2 |
| Test size | 0.2 |
| CNN filters | (16, 32, 64) |
| Dropout | 0.25 |
| LSTM units | 32 |

## Regression Results

| model                        | MSE    | RMSE   | MAE    | MAPE_%   | R2      |
| ---------------------------- | ------ | ------ | ------ | -------- | ------- |
| Mean signal baseline         | 0.0359 | 0.1895 | 0.1004 | 158.2895 | -0.0027 |
| Darkest-pixel trace baseline | 0.0046 | 0.0682 | 0.0393 | 225.3831 | 0.8703  |
| CNN/U-Net-style digitizer    | 0.0085 | 0.0925 | 0.0595 | 186.6246 | 0.7613  |

Best digitization result by RMSE: **Darkest-pixel trace baseline** with RMSE `0.0682`, MAE `0.0393`, MAPE `225.38%`, and R2 `0.8703`.

The mean-signal baseline is intentionally weak because it ignores the ECG image. The darkest-pixel baseline uses image content but is sensitive to grid lines, calibration pulses, and text. The CNN digitizer learns a smoother mapping from paper-like image evidence to waveform samples, so it should be judged against both baselines using RMSE/MAE and by visual overlay.

## Classification Results

| model                      | Accuracy | Precision_macro | Recall_macro | F1_macro |
| -------------------------- | -------- | --------------- | ------------ | -------- |
| Majority class baseline    | 0.3286   | 0.1095          | 0.3333       | 0.1649   |
| LSTM signal classifier     | 0.4000   | 0.4112          | 0.4002       | 0.3455   |
| ResNet-style + LSTM fusion | 0.3357   | 0.3583          | 0.3341       | 0.3127   |

Best downstream classifier by macro F1: **LSTM signal classifier** with accuracy `0.4000` and macro F1 `0.3455`.

The multimodal fusion model combines ResNet-style image features with LSTM temporal features. This is useful for downstream tasks because morphology and amplitude patterns can complement the signal stream, but digitization itself remains a regression problem and is better evaluated with MSE, RMSE, MAE, MAPE, R2, physical-vs-digital plots, residual plots, and waveform overlays.

## Generated Figures

- `plots/loss_curve.png`: digitizer training and validation loss.
- `plots/model_comparison.png`: regression comparison against base models.
- `plots/physical_vs_digital_ecg.png`: scanned ECG page, generated paper strip, and digital CSV waveform.
- `plots/waveform_overlay.png`: digital vs CNN reconstructed waveforms.
- `plots/scatter_residuals.png`: predicted-vs-actual and residual histogram.
- `plots/amplitude_bin_confusion_matrix.png`: downstream classifier confusion matrix.
- `plots/per_lead_rmse.png`: lead-wise digitization error.
- `plots/error_heatmap.png`: record/lead error heatmap.
- `plots/regression_metrics_table.png`, `plots/classification_metrics_table.png`, `plots/hyperparameters_table.png`: presentation tables.
