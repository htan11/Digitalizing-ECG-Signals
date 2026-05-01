 """
PhysioNet ECG Image Digitization project runner.
...
"""

from __future__ import annotations

# Imports
import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# TF Imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as exc:  # pragma: no cover - handled at runtime.
    tf = None
    keras = None
    layers = None
    TF_IMPORT_ERROR = exc
else:
    TF_IMPORT_ERROR = None

# Constants
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Config Data
@dataclass
class HyperParameters:
    seed: int = 42
    signal_length: int = 256
    image_height: int = 128
    image_width: int = 512
    digitizer_epochs: int = 8
    classifier_epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 0.001
    validation_size: float = 0.2
    test_size: float = 0.2
    conv_filters: Tuple[int, int, int] = (16, 32, 64)
    dropout: float = 0.25
    lstm_units: int = 32
    classes: Tuple[str, str, str] = ("low_amplitude", "normal_amplitude", "high_amplitude")

# Reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if tf is not None:
        tf.random.set_seed(seed)

# Folder Setup
def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "plots": output_dir / "plots",
        "metrics": output_dir / "metrics",
        "predictions": output_dir / "predictions",
        "cache": output_dir / "cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

# Find Data
def find_records(data_dir: Path) -> List[Path]:
    records = []
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and (child / f"{child.name}.csv").exists():
            records.append(child)
    if not records:
        raise FileNotFoundError(f"No record folders with matching CSV found in {data_dir}")
    return records

# Resize Signal
def interpolate_signal(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return np.zeros(length, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, length)
    out = np.interp(x_new, x_old, values).astype(np.float32)
    out = out - np.nanmedian(out)
    max_abs = float(np.nanmax(np.abs(out)))
    if max_abs > 1e-6:
        out = out / max_abs
    return out.astype(np.float32)

# Load CSVs
def load_signals(records: Iterable[Path], length: int) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    signals: List[np.ndarray] = []
    meta: List[Dict[str, str]] = []
    for record_dir in records:
        csv_path = record_dir / f"{record_dir.name}.csv"
        df = pd.read_csv(csv_path)
        for lead in LEADS:
            if lead not in df.columns:
                continue
            raw = pd.to_numeric(df[lead], errors="coerce").dropna().to_numpy(dtype=np.float32)
            if raw.size < 32:
                continue
            signals.append(interpolate_signal(raw, length))
            meta.append({"record": record_dir.name, "lead": lead, "csv": str(csv_path)})
    if not signals:
        raise ValueError("No usable lead signals were found.")
    return np.stack(signals).astype(np.float32), meta

# Generate Image
def draw_ecg_strip(signal: np.ndarray, hp: HyperParameters, rng: np.random.Generator) -> np.ndarray:
    """Render one normalized signal as an ECG-paper-like grayscale image."""
    h, w = hp.image_height, hp.image_width
    img = np.ones((h, w), dtype=np.float32)

    # Light ECG grid: minor and major boxes.
    minor = max(4, w // 64)
    major = minor * 5
    for x in range(0, w, minor):
        img[:, x : x + 1] = np.minimum(img[:, x : x + 1], 0.92)
    for y in range(0, h, minor):
        img[y : y + 1, :] = np.minimum(img[y : y + 1, :], 0.92)
    for x in range(0, w, major):
        img[:, x : x + 1] = np.minimum(img[:, x : x + 1], 0.78)
    for y in range(0, h, major):
        img[y : y + 1, :] = np.minimum(img[y : y + 1, :], 0.78)

    amp = rng.uniform(0.32, 0.42) * h
    baseline = rng.uniform(0.46, 0.56) * h
    x_signal = np.linspace(0, signal.size - 1, w)
    y = baseline - np.interp(x_signal, np.arange(signal.size), signal) * amp
    y += rng.normal(0.0, 0.8, size=w)
    y = np.clip(y, 3, h - 4)

    # Draw a 2-3 px trace.
    for x in range(w - 1):
        y0, y1 = int(round(y[x])), int(round(y[x + 1]))
        lo, hi = sorted((y0, y1))
        img[max(0, lo - 1) : min(h, hi + 2), x : x + 2] = 0.05

    # Calibration pulse.
    pulse_x = max(4, w // 50)
    pulse_w = max(8, w // 35)
    pulse_h = max(12, h // 4)
    base_y = int(baseline)
    img[base_y - pulse_h : base_y, pulse_x : pulse_x + 2] = 0.08
    img[base_y - pulse_h : base_y - pulse_h + 2, pulse_x : pulse_x + pulse_w] = 0.08
    img[base_y - pulse_h : base_y, pulse_x + pulse_w : pulse_x + pulse_w + 2] = 0.08

    noise = rng.normal(0.0, 0.015, size=img.shape)
    img = np.clip(img + noise, 0.0, 1.0)
    return img[..., None].astype(np.float32)

# Build Dataset
def make_dataset(signals: np.ndarray, hp: HyperParameters) -> np.ndarray:
    rng = np.random.default_rng(hp.seed)
    return np.stack([draw_ecg_strip(sig, hp, rng) for sig in signals]).astype(np.float32)

# Baseline Image
def trace_baseline_from_image(images: np.ndarray, length: int) -> np.ndarray:
    """Classical baseline: estimate the darkest waveform row per column."""
    preds = []
    for img in images[..., 0]:
        darkness = 1.0 - img
        y_idx = np.argmax(darkness, axis=0).astype(np.float32)
        # Remove obvious grid/pulse jumps with median-centering and normalization.
        y = -(y_idx - np.median(y_idx))
        x_old = np.linspace(0.0, 1.0, y.size)
        x_new = np.linspace(0.0, 1.0, length)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        y = y - np.median(y)
        scale = np.max(np.abs(y))
        preds.append(y / scale if scale > 1e-6 else y)
    return np.stack(preds).astype(np.float32)

# Baseline Signal
def mean_signal_baseline(y_train: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(np.mean(y_train, axis=0, keepdims=True), n, axis=0).astype(np.float32)

# Build CNN
def build_digitizer(hp: HyperParameters):
    if keras is None:
        raise RuntimeError(f"TensorFlow/Keras import failed: {TF_IMPORT_ERROR}")
    inp = keras.Input(shape=(hp.image_height, hp.image_width, 1), name="ecg_paper_strip")
    x = layers.Lambda(lambda t: 1.0 - t, name="darkness_transform")(inp)
    x = layers.Conv2D(hp.conv_filters[0], 5, strides=(2, 1), padding="same", activation="relu")(x)
    x = layers.Conv2D(hp.conv_filters[1], 5, strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.Conv2D(hp.conv_filters[2], 3, strides=(2, 1), padding="same", activation="relu")(x)
    # Collapse vertical paper evidence into a left-to-right sequence. With the
    # default 512 px strip width and one horizontal stride of 2, this produces
    # exactly 256 time positions for the regression target.
    x = layers.Lambda(lambda t: tf.reduce_max(t, axis=1), name="vertical_trace_pool")(x)
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(x)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.Dropout(hp.dropout)(x)
    x = layers.Dense(1, activation="linear")(x)
    out = layers.Reshape((hp.signal_length,), name="digitized_signal")(x)
    model = keras.Model(inp, out, name="compact_unet_style_digitizer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model

# Build LSTM
def build_signal_classifier(hp: HyperParameters):
    if keras is None:
        raise RuntimeError(f"TensorFlow/Keras import failed: {TF_IMPORT_ERROR}")
    seq_in = keras.Input(shape=(hp.signal_length, 1), name="digitized_signal")
    x = layers.Bidirectional(layers.LSTM(hp.lstm_units, return_sequences=False))(seq_in)
    x = layers.Dropout(hp.dropout)(x)
    x = layers.Dense(48, activation="relu")(x)
    out = layers.Dense(len(hp.classes), activation="softmax")(x)
    model = keras.Model(seq_in, out, name="lstm_signal_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(hp.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Build ResNet
def build_multimodal_classifier(hp: HyperParameters):
    if keras is None:
        raise RuntimeError(f"TensorFlow/Keras import failed: {TF_IMPORT_ERROR}")
    img_in = keras.Input(shape=(hp.image_height, hp.image_width, 1), name="resnet_image_input")
    x = layers.AveragePooling2D(pool_size=(4, 4), name="image_downsample")(img_in)
    x = layers.Conv2D(12, 3, padding="same", activation="relu")(x)
    shortcut = layers.Conv2D(16, 1, padding="same")(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(16, 3, padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    image_features = layers.Dense(48, activation="relu", name="resnet_style_features")(x)

    seq_in = keras.Input(shape=(hp.signal_length, 1), name="lstm_signal_input")
    y = layers.Bidirectional(layers.LSTM(hp.lstm_units))(seq_in)
    y = layers.Dense(48, activation="relu")(y)

    fused = layers.Concatenate(name="multimodal_fusion")([image_features, y])
    fused = layers.Dropout(hp.dropout)(fused)
    fused = layers.Dense(48, activation="relu")(fused)
    out = layers.Dense(len(hp.classes), activation="softmax")(fused)
    model = keras.Model([img_in, seq_in], out, name="resnet_lstm_multimodal_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(hp.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Calc Errors
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mse = mean_squared_error(yt, yp)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(yt, yp)
    denom = np.maximum(np.abs(yt), 1e-3)
    mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE_%": mape,
        "R2": float(r2_score(yt, yp)),
    }

# Create Labels
def make_labels(signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    peak_to_peak = np.ptp(signals, axis=1)
    q1, q2 = np.quantile(peak_to_peak, [1 / 3, 2 / 3])
    labels = np.digitize(peak_to_peak, bins=[q1, q2], right=False).astype(np.int64)
    return labels, peak_to_peak

# Plot Loss
def plot_loss(history, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history.get("loss", []), label="train loss", linewidth=2)
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="validation loss", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Plot Comparison
def plot_model_comparison(metrics_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metrics_df.plot.bar(x="model", y=["RMSE", "MAE"], ax=axes[0], rot=20)
    axes[0].set_title("Error comparison")
    axes[0].set_ylabel("Lower is better")
    axes[0].grid(axis="y", alpha=0.25)
    metrics_df.plot.bar(x="model", y=["R2"], ax=axes[1], color=["#4C78A8"], rot=20)
    axes[1].set_title("Explained variance")
    axes[1].set_ylabel("Higher is better")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Plot Overlay
def plot_waveform_overlay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: List[Dict[str, str]],
    out: Path,
    n: int = 12,
) -> None:
    n = min(n, len(y_true))
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(13, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    x = np.arange(y_true.shape[1])
    for i in range(rows * cols):
        ax = axes[i]
        if i < n:
            ax.plot(x, y_true[i], label="digital CSV", linewidth=1.5)
            ax.plot(x, y_pred[i], label="CNN digitized", linewidth=1.1, alpha=0.85)
            ax.set_title(f"{meta[i]['record']} lead {meta[i]['lead']}", fontsize=9)
            ax.grid(alpha=0.2)
        else:
            ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Digitized waveform overlay", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Plot Residuals
def plot_scatter_residuals(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    residuals = yt - yp
    if yt.size > 6000:
        rng = np.random.default_rng(7)
        idx = rng.choice(yt.size, size=6000, replace=False)
        yt, yp, residuals = yt[idx], yp[idx], residuals[idx]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(yt, yp, s=5, alpha=0.3)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set_title("Predicted vs digital amplitude")
    axes[0].set_xlabel("Digital CSV amplitude")
    axes[0].set_ylabel("Digitized amplitude")
    axes[0].grid(alpha=0.25)
    axes[1].hist(residuals, bins=60, color="#F58518", alpha=0.85)
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Digital - predicted")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Plot Heatmap
def plot_error_heatmap(y_true: np.ndarray, y_pred: np.ndarray, meta: List[Dict[str, str]], out: Path) -> None:
    errors = np.mean(np.abs(y_true - y_pred), axis=1)
    df = pd.DataFrame(meta).copy()
    df["MAE"] = errors
    pivot = df.pivot_table(index="lead", columns="record", values="MAE", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.35), 4.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_title("Per-record / per-lead absolute error")
    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Plot Errors
def plot_per_lead_rmse(y_true: np.ndarray, y_pred: np.ndarray, meta: List[Dict[str, str]], out: Path) -> None:
    df = pd.DataFrame(meta).copy()
    df["RMSE"] = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    order = [lead for lead in LEADS if lead in set(df["lead"])]
    grouped = df.groupby("lead")["RMSE"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    grouped.plot.bar(ax=ax, color="#54A24B")
    ax.set_title("Average RMSE by ECG lead")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Generate Table
def plot_table(df: pd.DataFrame, out: Path, title: str, float_fmt: str = ".4f") -> None:
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: format(x, float_fmt))
    fig_h = max(2.0, 0.45 * len(display_df) + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title(title, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Plot Comparison
def plot_physical_vs_digital(records: List[Path], signals: np.ndarray, images: np.ndarray, meta: List[Dict[str, str]], out: Path) -> None:
    record_dir = records[0]
    page = sorted(record_dir.glob("*-0001.png"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    if page:
        img = Image.open(page[0]).convert("RGB")
        # Crop to waveform region for readability.
        w, h = img.size
        crop = img.crop((0, int(h * 0.33), w, int(h * 0.93)))
        axes[0].imshow(crop)
        axes[0].set_title("Physical scanned ECG image")
    else:
        axes[0].text(0.5, 0.5, "No original page image found", ha="center")
    axes[0].axis("off")

    axes[1].imshow(images[0, :, :, 0], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Generated paper strip input")
    axes[1].axis("off")

    axes[2].plot(signals[0], color="#4C78A8")
    axes[2].set_title(f"Digital CSV waveform: {meta[0]['record']} lead {meta[0]['lead']}")
    axes[2].set_xlabel("Resampled time index")
    axes[2].set_ylabel("Normalized mV")
    axes[2].grid(alpha=0.25)
    fig.suptitle("Physical image vs digitized ECG signal")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Plot Confusion
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, hp: HyperParameters, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(hp.classes))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Amplitude-bin confusion matrix")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

# Format Metrics
def classification_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float | str]:
    return {
        "model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

# Save Predictions
def save_prediction_csv(y_true: np.ndarray, y_pred: np.ndarray, meta: List[Dict[str, str]], out: Path) -> None:
    pred_df = pd.DataFrame(
        {
            "time_index": np.arange(y_true.shape[1]),
            "digital_csv": y_true[0],
            "cnn_digitized": y_pred[0],
            "residual": y_true[0] - y_pred[0],
            "record": meta[0]["record"],
            "lead": meta[0]["lead"],
        }
    )
    pred_df.to_csv(out / f"{meta[0]['record']}_{meta[0]['lead']}_prediction.csv", index=False)

# Format Table
def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Small Markdown table formatter that avoids pandas' optional tabulate dependency."""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.4f}")
        else:
            display[col] = display[col].astype(str)
    headers = [str(col) for col in display.columns]
    rows = display.values.tolist()
    widths = []
    for i, header in enumerate(headers):
        values = [str(row[i]) for row in rows]
        widths.append(max([len(header), *[len(v) for v in values]]))

    def fmt_row(values: Iterable[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row([str(v) for v in row]) for row in rows)
    return "\n".join(lines)

# Save Report
def write_report(
    out: Path,
    hp: HyperParameters,
    regression_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    n_records: int,
    n_samples: int,
) -> None:
    best_reg = regression_df.sort_values("RMSE").iloc[0]
    best_cls = classification_df.sort_values("F1_macro", ascending=False).iloc[0]
    report = f"""# PhysioNet ECG Image Digitization: Results and Analysis
...
"""
    (out / "results_report.md").write_text(report, encoding="utf-8")

# Main Execution
def run(args: argparse.Namespace) -> None:
    hp = HyperParameters(
        seed=args.seed,
        signal_length=args.signal_length,
        image_height=args.image_height,
        image_width=args.image_width,
        digitizer_epochs=args.digitizer_epochs,
        classifier_epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    if args.quick:
        hp.digitizer_epochs = min(hp.digitizer_epochs, 3)
        hp.classifier_epochs = min(hp.classifier_epochs, 3)

    set_seed(hp.seed)
    paths = ensure_dirs(args.output_dir)
    records = find_records(args.data_dir)
    if args.max_records:
        records = records[: args.max_records]

    signals, meta = load_signals(records, hp.signal_length)
    images = make_dataset(signals, hp)
    labels, peak_to_peak = make_labels(signals)

    idx = np.arange(len(signals))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=hp.test_size,
        random_state=hp.seed,
        stratify=labels if len(np.unique(labels)) == 3 else None,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=hp.validation_size,
        random_state=hp.seed,
        stratify=labels[train_idx] if len(np.unique(labels[train_idx])) == 3 else None,
    )

    x_train, y_train = images[train_idx], signals[train_idx]
    x_val, y_val = images[val_idx], signals[val_idx]
    x_test, y_test = images[test_idx], signals[test_idx]
    test_meta = [meta[i] for i in test_idx]

    digitizer = build_digitizer(hp)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, verbose=0),
    ]
    hist = digitizer.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=hp.digitizer_epochs,
        batch_size=hp.batch_size,
        verbose=1,
        callbacks=callbacks,
    )
    pred_cnn = digitizer.predict(x_test, batch_size=hp.batch_size, verbose=0).astype(np.float32)
    pred_mean = mean_signal_baseline(y_train, len(y_test))
    pred_trace = trace_baseline_from_image(x_test, hp.signal_length)

    regression_rows = []
    for model_name, pred in [
        ("Mean signal baseline", pred_mean),
        ("Darkest-pixel trace baseline", pred_trace),
        ("CNN/U-Net-style digitizer", pred_cnn),
    ]:
        row = {"model": model_name}
        row.update(regression_metrics(y_test, pred))
        regression_rows.append(row)
    regression_df = pd.DataFrame(regression_rows)
    regression_df.to_csv(paths["metrics"] / "regression_metrics.csv", index=False)

    plot_loss(hist, paths["plots"] / "loss_curve.png", "CNN digitizer loss curve")
    plot_model_comparison(regression_df, paths["plots"] / "model_comparison.png")
    plot_waveform_overlay(y_test, pred_cnn, test_meta, paths["plots"] / "waveform_overlay.png")
    plot_scatter_residuals(y_test, pred_cnn, paths["plots"] / "scatter_residuals.png")
    plot_error_heatmap(y_test, pred_cnn, test_meta, paths["plots"] / "error_heatmap.png")
    plot_per_lead_rmse(y_test, pred_cnn, test_meta, paths["plots"] / "per_lead_rmse.png")
    plot_physical_vs_digital(records, signals, images, meta, paths["plots"] / "physical_vs_digital_ecg.png")
    save_prediction_csv(y_test, pred_cnn, test_meta, paths["predictions"])

    del digitizer
    keras.backend.clear_session()

    # Downstream classification.
    y_cls_train, y_cls_val, y_cls_test = labels[train_idx], labels[val_idx], labels[test_idx]
    seq_train = signals[train_idx][..., None]
    seq_val = signals[val_idx][..., None]
    seq_test = signals[test_idx][..., None]

    majority = int(pd.Series(y_cls_train).mode().iloc[0])
    pred_majority = np.full_like(y_cls_test, majority)

    lstm = build_signal_classifier(hp)
    lstm_hist = lstm.fit(
        seq_train,
        y_cls_train,
        validation_data=(seq_val, y_cls_val),
        epochs=hp.classifier_epochs,
        batch_size=hp.batch_size,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    pred_lstm = np.argmax(lstm.predict(seq_test, batch_size=hp.batch_size, verbose=0), axis=1)
    del lstm
    keras.backend.clear_session()

    multimodal = build_multimodal_classifier(hp)
    multimodal.fit(
        [images[train_idx], seq_train],
        y_cls_train,
        validation_data=([images[val_idx], seq_val], y_cls_val),
        epochs=hp.classifier_epochs,
        batch_size=hp.batch_size,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    pred_multi = np.argmax(multimodal.predict([x_test, seq_test], batch_size=hp.batch_size, verbose=0), axis=1)

    classification_rows = [
        classification_metrics_table(y_cls_test, pred_majority, "Majority class baseline"),
        classification_metrics_table(y_cls_test, pred_lstm, "LSTM signal classifier"),
        classification_metrics_table(y_cls_test, pred_multi, "ResNet-style + LSTM fusion"),
    ]
    classification_df = pd.DataFrame(classification_rows)
    classification_df.to_csv(paths["metrics"] / "classification_metrics.csv", index=False)
    plot_loss(lstm_hist, paths["plots"] / "classification_loss_curve.png", "LSTM classifier loss curve")
    plot_confusion(y_cls_test, pred_multi, hp, paths["plots"] / "amplitude_bin_confusion_matrix.png")

    hyper_df = pd.DataFrame(
        [{"hyperparameter": key, "value": json.dumps(value) if isinstance(value, (tuple, list)) else value} for key, value in asdict(hp).items()]
    )
    hyper_df.to_csv(paths["metrics"] / "hyperparameters.csv", index=False)
    plot_table(regression_df, paths["plots"] / "regression_metrics_table.png", "Regression Metrics")
    plot_table(classification_df, paths["plots"] / "classification_metrics_table.png", "Classification Metrics")
    plot_table(hyper_df, paths["plots"] / "hyperparameters_table.png", "Hyperparameters", float_fmt=".6g")

    write_report(paths["root"], hp, regression_df, classification_df, len(records), len(signals))

    print("\nDone.")
    print(f"Records: {len(records)}")
    print(f"Lead samples: {len(signals)}")
    print(f"Outputs: {paths['root'].resolve()}")
    print("\nRegression metrics:")
    print(regression_df.to_string(index=False))
    print("\nClassification metrics:")
    print(classification_df.to_string(index=False))

# CLI Parser
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECG image digitization experiment runner")
    parser.add_argument("--data-dir", type=Path, default=Path("dl-lms-data") / "sample_data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-records", type=int, default=0, help="Limit records for faster experimentation; 0 means all.")
    parser.add_argument("--quick", action="store_true", help="Use at most 3 epochs per model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signal-length", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--digitizer-epochs", type=int, default=8)
    parser.add_argument("--classifier-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if args.max_records == 0:
        args.max_records = None
    return args


if __name__ == "__main__":
    run(parse_args())
