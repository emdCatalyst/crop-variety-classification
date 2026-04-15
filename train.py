#!/usr/bin/env python3
"""
Crop Variety Classification — MuST-C Dataset
CNN + LSTM hybrid model with Random Forest and SVM baselines.

Usage:
    python3 train.py              # full training
    python3 train.py --test       # sanity check on 5 plots, 3 epochs
"""
from model import CropCNNLSTM, CNNEncoder
import os, sys, glob, warnings, argparse, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, precision_score,
                              recall_score)
try:
    import rasterio
except ImportError:
    sys.exit("rasterio not found. Run: pip install rasterio")

# CONFIG
CONFIG = {
    "plots_dir":        "./mustc_plots",
    "labels_csv":       "./metadata/plot_metadata.csv",
    "results_dir":      "./results",
    "num_timesteps":    12,
    "patch_size":       32,
    "patches_per_plot": 48,
    "batch_size":       32,
    "epochs":           60,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "lstm_hidden":      128,
    "cnn_out":          64,
    "dropout":          0.3,
    "early_stop":       15,
    "seed":             42,
    "num_workers":      0,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# RUN DIRECTORY

def make_run_dir(config, num_plots_loaded, test_mode):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "_TEST" if test_mode else ""
    name = (
        f"run_{ts}"
        f"_ep{config['epochs']}"
        f"_plots{num_plots_loaded}"
        f"_p{config['patches_per_plot']}"
        f"_ps{config['patch_size']}"
        f"_lr{config['lr']}"
        f"_lstm{config['lstm_hidden']}"
        f"{mode}"
    )
    path = os.path.join(config["results_dir"], name)
    os.makedirs(path, exist_ok=True)
    return path


def save_run_config(run_dir, config, num_plots, num_classes, classes,
                    num_params, train_size, val_size, test_mode, start_time):
    """
    run_config.json — everything needed to reproduce this run exactly.
    Written BEFORE training starts so it exists even if training crashes.
    """
    doc = {
        "run": {
            "timestamp":    start_time.isoformat(),
            "test_mode":    test_mode,
            "device":       str(DEVICE),
            "run_dir":      run_dir,
        },
        "dataset": {
            "plots_dir":      config["plots_dir"],
            "labels_csv":     config["labels_csv"],
            "plots_loaded":   num_plots,
            "num_classes":    num_classes,
            "classes":        classes,
            "train_samples":  train_size,
            "val_samples":    val_size,
        },
        "model": {
            "architecture":   "CNN+LSTM",
            "trainable_params": num_params,
            "in_channels":    13,
            "cnn_out":        config["cnn_out"],
            "lstm_hidden":    config["lstm_hidden"],
            "lstm_layers":    2,
            "dropout":        config["dropout"],
        },
        "training": {
            "epochs_max":     config["epochs"],
            "batch_size":     config["batch_size"],
            "lr":             config["lr"],
            "weight_decay":   config["weight_decay"],
            "early_stop_patience": config["early_stop"],
            "optimiser":      "AdamW",
            "lr_scheduler":   "CosineAnnealingLR",
            "loss":           "CrossEntropyLoss(label_smoothing=0.05)",
        },
        "preprocessing": {
            "num_timesteps":    config["num_timesteps"],
            "patch_size":       config["patch_size"],
            "patches_per_plot": config["patches_per_plot"],
            "bands_raw":        10,
            "bands_computed":   3,
            "bands_total":      13,
            "vi_added":         ["NDVI", "NDRE", "SAVI"],
            "normalisation":    "divide by 10000, clip [0,1]",
            "seed":             config["seed"],
        },
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Run directory: {run_dir}/")


def save_results_json(run_dir, baseline_results, dl_acc, dl_f1,
                      history, start_time, y_true, y_pred_dl,
                      rf_pred, svm_pred, y_val, classes):
    """
    results.json — all final metrics in one machine-readable file.
    Written AFTER training completes.
    """
    elapsed = int((datetime.now() - start_time).total_seconds())

    def metrics_dict(y_true_, y_pred_, name):
        acc  = accuracy_score(y_true_, y_pred_)
        f1w  = f1_score(y_true_, y_pred_, average="weighted",  zero_division=0)
        f1m  = f1_score(y_true_, y_pred_, average="macro",     zero_division=0)
        prec = precision_score(y_true_, y_pred_, average="weighted", zero_division=0)
        rec  = recall_score(y_true_, y_pred_,    average="weighted", zero_division=0)
        return {
            "model":        name,
            "accuracy":     round(acc,  4),
            "f1_weighted":  round(f1w,  4),
            "f1_macro":     round(f1m,  4),
            "precision_w":  round(prec, 4),
            "recall_w":     round(rec,  4),
        }

    # Per-class metrics for CNN+LSTM
    present       = sorted(set(y_true) | set(y_pred_dl))
    present_names = [classes[i] for i in present]
    per_class_raw = classification_report(
        y_true, y_pred_dl, labels=present,
        target_names=present_names, zero_division=0, output_dict=True
    )
    per_class = {k: {m: round(v, 4) for m, v in vals.items()}
                 for k, vals in per_class_raw.items()
                 if k not in ("accuracy", "macro avg", "weighted avg")}

    doc = {
        "run_timestamp":   start_time.isoformat(),
        "training_seconds": elapsed,
        "epochs_completed": len(history["epoch"]),
        "best_f1_epoch":   history["val_f1"].index(max(history["val_f1"])) + 1,
        "models": [
            metrics_dict(y_val, rf_pred,   "Random Forest"),
            metrics_dict(y_val, svm_pred,  "SVM"),
            metrics_dict(y_true, y_pred_dl, "CNN+LSTM"),
        ],
        "cnn_lstm_per_class": per_class,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("Saved results.json")


def save_history_csv(run_dir, history):
    """history.csv — epoch-by-epoch training metrics."""
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    print("Saved history.csv")


def save_per_class_csv(run_dir, y_true, y_pred, classes, model_name):
    """
    per_class_{model}.csv — one row per class with precision/recall/F1/support.
    Clean format, easy to paste into a thesis table.
    """
    present       = sorted(set(y_true) | set(y_pred))
    present_names = [classes[i] for i in present]
    report_dict   = classification_report(
        y_true, y_pred, labels=present,
        target_names=present_names, zero_division=0, output_dict=True
    )
    rows = []
    for cls in present_names:
        if cls in report_dict:
            r = report_dict[cls]
            rows.append({
                "class":     cls,
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"],    4),
                "f1_score":  round(r["f1-score"],  4),
                "support":   int(r["support"]),
            })
    fname = f"per_class_{model_name.lower().replace(' ', '_')}.csv"
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, fname), index=False)
    print(f"Saved {fname}")


def save_confusion_csv(run_dir, y_true, y_pred, classes, model_name):
    """confusion_{model}.csv — confusion matrix as a labelled CSV."""
    present       = sorted(set(y_true) | set(y_pred))
    present_names = [classes[i] for i in present]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    df = pd.DataFrame(cm, index=present_names, columns=present_names)
    df.index.name = "true / predicted"
    fname = f"confusion_{model_name.lower().replace(' ', '_')}.csv"
    df.to_csv(os.path.join(run_dir, fname))
    print(f"Saved {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(folder_name):
    base = folder_name.split("-")[0]
    base = base[:6]
    try:
        return datetime.strptime(base, "%y%m%d")
    except ValueError:
        return datetime.min


def find_ms_tifs(plot_id, plots_dir, num_timesteps):
    inner = os.path.join(plots_dir, f"plot_{plot_id}", "plot-wise", f"plot{plot_id}")
    if not os.path.isdir(inner):
        return []
    date_folders = sorted(
        [d for d in os.listdir(inner) if os.path.isdir(os.path.join(inner, d))],
        key=parse_date
    )
    tif_paths = []
    for df in date_folders:
        ms_dir = os.path.join(inner, df, "raster_data", "UAV3-MS")
        if not os.path.isdir(ms_dir):
            continue
        tifs = sorted(glob.glob(os.path.join(ms_dir, "*.tif")))
        if tifs:
            tif_paths.append(tifs[0])
        if len(tif_paths) >= num_timesteps:
            break
    return tif_paths


def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    if data.max() > 1.5:
        data = data / 10000.0
    # Replace NoData (NaN, inf, negative) with 0 before any further processing
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0)


def compute_vi(bands):
    eps = 1e-8
    NIR, Red, RE = bands[9], bands[4], bands[6]
    NDVI = (NIR - Red)  / (NIR + Red  + eps)
    NDRE = (NIR - RE)   / (NIR + RE   + eps)
    SAVI = 1.5 * (NIR - Red) / (NIR + Red + 0.5 + eps)
    return np.concatenate([bands, NDVI[None], NDRE[None], SAVI[None]], axis=0)

def load_dataset(config, max_plots=None):
    meta = pd.read_csv(config["labels_csv"])
    meta["label_str"] = meta["species"] + "_" + meta["variety"]
    classes = sorted(meta["label_str"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    print(f"\n{len(classes)} classes: {classes}")

    if max_plots:
        meta = meta.head(max_plots)

    T, PS, NP = config["num_timesteps"], config["patch_size"], config["patches_per_plot"]
    X_list, y_list = [], []
    skipped = 0

    for _, row in meta.iterrows():
        pid   = row["plot_id"]
        label = c2i[row["label_str"]]
        tifs  = find_ms_tifs(pid, config["plots_dir"], T)
        if len(tifs) < T:
            print(f"  skip plot {pid}: only {len(tifs)}/{T} MS dates")
            skipped += 1; continue
        try:
            plot_data = np.stack([compute_vi(read_tif(t)) for t in tifs[:T]])
        except Exception as e:
            print(f"  skip plot {pid}: {e}")
            skipped += 1; continue
        _, C, H, W = plot_data.shape
        if H < PS or W < PS:
            print(f"  skip plot {pid}: image too small ({H}x{W})")
            skipped += 1; continue
        for _ in range(NP):
            y0 = np.random.randint(0, H - PS + 1)
            x0 = np.random.randint(0, W - PS + 1)
            X_list.append(plot_data[:, :, y0:y0+PS, x0:x0+PS])
            y_list.append(label)

    n_loaded = len(meta) - skipped
    print(f"Loaded {len(X_list)} patches from {n_loaded}/{len(meta)} plots  ({skipped} skipped)")
    return (np.stack(X_list).astype(np.float32),
            np.array(y_list, dtype=np.int64),
            classes, n_loaded)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & DATALOADER
# ─────────────────────────────────────────────────────────────────────────────

class MuSTCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class AugmentedDataset(Dataset):
    """Wraps MuSTCDataset and applies random flips during training."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]  # (T, C, H, W)
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[-1])   # horizontal flip
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[-2])   # vertical flip
        return x, self.y[i]
def make_loaders(X, y, config):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config["seed"])
    print(f"Train: {len(y_tr)} | Val: {len(y_val)}")
    tr = DataLoader(AugmentedDataset(X_tr, y_tr), batch_size=config["batch_size"],
                    shuffle=True,  num_workers=config["num_workers"])
    va = DataLoader(MuSTCDataset(X_val, y_val), batch_size=config["batch_size"],
                    shuffle=False, num_workers=config["num_workers"])
    return tr, va, X_tr, X_val, y_tr, y_val


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimiser, training):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)
            if training:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total


def train_model(model, tr_loader, va_loader, config, run_dir):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt       = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epochs"])

    history = {"epoch": [], "tr_loss": [], "va_loss": [],
               "tr_acc": [], "va_acc": [], "val_f1": []}
    best_f1, patience = 0.0, 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, tr_loader, criterion, opt, True)
        va_loss, va_acc = run_epoch(model, va_loader, criterion, opt, False)
        scheduler.step()

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for X, y in va_loader:
                all_pred.extend(model(X.to(DEVICE)).argmax(1).cpu().tolist())
                all_true.extend(y.tolist())
        val_f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

        history["epoch"].append(epoch)
        history["tr_loss"].append(round(tr_loss, 4))
        history["va_loss"].append(round(va_loss, 4))
        history["tr_acc"].append(round(tr_acc,   4))
        history["va_acc"].append(round(va_acc,   4))
        history["val_f1"].append(round(val_f1,   4))

        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"loss {tr_loss:.3f}/{va_loss:.3f} | "
              f"acc {tr_acc*100:.1f}%/{va_acc*100:.1f}% | "
              f"val_F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config["early_stop"]:
                print(f"Early stop at epoch {epoch}  (best F1={best_f1:.3f})")
                break

    save_history_csv(run_dir, history)
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(run_dir, "cnn_lstm.pth"))
    print("Saved cnn_lstm.pth")
    return history, best_f1


def evaluate_model(model, va_loader, classes):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in va_loader:
            all_pred.extend(model(X.to(DEVICE)).argmax(1).cpu().tolist())
            all_true.extend(y.tolist())
    present       = sorted(set(all_true) | set(all_pred))
    present_names = [classes[i] for i in present]
    print("\n=== CNN+LSTM ===")
    print(classification_report(all_true, all_pred, labels=present,
                                target_names=present_names, zero_division=0))
    return all_true, all_pred

# BASELINES

def train_baselines(X_tr, X_val, y_tr, y_val):
    Xf_tr  = X_tr.reshape(len(X_tr),  -1)
    Xf_val = X_val.reshape(len(X_val), -1)

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                random_state=42, class_weight="balanced")
    rf.fit(Xf_tr, y_tr)
    rf_pred = rf.predict(Xf_val)
    rf_acc  = accuracy_score(y_val, rf_pred)
    rf_f1   = f1_score(y_val, rf_pred, average="weighted", zero_division=0)
    print(f"RF  -> Acc={rf_acc*100:.1f}%  F1={rf_f1:.3f}")

    print("Training SVM (PCA-50)...")
    n_comp = min(50, Xf_tr.shape[1], Xf_tr.shape[0]-1)
    pca    = PCA(n_components=n_comp, random_state=42)
    svm    = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    svm.fit(pca.fit_transform(Xf_tr), y_tr)
    svm_pred = svm.predict(pca.transform(Xf_val))
    svm_acc  = accuracy_score(y_val, svm_pred)
    svm_f1   = f1_score(y_val, svm_pred, average="weighted", zero_division=0)
    print(f"SVM -> Acc={svm_acc*100:.1f}%  F1={svm_f1:.3f}")

    baseline_results = {
        "Random Forest": (rf_acc, rf_f1),
        "SVM":           (svm_acc, svm_f1),
    }
    return baseline_results, rf_pred, svm_pred

# PLOTS

def save_training_curves(history, run_dir):
    ep = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ep, history["tr_loss"], label="Train")
    ax1.plot(ep, history["va_loss"], label="Val")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="CrossEntropy")
    ax1.legend(); ax1.grid(True, alpha=.3)
    ax2.plot(ep, [a*100 for a in history["tr_acc"]], label="Train")
    ax2.plot(ep, [a*100 for a in history["va_acc"]], label="Val")
    ax2.set(title="Accuracy", xlabel="Epoch", ylabel="%")
    ax2.legend(); ax2.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
    plt.close(); print("Saved training_curves.png")


def save_confusion_png(y_true, y_pred, classes, fname, title, run_dir):
    present       = sorted(set(y_true) | set(y_pred))
    present_names = [classes[i] for i in present]
    cm  = confusion_matrix(y_true, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(max(8, len(present)), max(6, len(present)-1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, fname), dpi=150)
    plt.close(); print(f"Saved {fname}")


def save_comparison(baseline_results, dl_acc, dl_f1, run_dir):
    models = list(baseline_results.keys()) + ["CNN+LSTM"]
    accs   = [v[0]*100 for v in baseline_results.values()] + [dl_acc*100]
    f1s    = [v[1] for v in baseline_results.values()] + [dl_f1]
    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="#4C9BE8")
    ax.bar(x + w/2, [f*100 for f in f1s], w, label="F1 x 100", color="#E87C4C")
    ax.set(title="Model Comparison", xticks=x, xticklabels=models, ylabel="Score (%)")
    ax.legend(); ax.set_ylim(0, 110); ax.grid(axis="y", alpha=.3)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i-w/2, a+1, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i+w/2, f*100+1, f"{f:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "model_comparison.png"), dpi=150)
    plt.close(); print("Saved model_comparison.png")

# MAIN

def main(test_mode=False):
    start_time = datetime.now()
    print("=" * 60)
    print("  MuST-C Crop Classification  —  CNN+LSTM")
    print("=" * 60)

    cfg = CONFIG.copy()
    if test_mode:
        cfg["epochs"]           = 3
        cfg["patches_per_plot"] = 4
        print("\n[TEST MODE] 5 plots, 3 epochs, 4 patches/plot")

    X, y, classes, n_plots = load_dataset(cfg, max_plots=5 if test_mode else None)
    if len(X) == 0:
        sys.exit("No data loaded — check plots_dir and labels_csv paths.")

    tr_loader, va_loader, X_tr, X_val, y_tr, y_val = make_loaders(X, y, cfg)

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    run_dir = make_run_dir(cfg, n_plots, test_mode)

    model = CropCNNLSTM(
        num_classes = len(classes),
        in_ch       = 13,
        cnn_out     = cfg["cnn_out"],
        lstm_hidden = cfg["lstm_hidden"],
        dropout     = cfg["dropout"],
    ).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")

    save_run_config(run_dir, cfg, n_plots, len(classes), classes,
                    num_params, len(y_tr), len(y_val), test_mode, start_time)

    history, best_f1 = train_model(model, tr_loader, va_loader, cfg, run_dir)
    y_true, y_pred_dl = evaluate_model(model, va_loader, classes)
    dl_acc = accuracy_score(y_true, y_pred_dl)

    baseline_results, rf_pred, svm_pred = train_baselines(X_tr, X_val, y_tr, y_val)

    save_results_json(run_dir, baseline_results, dl_acc, best_f1,
                      history, start_time, y_true, y_pred_dl,
                      rf_pred, svm_pred, y_val, classes)

    for model_name, y_p in [("cnn_lstm",       y_pred_dl),
                             ("random_forest",  rf_pred),
                             ("svm",            svm_pred)]:
        y_t = y_true if model_name == "cnn_lstm" else list(y_val)
        save_per_class_csv(run_dir, y_t, y_p, classes, model_name)
        save_confusion_csv(run_dir, y_t, y_p, classes, model_name)

    save_training_curves(history, run_dir)
    save_confusion_png(y_true, y_pred_dl, classes,
                       "cm_cnn_lstm.png", "CNN+LSTM Confusion Matrix", run_dir)
    save_comparison(baseline_results, dl_acc, best_f1, run_dir)

    elapsed = int((datetime.now() - start_time).total_seconds())
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for name, (acc, f1) in baseline_results.items():
        print(f"  {name:<20} Acc={acc*100:.1f}%  F1={f1:.3f}")
    print(f"  {'CNN+LSTM':<20} Acc={dl_acc*100:.1f}%  F1={best_f1:.3f}")
    print(f"\nDuration: {elapsed//60}m {elapsed%60}s")
    print(f"All results saved to: {run_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Quick sanity check: 5 plots, 3 epochs")
    args = parser.parse_args()
    main(test_mode=args.test)