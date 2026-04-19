#!/usr/bin/env python3
"""
Shared inference core.

"""

import os
import re
import glob
import base64
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

import rasterio
from model import CropCNNLSTM

CONFIG = {
    "model_path": "./results/cnn_lstm.pth",
    "plots_dir": "./mustc_plots",
    "labels_csv": "./metadata/plot_metadata.csv",
    "output_dir": "./classification_maps",
    "num_timesteps": 12,
    "patch_size": 32,
    "lstm_hidden": 128,
    "cnn_out": 64,
    "dropout": 0.0,
    "stride": 8,
    "batch_size": 64,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_GENERIC_ADVICE = {
    "CRITICAL": "Immediate field inspection required. Potential crop failure, severe water deficit, or disease outbreak.",
    "STRESSED": "Check irrigation schedule and soil moisture. Inspect for pest or disease pressure. Consider foliar nutrient application.",
    "MODERATE": "Monitor closely over the next 7-10 days. Verify fertilisation plan is on schedule.",
    "HEALTHY": "Crop developing normally. Continue standard management protocol.",
    "VIGOROUS": "Excellent canopy development. Verify no lodging risk for dense-canopy crops.",
}

_CROP_ADVICE = {
    "Maize": {
        "STRESSED": "Maize NDRE decline often precedes visible wilting by 5-7 days. Check soil moisture at 30 cm depth and inspect for rootworm damage.",
        "CRITICAL": "Severe Maize stress. Assess for Northern Corn Leaf Blight or Gray Leaf Spot if conditions are humid.",
    },
    "Wheat": {
        "STRESSED": "Wheat NDRE drop frequently indicates nitrogen deficiency. Consider topdressing (30-40 kg N/ha) if at tillering or stem extension stage.",
        "MODERATE": "Verify fungicide schedule if humid conditions persist — Septoria risk is elevated.",
    },
    "Potato": {
        "STRESSED": "Check irrigation uniformity and Late Blight (Phytophthora) pressure. Potato canopy collapse is rapid once disease spreads.",
        "CRITICAL": "Severe Potato stress — inspect immediately for Late Blight. Full canopy loss can occur within 7 days under humid conditions.",
    },
    "Soybean": {
        "STRESSED": "NDRE drop may indicate iron deficiency chlorosis (IDC) or soybean cyst nematode. Check nodulation status at root level.",
    },
    "SugarBeet": {
        "STRESSED": "Check for Cercospora leaf spot and verify soil pH. Low NDRE in SugarBeet correlates with nitrogen or boron deficiency.",
        "MODERATE": "Monitor leaf area index. SugarBeet is sensitive to water stress during the root-filling phase.",
    },
    "Intercrop": {
        "MODERATE": "Intercrop canopy competition naturally lowers per-species NDVI. Assess species balance rather than total NDVI alone.",
        "STRESSED": "One intercrop component may be suppressing the other. Inspect competitive balance between species.",
    },
}


def parse_date(folder_name: str):
    base = folder_name.split("-")[0][:6]
    try:
        return datetime.strptime(base, "%y%m%d")
    except ValueError:
        return datetime.min


def auto_find_model(results_dir="./results"):
    candidates = glob.glob(os.path.join(results_dir, "cnn_lstm.pth"))
    candidates += sorted(glob.glob(os.path.join(results_dir, "run_*", "cnn_lstm.pth")))
    return candidates[-1] if candidates else None


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return value or "input_sequence"


def canonical_crop_species(predicted_crop: str) -> str:
    base = str(predicted_crop).split("_")[0].replace(" ", "")
    mapping = {
        "Sugar": "SugarBeet",
        "SugarBeet": "SugarBeet",
        "Sugarbeet": "SugarBeet",
        "Maize": "Maize",
        "Wheat": "Wheat",
        "Potato": "Potato",
        "Soybean": "Soybean",
        "Intercrop": "Intercrop",
        "Mixture": "Intercrop",
    }
    return mapping.get(base, base)


def find_ms_tifs(plot_id, plots_dir, num_timesteps):
    inner = os.path.join(plots_dir, f"plot_{plot_id}", "plot-wise", f"plot{plot_id}")
    if not os.path.isdir(inner):
        return [], None, None

    date_folders = sorted(
        [d for d in os.listdir(inner) if os.path.isdir(os.path.join(inner, d))],
        key=parse_date,
    )

    tif_paths, transform, crs = [], None, None
    for df in date_folders:
        ms_dir = os.path.join(inner, df, "raster_data", "UAV3-MS")
        if not os.path.isdir(ms_dir):
            continue
        tifs = sorted(glob.glob(os.path.join(ms_dir, "*.tif")))
        if tifs:
            if transform is None:
                with rasterio.open(tifs[0]) as src:
                    transform = src.transform
                    crs = src.crs
            tif_paths.append(tifs[0])
        if len(tif_paths) >= num_timesteps:
            break

    return tif_paths, transform, crs


def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs

    if data.max() > 1.5:
        data = data / 10000.0

    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0), transform, crs


def compute_vi(bands):
    eps = 1e-8
    nir, red, re = bands[9], bands[4], bands[6]
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - re) / (nir + re + eps)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    return np.concatenate([bands, ndvi[None], ndre[None], savi[None]], axis=0)


def load_plot_rasters(plot_id, cfg):
    tif_paths, transform, crs = find_ms_tifs(plot_id, cfg["plots_dir"], cfg["num_timesteps"])
    if len(tif_paths) < cfg["num_timesteps"]:
        raise ValueError(f"Plot {plot_id}: only {len(tif_paths)}/{cfg['num_timesteps']} dates found")

    plot_data = np.stack([compute_vi(read_tif(p)[0]) for p in tif_paths[:cfg["num_timesteps"]]])
    return plot_data, transform, crs


def load_tif_sequence(tif_paths: Sequence[str], cfg):
    tif_paths = list(tif_paths)
    if len(tif_paths) != cfg["num_timesteps"]:
        raise ValueError(f"Expected exactly {cfg['num_timesteps']} GeoTIFF files, got {len(tif_paths)}")

    bands_list = []
    ref_shape = None
    ref_transform = None
    ref_crs = None

    for path in tif_paths:
        bands, transform, crs = read_tif(path)
        if bands.shape[0] < 10:
            raise ValueError(f"GeoTIFF '{os.path.basename(path)}' has {bands.shape[0]} bands; at least 10 are required")
        if ref_shape is None:
            ref_shape = bands.shape
            ref_transform = transform
            ref_crs = crs
        elif bands.shape != ref_shape:
            raise ValueError(
                f"All GeoTIFFs must have the same shape. Expected {ref_shape}, got {bands.shape} for '{os.path.basename(path)}'"
            )
        bands_list.append(compute_vi(bands))

    plot_data = np.stack(bands_list)
    return plot_data, ref_transform, ref_crs


def build_class_palette(class_names):
    base = list(plt.cm.get_cmap("tab20").colors) + list(plt.cm.get_cmap("Set3").colors)
    colors = np.array([base[i % len(base)] for i in range(len(class_names))], dtype=np.float32)
    return ListedColormap(colors.tolist()), colors


def shorten(label, n=30):
    return label if len(label) <= n else label[: n - 3] + "..."


def sliding_window_pixel_inference(model, plot_data, n_classes, cfg):
    _, _, H, W = plot_data.shape
    ps, stride = cfg["patch_size"], cfg["stride"]
    if H < ps or W < ps:
        raise ValueError(f"Raster ({H}x{W}) smaller than patch size ({ps})")

    n_rows = (H - ps) // stride + 1
    n_cols = (W - ps) // stride + 1
    patches, positions = [], []

    for r in range(n_rows):
        for c in range(n_cols):
            y0 = min(r * stride, H - ps)
            x0 = min(c * stride, W - ps)
            y1, x1 = y0 + ps, x0 + ps
            patches.append(plot_data[:, :, y0:y1, x0:x1])
            positions.append((r, c, y0, y1, x0, x1))

    coarse_prob_cube = np.zeros((n_classes, n_rows, n_cols), dtype=np.float32)
    pixel_prob_sum = np.zeros((n_classes, H, W), dtype=np.float32)
    pixel_count = np.zeros((H, W), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(patches), cfg["batch_size"]):
            batch_np = np.stack(patches[start : start + cfg["batch_size"]])
            batch = torch.from_numpy(batch_np).float().to(DEVICE)
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()

            for j, prob_vec in enumerate(probs):
                idx = start + j
                r, c, y0, y1, x0, x1 = positions[idx]
                coarse_prob_cube[:, r, c] = prob_vec
                pixel_prob_sum[:, y0:y1, x0:x1] += prob_vec[:, None, None]
                pixel_count[y0:y1, x0:x1] += 1.0

    pixel_prob_cube = pixel_prob_sum / np.maximum(pixel_count[None, :, :], 1.0)
    coarse_conf_grid = coarse_prob_cube.max(axis=0)
    pixel_conf_map = pixel_prob_cube.max(axis=0)
    return coarse_prob_cube, pixel_prob_cube, coarse_conf_grid, pixel_conf_map


def smooth_pixel_prob_cube(pixel_prob_cube, sigma):
    if sigma <= 0 or not SCIPY_OK:
        return pixel_prob_cube
    return np.stack(
        [
            gaussian_filter(pixel_prob_cube[c], sigma=sigma, mode="reflect")
            for c in range(pixel_prob_cube.shape[0])
        ]
    ).astype(np.float32)


def hard_class_map_from_probs(pixel_prob_cube):
    return pixel_prob_cube.argmax(axis=0).astype(np.int16)


def probability_rgb(pixel_prob_cube, class_colors, softness_gamma=1.0):
    probs = np.clip(pixel_prob_cube, 1e-8, 1.0)
    probs = probs ** softness_gamma
    probs /= probs.sum(axis=0, keepdims=True)
    rgb = np.einsum("chw,cd->hwd", probs, class_colors[:, :3])
    return np.clip(rgb, 0.0, 1.0)


def one_hot_from_labels(labels, n_classes):
    return np.eye(n_classes, dtype=np.float32)[labels].transpose(2, 0, 1)


def renormalize_prob_cube(prob_cube, eps=1e-8):
    prob_cube = np.clip(prob_cube, eps, None)
    prob_cube /= prob_cube.sum(axis=0, keepdims=True)
    return prob_cube.astype(np.float32)


def make_display_probability_cube(raw_pixel_prob_cube, raw_hard_class_map, n_classes, sigma):
    if sigma <= 0 or not SCIPY_OK:
        return raw_pixel_prob_cube

    base_sigma = max(0.8, 1.35 * sigma)
    support_sigma = max(0.6, 0.80 * sigma)

    base = smooth_pixel_prob_cube(raw_pixel_prob_cube, sigma=base_sigma)
    support = one_hot_from_labels(raw_hard_class_map, n_classes)
    support = smooth_pixel_prob_cube(support, sigma=support_sigma)

    display = 0.88 * base + 0.12 * support
    display = np.maximum(display, 0.10 * support)
    return renormalize_prob_cube(display)


def display_confidence_map(raw_pixel_conf_map, smooth, sigma):
    if not smooth or sigma <= 0 or not SCIPY_OK:
        return raw_pixel_conf_map
    return np.clip(
        gaussian_filter(raw_pixel_conf_map, sigma=max(0.8, 1.25 * sigma), mode="reflect"),
        0.0,
        1.0,
    )


def assess_health(ndvi_mean, ndre_mean, savi_mean, crop_species, temporal_trend):
    if ndvi_mean < 0.20:
        status = "CRITICAL"
    elif ndvi_mean < 0.35:
        status = "STRESSED"
    elif ndvi_mean < 0.55:
        status = "STRESSED" if ndre_mean < 0.25 else "MODERATE"
    elif ndvi_mean < 0.75:
        status = "HEALTHY"
    else:
        status = "VIGOROUS"

    if temporal_trend < -0.10:
        trajectory = "DECLINING (significant drop across season)"
    elif temporal_trend < -0.03:
        trajectory = "SLIGHTLY DECLINING"
    elif temporal_trend > 0.10:
        trajectory = "STRONG GROWTH"
    elif temporal_trend > 0.03:
        trajectory = "GROWING"
    else:
        trajectory = "STABLE"

    advice = _CROP_ADVICE.get(crop_species, {}).get(status) or _GENERIC_ADVICE[status]

    return {
        "ndvi_mean": round(float(ndvi_mean), 3),
        "ndre_mean": round(float(ndre_mean), 3),
        "savi_mean": round(float(savi_mean), 3),
        "health_status": status,
        "trajectory": trajectory,
        "advice": advice,
    }


def write_class_geotiff(grid, transform, crs, out_path, dtype="int16", nodata=-1):
    h, w = grid.shape
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        dtype=dtype,
        width=w,
        height=h,
        count=1,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(grid, 1)


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def render_classification_map(
    raw_hard_class_map,
    display_prob_cube,
    conf_map_for_display,
    raw_pixel_conf_map,
    source_name,
    class_names,
    out_dir,
    with_confidence=False,
    sigma=0.0,
    smooth_render=False,
    subtitle=None,
):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_name(source_name)
    cmap, class_colors = build_class_palette(class_names)

    if smooth_render:
        vis_main = probability_rgb(display_prob_cube, class_colors, softness_gamma=1.0)
        sigma_note = f"  (probability-flow Gaussian display sigma={sigma:.1f})"
        conf_title = "Smoothed Confidence Heatmap"
        map_interp = "bicubic"
        conf_interp = "bicubic"
    else:
        vis_main = raw_hard_class_map
        sigma_note = ""
        conf_title = "Pixel-space Confidence Heatmap"
        map_interp = "nearest"
        conf_interp = "nearest"

    ncols = 2 if with_confidence else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7), dpi=150)
    ax_map = axes[0] if with_confidence else axes
    ax_conf = axes[1] if with_confidence else None

    if smooth_render:
        ax_map.imshow(vis_main, interpolation=map_interp)
    else:
        ax_map.imshow(
            vis_main,
            cmap=cmap,
            interpolation=map_interp,
            vmin=0,
            vmax=len(class_names) - 1,
        )

    subtitle = subtitle or f"Source: {shorten(source_name, 26)}"
    ax_map.set_title(f"Classification Map", fontsize=10, fontweight="bold")
    ax_map.axis("off")

    vals, counts = np.unique(raw_hard_class_map, return_counts=True)
    order = np.argsort(counts)[::-1]
    vals = vals[order]
    dominant = int(vals[0])

    handles = [
        mpatches.Patch(color=class_colors[i], label=shorten(class_names[i], 36))
        for i in vals[:12]
    ]
    ax_map.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        title="Predicted class",
        title_fontsize=8.5,
        framealpha=0.92,
    )

    info = (
        f"Dominant class: {shorten(class_names[dominant], 26)}\n"\
        f"{subtitle}\n"
        f"Avg. confidence: {raw_pixel_conf_map.mean():.2f}\n"
        f"Pixel map size : {raw_hard_class_map.shape[1]} x {raw_hard_class_map.shape[0]}"
        
    )
    ax_map.text(
        0.01,
        0.99,
        info,
        transform=ax_map.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#111111", alpha=0.82),
    )

    if ax_conf is not None:
        im2 = ax_conf.imshow(
            conf_map_for_display,
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            interpolation=conf_interp,
        )
        ax_conf.set_title(conf_title, fontsize=10, fontweight="bold")
        ax_conf.axis("off")
        cbar = plt.colorbar(im2, ax=ax_conf, fraction=0.046, pad=0.04)
        cbar.set_label("Confidence (0 to 1)", fontsize=8)

    plt.suptitle(
        f"{source_name}\n{sigma_note}",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    suffix = f"_pixelsmooth{sigma:.1f}" if smooth_render else "_raw"
    out_png = os.path.join(out_dir, f"classification_map_{safe_name}{suffix}.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    return out_png


@dataclass
class InferenceArtifacts:
    source_name: str
    predicted_crop: str
    health_status: str
    advice: str
    mean_confidence: float
    mean_ndvi: float
    map_png_path: str
    geotiff_path: str
    class_distribution: Dict[str, float]
    ground_truth_crop: str = ""
    plot_id: Optional[int] = None
    ndre_mean: float = 0.0
    savi_mean: float = 0.0
    temporal_trend: float = 0.0
    trajectory: str = ""

    def to_dict(self, include_map_base64=False):
        data = {
            "source_name": self.source_name,
            "plot_id": self.plot_id,
            "predicted_crop": self.predicted_crop,
            "ground_truth_crop": self.ground_truth_crop,
            "health_status": self.health_status,
            "trajectory": self.trajectory,
            "advice": self.advice,
            "mean_confidence": round(float(self.mean_confidence), 4),
            "mean_ndvi": round(float(self.mean_ndvi), 4),
            "mean_ndre": round(float(self.ndre_mean), 4),
            "mean_savi": round(float(self.savi_mean), 4),
            "temporal_trend": round(float(self.temporal_trend), 4),
            "classification_map_path": self.map_png_path,
            "classification_geotiff_path": self.geotiff_path,
            "class_distribution": self.class_distribution,
        }
        if include_map_base64:
            data["classification_map_base64"] = image_to_base64(self.map_png_path)
        return data


class InferenceEngine:
    def __init__(self, config=None):
        self.cfg = CONFIG.copy()
        if config:
            self.cfg.update(config)
        os.makedirs(self.cfg["output_dir"], exist_ok=True)

        self.meta = pd.read_csv(self.cfg["labels_csv"])
        self.meta["label_str"] = self.meta["species"] + "_" + self.meta["variety"]
        self.class_names = sorted(self.meta["label_str"].unique().tolist())

        model_path = self.cfg["model_path"]
        if not os.path.exists(model_path):
            found = auto_find_model()
            if not found:
                raise FileNotFoundError(f"Checkpoint not found at {self.cfg['model_path']}")
            model_path = found

        self.model = CropCNNLSTM(
            num_classes=len(self.class_names),
            in_ch=13,
            cnn_out=self.cfg["cnn_out"],
            lstm_hidden=self.cfg["lstm_hidden"],
            dropout=self.cfg["dropout"],
        ).to(DEVICE)

        state = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model_path = model_path

    def _run_inference(
        self,
        plot_data,
        transform,
        crs,
        source_name,
        ground_truth_crop="",
        plot_id=None,
        smooth=False,
        sigma=3.0,
        with_confidence=False,
        save_geotiff=True,
    ):
        _, raw_pixel_prob_cube, _, raw_pixel_conf_map = sliding_window_pixel_inference(
            self.model,
            plot_data,
            len(self.class_names),
            self.cfg,
        )

        raw_hard_class_map = hard_class_map_from_probs(raw_pixel_prob_cube)

        display_prob_cube = make_display_probability_cube(
            raw_pixel_prob_cube=raw_pixel_prob_cube,
            raw_hard_class_map=raw_hard_class_map,
            n_classes=len(self.class_names),
            sigma=sigma if smooth else 0.0,
        )

        conf_map_for_display = display_confidence_map(
            raw_pixel_conf_map,
            smooth=smooth,
            sigma=sigma,
        )

        vals, counts = np.unique(raw_hard_class_map, return_counts=True)
        order = np.argsort(counts)[::-1]
        vals, counts = vals[order], counts[order]
        dominant_idx = int(vals[0])
        predicted_crop = self.class_names[dominant_idx]

        class_distribution = {
            self.class_names[int(v)]: round(float(c / counts.sum()), 4)
            for v, c in zip(vals[:10], counts[:10])
        }

        ndvi_series = plot_data[:, -3, :, :].mean(axis=(1, 2))
        ndre_series = plot_data[:, -2, :, :].mean(axis=(1, 2))
        savi_series = plot_data[:, -1, :, :].mean(axis=(1, 2))

        mean_ndvi = float(ndvi_series.mean())
        mean_ndre = float(ndre_series.mean())
        mean_savi = float(savi_series.mean())
        temporal_trend = float(ndvi_series[-1] - ndvi_series[0])

        crop_species = canonical_crop_species(predicted_crop)
        health = assess_health(
            ndvi_mean=mean_ndvi,
            ndre_mean=mean_ndre,
            savi_mean=mean_savi,
            crop_species=crop_species,
            temporal_trend=temporal_trend,
        )

        subtitle = f"Ground truth: {shorten(ground_truth_crop, 26)}" if ground_truth_crop else f"Source: {shorten(source_name, 26)}"
        map_png_path = render_classification_map(
            raw_hard_class_map=raw_hard_class_map,
            display_prob_cube=display_prob_cube,
            conf_map_for_display=conf_map_for_display,
            raw_pixel_conf_map=raw_pixel_conf_map,
            source_name=source_name,
            class_names=self.class_names,
            out_dir=self.cfg["output_dir"],
            with_confidence=with_confidence,
            sigma=(sigma if smooth else 0.0),
            smooth_render=smooth,
            subtitle=subtitle,
        )

        geotiff_path = ""
        if save_geotiff and transform is not None and crs is not None:
            safe_name = sanitize_name(source_name)
            geotiff_path = os.path.join(self.cfg["output_dir"], f"classification_{safe_name}.tif")
            write_class_geotiff(raw_hard_class_map, transform, crs, geotiff_path)

        return InferenceArtifacts(
            source_name=source_name,
            plot_id=plot_id,
            predicted_crop=predicted_crop,
            ground_truth_crop=ground_truth_crop,
            health_status=health["health_status"],
            advice=health["advice"],
            mean_confidence=float(raw_pixel_conf_map.mean()),
            mean_ndvi=mean_ndvi,
            ndre_mean=mean_ndre,
            savi_mean=mean_savi,
            temporal_trend=temporal_trend,
            trajectory=health["trajectory"],
            map_png_path=map_png_path,
            geotiff_path=geotiff_path,
            class_distribution=class_distribution,
        )

    def infer_plot(self, plot_id: int, smooth=False, sigma=3.0, with_confidence=False, save_geotiff=True):
        row = self.meta[self.meta["plot_id"] == plot_id]
        if row.empty:
            raise ValueError(f"Plot {plot_id} not found in metadata")

        species = str(row["species"].values[0])
        variety = str(row["variety"].values[0])
        ground_truth_crop = f"{species}_{variety}"
        plot_data, transform, crs = load_plot_rasters(plot_id, self.cfg)

        return self._run_inference(
            plot_data=plot_data,
            transform=transform,
            crs=crs,
            source_name=f"plot_{plot_id}",
            ground_truth_crop=ground_truth_crop,
            plot_id=plot_id,
            smooth=smooth,
            sigma=sigma,
            with_confidence=with_confidence,
            save_geotiff=save_geotiff,
        )

    def infer_tif_sequence(
        self,
        tif_paths: Sequence[str],
        source_name="uploaded_sequence",
        smooth=False,
        sigma=3.0,
        with_confidence=False,
        save_geotiff=True,
    ):
        plot_data, transform, crs = load_tif_sequence(tif_paths, self.cfg)

        return self._run_inference(
            plot_data=plot_data,
            transform=transform,
            crs=crs,
            source_name=source_name,
            ground_truth_crop="",
            plot_id=None,
            smooth=smooth,
            sigma=sigma,
            with_confidence=with_confidence,
            save_geotiff=save_geotiff,
        )
