#!/usr/bin/env python3
"""
CLI wrapper for MuST-C inference.
Fresh build timestamp: 2026-04-17 16:45:56 UTC
"""

import argparse
import json
from inference_core import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="MuST-C CLI inference")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--plot", type=int, help="Local test mode using MuST-C plot_id from the training dataset")
    src.add_argument("--tifs", nargs='+', help="Explicit GeoTIFF sequence for general inference; provide exactly 12 files")
    parser.add_argument("--source-name", default="cli_sequence", help="Label for the input sequence when using --tifs")
    parser.add_argument("--smooth", action="store_true", help="Enable non-destructive display-only smoothing for the PNG")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma when --smooth is set")
    parser.add_argument("--with-confidence", action="store_true", help="Include confidence panel in the PNG")
    parser.add_argument("--no-geotiff", action="store_true", help="Skip GeoTIFF export")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable summary")
    args = parser.parse_args()

    engine = InferenceEngine()
    if args.plot is not None:
        result = engine.infer_plot(plot_id=args.plot, smooth=args.smooth, sigma=args.sigma, with_confidence=args.with_confidence, save_geotiff=(not args.no_geotiff))
    else:
        result = engine.infer_tif_sequence(tif_paths=args.tifs, source_name=args.source_name, smooth=args.smooth, sigma=args.sigma, with_confidence=args.with_confidence, save_geotiff=(not args.no_geotiff))

    if args.json:
        print(json.dumps(result.to_dict(include_map_base64=False), indent=2))
        return

    print(f"Source name       : {result.source_name}")
    print(f"Plot ID           : {result.plot_id if result.plot_id is not None else '(external GeoTIFF sequence)'}")
    print(f"Predicted crop    : {result.predicted_crop}")
    print(f"Ground truth crop : {result.ground_truth_crop or '(unknown / external input)'}")
    print(f"Health status     : {result.health_status}")
    print(f"Advice            : {result.advice}")
    print(f"Mean confidence   : {result.mean_confidence:.4f}")
    print(f"Mean NDVI         : {result.mean_ndvi:.4f}")
    print(f"Mean SAVI         : {result.savi_mean:.4f}")
    print(f"Mean NDRE         : {result.ndre_mean:.4f}")
    print(f"Temporal trend         : {result.temporal_trend:.4f}")
    print(f"Map PNG           : {result.map_png_path}")
    print(f"GeoTIFF           : {result.geotiff_path or '(skipped)'}")


if __name__ == "__main__":
    main()
