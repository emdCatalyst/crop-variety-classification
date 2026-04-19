#!/usr/bin/env python3
"""
HTTP inference server.
Fresh build timestamp: 2026-04-17 16:45:56 UTC
"""

import os
import shutil
import tempfile
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from inference_core import InferenceEngine

app = Flask(__name__)
engine = InferenceEngine()


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/infer")
def infer():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No GeoTIFF files uploaded. Send 12 files in the 'files' field."}), 400

    source_name = request.form.get("source_name", "uploaded_sequence")
    smooth = parse_bool(request.form.get("smooth"), False)
    sigma = float(request.form.get("sigma", 3.0))
    with_confidence = parse_bool(request.form.get("with_confidence"), False)
    include_map_base64 = parse_bool(request.form.get("include_map_base64"), True)
    save_geotiff = parse_bool(request.form.get("export_geotiff"), True)

    temp_dir = tempfile.mkdtemp(prefix="agri_infer_")
    saved_paths = []
    try:
        for f in files:
            name = secure_filename(f.filename or "input.tif")
            path = os.path.join(temp_dir, name)
            f.save(path)
            saved_paths.append(path)
        saved_paths = sorted(saved_paths)

        result = engine.infer_tif_sequence(tif_paths=saved_paths, source_name=source_name, smooth=smooth, sigma=sigma, with_confidence=with_confidence, save_geotiff=save_geotiff)
        return jsonify(result.to_dict(include_map_base64=include_map_base64))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
