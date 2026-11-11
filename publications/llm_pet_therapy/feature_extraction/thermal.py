"""Thermal ROI geometry and automatic frame QC; side labels follow the image pipeline."""
from pathlib import Path
import math
import numpy as np

def stage_mean(frame_temperatures,valid_flags,roi_key):
    if any(type(x) not in (bool,np.bool_) for x in valid_flags):raise ValueError('Validity flags must be explicit booleans, not text')
    values=np.asarray(frame_temperatures,float);valid=np.asarray(valid_flags,bool)
    if len(values)!=30 or len(valid)!=30:raise ValueError('Exactly 30 selected frames required')
    minimum={'nasal_tip':23,'left_cheek':22,'right_cheek':22}[roi_key]
    if not np.isfinite(values[valid]).all():raise ValueError('A frame marked valid has no finite ROI temperature')
    count=int(valid.sum())
    return dict(value=float(np.mean(values[valid])) if count>=minimum else None,valid_n=count,minimum_valid_n=minimum,usable=count>=minimum)

def build_tfan_model(model_path):
    """CPU inference with caller-provided TFAN and detector weights; no download."""
    import sys,types,importlib.machinery
    model_path=Path(model_path)
    detector=Path.home()/'.neurovc/models/tfw/YOLOv5n-Face.modern.pt'
    if not model_path.is_file() or not detector.is_file():raise FileNotFoundError('Provide the TFAN landmark weights and the detector at its documented cache path')
    stub=types.ModuleType('gdown');stub.__spec__=importlib.machinery.ModuleSpec('gdown',loader=None)
    def refuse(*args,**kwargs):raise RuntimeError('Offline TFAN execution: weights must be provided locally')
    stub.download=refuse;sys.modules['gdown']=stub
    import torch,timm
    torch.set_num_threads(1)
    original=timm.create_model
    def offline(*args,**kwargs):
        kwargs['pretrained']=False
        return original(*args,**kwargs)
    timm.create_model=offline
    try:
        from tfan._version import __version__
        if __version__!='1.0.0':raise RuntimeError('This pipeline requires TFAN 1.0.0 with the TFW detector')
        from tfan import ThermalLandmarks
        return ThermalLandmarks(model_path=model_path,device='cpu',n_landmarks=478)
    finally:timm.create_model=original
ROI_ORDER=('nasal_tip', 'left_cheek', 'right_cheek')
ROI_CONFIG={'nasal_tip': {'name_en': 'nasal_tip', 'anchor': 1, 'uncertainty_indices': (1, 2, 4, 5, 6, 19, 94, 97, 98, 326, 327), 'radius_x_face_width': 0.035, 'radius_y_face_width': 0.03, 'color': (0, 255, 255)}, 'left_cheek': {'name_en': 'left_cheek', 'anchor': 425, 'secondary_anchor': 291, 'anchor_weight': 0.6, 'uncertainty_indices': (280, 330, 347, 348, 349, 352, 411, 425), 'radius_x_face_width': 0.055, 'radius_y_face_width': 0.045, 'color': (255, 64, 255)}, 'right_cheek': {'name_en': 'right_cheek', 'anchor': 205, 'secondary_anchor': 61, 'anchor_weight': 0.6, 'uncertainty_indices': (50, 101, 118, 119, 120, 123, 187, 205), 'radius_x_face_width': 0.055, 'radius_y_face_width': 0.045, 'color': (64, 192, 255)}}
ROBUST_Z_THRESHOLD=5.0
MIN_ABRUPT_TEMPERATURE_C=0.75
MIN_SPATIAL_IQR_C=2.0
MIN_ROI_PIXELS=20

def robust_outlier_mask(values: np.ndarray, high_only: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.zeros(values.shape, dtype=bool)
    if finite.sum() < 6:
        return result
    center = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return result
    z = (values - center) / scale
    if high_only:
        result[finite] = z[finite] > ROBUST_Z_THRESHOLD
    else:
        result[finite] = np.abs(z[finite]) > ROBUST_Z_THRESHOLD
    return result

def face_coordinate_system(points: np.ndarray) -> dict:
    left_edge = points[234]
    right_edge = points[454]
    horizontal = right_edge - left_edge
    face_width = float(np.linalg.norm(horizontal))
    if not np.isfinite(face_width) or face_width <= 0:
        raise ValueError("Invalid face width")
    x_axis = horizontal / face_width
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=float)
    if float(np.dot(y_axis, points[152] - points[10])) < 0:
        y_axis *= -1.0
    face_height = float(np.linalg.norm(points[152] - points[10]))
    face_center = (left_edge + right_edge) / 2.0
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "face_width": face_width,
        "face_height": face_height,
        "face_center_x": float(face_center[0]),
        "face_center_y": float(face_center[1]),
        "face_angle_radians": float(math.atan2(x_axis[1], x_axis[0])),
    }

def roi_geometry(points: np.ndarray, face: dict, roi_key: str) -> dict:
    config = ROI_CONFIG[roi_key]
    if "secondary_anchor" in config:
        weight = config["anchor_weight"]
        center = (
            weight * points[config["anchor"]]
            + (1.0 - weight) * points[config["secondary_anchor"]]
        )
    else:
        center = points[config["anchor"]]
    return {
        "center": np.asarray(center, dtype=float),
        "x_axis": face["x_axis"],
        "y_axis": face["y_axis"],
        "radius_x": config["radius_x_face_width"] * face["face_width"],
        "radius_y": config["radius_y_face_width"] * face["face_width"],
    }

def sample_ellipse(matrix: np.ndarray, geometry: dict) -> dict:
    center = geometry["center"]
    radius = max(geometry["radius_x"], geometry["radius_y"])
    x0 = max(0, int(math.floor(center[0] - radius - 2)))
    x1 = min(matrix.shape[1], int(math.ceil(center[0] + radius + 2)) + 1)
    y0 = max(0, int(math.floor(center[1] - radius - 2)))
    y1 = min(matrix.shape[0], int(math.ceil(center[1] + radius + 2)) + 1)
    if x0 >= x1 or y0 >= y1:
        return {"pixel_count": 0, "temperature_c": np.nan, "spatial_iqr_c": np.nan}
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - center[0]
    dy = yy - center[1]
    u = dx * geometry["x_axis"][0] + dy * geometry["x_axis"][1]
    v = dx * geometry["y_axis"][0] + dy * geometry["y_axis"][1]
    mask = (u / geometry["radius_x"]) ** 2 + (v / geometry["radius_y"]) ** 2 <= 1.0
    values = matrix[y0:y1, x0:x1][mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"pixel_count": 0, "temperature_c": np.nan, "spatial_iqr_c": np.nan}
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {
        "pixel_count": int(values.size),
        "temperature_c": float(np.median(values)),
        "spatial_iqr_c": float(q3 - q1),
    }

def infer_frame(
    model,
    matrix: np.ndarray,
    saved_landmarks: np.ndarray | None = None,
    saved_scores: np.ndarray | None = None,
) -> dict:
    if saved_landmarks is None or saved_scores is None:
        landmarks, scores = model.process(matrix, mode="temperature")
    else:
        landmarks, scores = [saved_landmarks], [saved_scores]
    if not isinstance(landmarks, list) or len(landmarks) != 1:
        return {
            "global_valid": False,
            "global_reasons": ["face_detection_failed"],
            "landmarks": np.full((478, 2), np.nan, dtype=np.float32),
            "scores": np.full(478, np.nan, dtype=np.float32),
            "rois": {},
        }
    points = np.asarray(landmarks[0], dtype=np.float64)
    uncertainty = np.asarray(scores[0], dtype=np.float64)
    reasons: list[str] = []
    if points.shape != (478, 2) or uncertainty.shape != (478,):
        reasons.append("unexpected_landmark_shape")
    if not np.isfinite(points).all() or not np.isfinite(uncertainty).all():
        reasons.append("non_finite_landmarks")
    if reasons:
        return {
            "global_valid": False,
            "global_reasons": reasons,
            "landmarks": points.astype(np.float32, copy=False),
            "scores": uncertainty.astype(np.float32, copy=False),
            "rois": {},
        }
    in_bounds = (
        (points[:, 0] >= 0)
        & (points[:, 0] < matrix.shape[1])
        & (points[:, 1] >= 0)
        & (points[:, 1] < matrix.shape[0])
    )
    in_bounds_fraction = float(in_bounds.mean())
    try:
        face = face_coordinate_system(points)
    except ValueError:
        reasons.append("invalid_face_geometry")
        face = None
    if face is not None:
        if not 50 <= face["face_width"] <= 400:
            reasons.append("implausible_face_width")
        if not 50 <= face["face_height"] <= 450:
            reasons.append("implausible_face_height")
    rois = {}
    if face is not None:
        for roi_key in ROI_ORDER:
            geometry = roi_geometry(points, face, roi_key)
            sampled = sample_ellipse(matrix, geometry)
            indices = np.asarray(ROI_CONFIG[roi_key]["uncertainty_indices"], dtype=int)
            rois[roi_key] = {
                **sampled,
                "uncertainty": float(np.median(uncertainty[indices])),
                "geometry": geometry,
                "valid": True,
                "reasons": [],
                "review_flags": [],
            }
    return {
        "global_valid": not reasons,
        "global_reasons": reasons,
        "global_review_flags": (
            ["partial_landmarks_out_of_bounds"] if in_bounds_fraction < 0.98 else []
        ),
        "landmark_in_bounds_fraction": in_bounds_fraction,
        "global_uncertainty": float(np.median(uncertainty)),
        "landmarks": points.astype(np.float32),
        "scores": uncertainty.astype(np.float32),
        "face": face,
        "rois": rois,
    }

def append_reason(target: list[str], reason: str) -> None:
    if reason not in target:
        target.append(reason)

def apply_phase_qc(observations: list[dict]) -> None:
    initially_valid = np.array([obs["global_valid"] for obs in observations], dtype=bool)

    # Compare landmark shape after removing translation, scale, and in-plane
    # rotation.  Ordinary head movement is retained; only severe shape failure
    # can invalidate a frame automatically.
    normalized_shapes = []
    normalized_indices = []
    for index, obs in enumerate(observations):
        if not obs["global_valid"] or not obs.get("face"):
            continue
        points = obs["landmarks"].astype(float)
        face = obs["face"]
        center = np.array([face["face_center_x"], face["face_center_y"]])
        centered = points - center
        normalized = np.column_stack(
            (centered @ face["x_axis"], centered @ face["y_axis"])
        ) / face["face_width"]
        normalized_shapes.append(normalized)
        normalized_indices.append(index)
    if len(normalized_shapes) >= 6:
        shapes = np.stack(normalized_shapes)
        median_shape = np.median(shapes, axis=0)
        residuals = np.median(np.linalg.norm(shapes - median_shape, axis=2), axis=1)
        residual_outliers = robust_outlier_mask(residuals, high_only=True)
        for local_index, observation_index in enumerate(normalized_indices):
            residual = float(residuals[local_index])
            observations[observation_index]["normalized_shape_residual"] = residual
            if residual_outliers[local_index] and residual > 0.08:
                append_reason(
                    observations[observation_index]["global_review_flags"],
                    "normalized_landmark_shape_outlier",
                )
    global_uncertainty = np.array([obs.get("global_uncertainty", np.nan) for obs in observations])
    mask = robust_outlier_mask(global_uncertainty, high_only=True)
    for index in np.flatnonzero(mask & initially_valid):
        append_reason(observations[index]["global_review_flags"], "high_global_uncertainty")

    for obs in observations:
        for roi_key in ROI_ORDER:
            roi = obs.get("rois", {}).get(roi_key)
            if roi is None:
                continue
            if not obs["global_valid"]:
                roi["valid"] = False
                append_reason(roi["reasons"], "global_frame_invalid")
            if roi["pixel_count"] < MIN_ROI_PIXELS:
                roi["valid"] = False
                append_reason(roi["reasons"], "too_few_roi_pixels")
            if not np.isfinite(roi["temperature_c"]) or not 20.0 <= roi["temperature_c"] <= 40.0:
                roi["valid"] = False
                append_reason(roi["reasons"], "implausible_roi_temperature")

    for roi_key in ROI_ORDER:
        uncertainty = np.array(
            [obs.get("rois", {}).get(roi_key, {}).get("uncertainty", np.nan) for obs in observations]
        )
        spatial_iqr = np.array(
            [obs.get("rois", {}).get(roi_key, {}).get("spatial_iqr_c", np.nan) for obs in observations]
        )
        temperatures = np.array(
            [obs.get("rois", {}).get(roi_key, {}).get("temperature_c", np.nan) for obs in observations]
        )

        for index in np.flatnonzero(robust_outlier_mask(uncertainty, high_only=True)):
            roi = observations[index].get("rois", {}).get(roi_key)
            if roi:
                append_reason(roi["review_flags"], "high_roi_uncertainty")

        iqr_outlier = robust_outlier_mask(spatial_iqr, high_only=True) & (
            spatial_iqr > MIN_SPATIAL_IQR_C
        )
        for index in np.flatnonzero(iqr_outlier):
            roi = observations[index].get("rois", {}).get(roi_key)
            if roi:
                roi["valid"] = False
                append_reason(roi["reasons"], "heterogeneous_roi_possible_occlusion")

        residuals = np.full(len(observations), np.nan)
        for index, value in enumerate(temperatures):
            neighbor_indices = [
                other
                for other in range(max(0, index - 2), min(len(observations), index + 3))
                if other != index
                and np.isfinite(temperatures[other])
                and observations[other].get("rois", {}).get(roi_key, {}).get("valid", False)
            ]
            if len(neighbor_indices) >= 3 and np.isfinite(value):
                residuals[index] = abs(value - float(np.median(temperatures[neighbor_indices])))
        abrupt = robust_outlier_mask(residuals, high_only=True) & (
            residuals > MIN_ABRUPT_TEMPERATURE_C
        )
        for index in np.flatnonzero(abrupt):
            roi = observations[index].get("rois", {}).get(roi_key)
            if roi:
                roi["valid"] = False
                append_reason(roi["reasons"], "abrupt_temperature_possible_occlusion")

def read_temperature_matrix(path: Path) -> np.ndarray:
    """Read a TIStudio CSV as an exact 512x640 Celsius matrix."""

    raw = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=4,
        encoding="utf-8-sig",
    )
    matrix = raw[:, 1:]
    if matrix.shape[1] and np.isnan(matrix[:, -1]).all():
        matrix = matrix[:, :-1]
    if matrix.shape != (512, 640):
        raise ValueError(f"Unexpected temperature matrix shape {matrix.shape}: {path}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite temperature pixels: {path}")
    return matrix.astype(np.float32)
