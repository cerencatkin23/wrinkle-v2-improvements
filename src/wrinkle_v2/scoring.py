from typing import Dict, Tuple


def _apply_under_eye_invariant(vals: Dict, scoring_cfg: Dict, region: str) -> Dict:
    feat_cfg = scoring_cfg.get("features", {}).get("under_eye_invariant", {})
    if not feat_cfg.get("enabled", False):
        return vals
    if region not in {"under_eye_left", "under_eye_right"}:
        return vals

    area_ratio = float(vals.get("area_ratio", 0.0))
    length_density = float(vals.get("skeleton_length_density", 0.0))
    alpha = float(feat_cfg.get("alpha", 0.5))
    alpha = max(0.0, min(1.0, alpha))
    invariant = alpha * area_ratio + (1.0 - alpha) * length_density

    updated = dict(vals)
    updated["area_ratio"] = invariant
    updated["density"] = invariant
    if "length_density_scale" in feat_cfg:
        scale = float(feat_cfg.get("length_density_scale", 1200.0))
        updated["skeleton_length"] = length_density * scale
    return updated


def _apply_crows_feet_invariant(vals: Dict, scoring_cfg: Dict, region: str) -> Dict:
    feat_cfg = scoring_cfg.get("features", {}).get("crows_feet_invariant", {})
    if not feat_cfg.get("enabled", False):
        return vals
    if region not in {"crows_feet_left", "crows_feet_right"}:
        return vals

    area_ratio = float(vals.get("area_ratio", 0.0))
    length_density = float(vals.get("skeleton_length_density", 0.0))
    alpha = float(feat_cfg.get("alpha", 0.5))
    alpha = max(0.0, min(1.0, alpha))
    invariant = alpha * area_ratio + (1.0 - alpha) * length_density

    updated = dict(vals)
    updated["area_ratio"] = invariant
    updated["density"] = invariant
    if "length_density_scale" in feat_cfg:
        scale = float(feat_cfg.get("length_density_scale", 1200.0))
        updated["skeleton_length"] = length_density * scale
    return updated


def _normalize(value: float, min_max: Tuple[float, float]) -> float:
    vmin, vmax = min_max
    if vmax <= vmin:
        return 0.0
    norm = (value - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, norm))


def _normalize_stat(value: float, center: float, scale: float, max_abs_z: float | None = None) -> float:
    if scale <= 0:
        return 0.5
    z = (value - center) / scale
    if max_abs_z is not None:
        z = max(-max_abs_z, min(max_abs_z, z))
    return float(0.5 + 0.5 * (z / (1.0 + abs(z))))


def _region_normalized_scores(measurements: Dict, scoring_cfg: Dict) -> Dict:
    region_norm = scoring_cfg["region_normalization"]
    method = region_norm.get("method", "robust")
    priors = region_norm.get("priors", {})
    metric_weights = scoring_cfg.get("per_region_metric_weights", scoring_cfg.get("per_region_weights", {}))
    region_weights = scoring_cfg.get("region_weights", {})
    stab_cfg = scoring_cfg.get("stabilization", {})
    stab_enabled = bool(stab_cfg.get("enabled", False))
    max_abs_z = float(stab_cfg.get("max_abs_z", 3.0)) if stab_enabled else None

    per_region_scores = {}
    region_contrib = {}
    for region, vals in measurements["per_region"].items():
        vals = _apply_under_eye_invariant(vals, scoring_cfg, region)
        vals = _apply_crows_feet_invariant(vals, scoring_cfg, region)
        score = 0.0
        total_w = 0.0
        for key, weight in metric_weights.items():
            prior = priors.get(key, {})
            if method == "zscore":
                center = float(prior.get("mean", 0.0))
                scale = float(prior.get("std", 1.0))
            else:
                center = float(prior.get("median", 0.0))
                scale = float(prior.get("iqr", 1.0))
            norm = _normalize_stat(float(vals.get(key, 0.0)), center, scale, max_abs_z=max_abs_z)
            score += weight * norm
            total_w += weight
        score = score / max(1e-6, total_w)
        per_region_scores[region] = float(max(0.0, min(1.0, score)) * 100.0)
        region_contrib[region] = per_region_scores[region]

    if not region_weights:
        region_weights = {k: 1.0 for k in per_region_scores.keys()}

    if stab_enabled:
        region_caps = stab_cfg.get("region_score_caps", {})
        for region, cap in region_caps.items():
            if region in per_region_scores:
                per_region_scores[region] = float(min(per_region_scores[region], float(cap)))

    aggregation = stab_cfg.get("aggregation", "weighted_mean")
    if stab_enabled and aggregation in {"median", "trimmed_mean"}:
        vals = sorted(per_region_scores.values())
        if aggregation == "median":
            mid = len(vals) // 2
            global_score = float(vals[mid] if len(vals) % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2.0)
        else:
            trim = float(stab_cfg.get("trim_ratio", 0.1))
            k = int(len(vals) * trim)
            trimmed = vals[k:len(vals) - k] if len(vals) > 2 * k else vals
            global_score = float(sum(trimmed) / max(1, len(trimmed)))
    else:
        weighted_sum = 0.0
        weight_total = 0.0
        for region, score in per_region_scores.items():
            w = float(region_weights.get(region, 1.0))
            weighted_sum += w * score
            weight_total += w
        global_score = float(weighted_sum / max(1e-6, weight_total))

    if stab_enabled:
        max_pct = float(stab_cfg.get("max_region_pct", 0.4))
        contribs = {k: per_region_scores[k] * float(region_weights.get(k, 1.0)) for k in per_region_scores}
        total = sum(contribs.values())
        cap = max_pct * total if total > 0 else 0.0
        if cap > 0:
            for k in list(contribs.keys()):
                if contribs[k] > cap:
                    contribs[k] = cap
            total = sum(contribs.values())
            global_score = float(total / max(1e-6, sum(region_weights.values())))

    return {
        "global_score": global_score,
        "per_region_scores": per_region_scores,
    }


def compute_scores(measurements: Dict, scoring_cfg: Dict) -> Dict:
    if "region_normalization" in scoring_cfg:
        return _region_normalized_scores(measurements, scoring_cfg)

    norm_cfg = scoring_cfg["norm"]
    global_weights = scoring_cfg["global_weights"]
    per_region_weights = scoring_cfg["per_region_weights"]

    global_meas = measurements["global"]
    global_score = 0.0
    for key, weight in global_weights.items():
        global_score += weight * _normalize(global_meas[key], tuple(norm_cfg[key]))
    global_score = float(max(0.0, min(1.0, global_score)) * 100.0)

    per_region_scores = {}
    for region, vals in measurements["per_region"].items():
        vals = _apply_under_eye_invariant(vals, scoring_cfg, region)
        vals = _apply_crows_feet_invariant(vals, scoring_cfg, region)
        score = 0.0
        for key, weight in per_region_weights.items():
            score += weight * _normalize(vals[key], tuple(norm_cfg[key]))
        per_region_scores[region] = float(max(0.0, min(1.0, score)) * 100.0)

    return {
        "global_score": global_score,
        "per_region_scores": per_region_scores,
    }


def _bucket(value: float, thresholds: Tuple[float, float, float]) -> str:
    low, mid, high = thresholds
    if value < low:
        return "low"
    if value < mid:
        return "moderate"
    if value < high:
        return "high"
    return "very_high"


def build_reasoning(quality: Dict, measurements: Dict, scores: Dict, scoring_cfg: Dict) -> str:
    if not quality["quality_pass"]:
        reasons = ", ".join(quality["reasons"]) if quality["reasons"] else "unknown_quality_issue"
        return f"No score due to quality gate: {reasons}."

    quality_flags = quality.get("flags", {})
    flag_notes = []
    if quality_flags.get("quality_penalty", 0) > 0:
        flag_notes.append(f"soft penalty {quality_flags['quality_penalty']:.1f} applied")
    if quality_flags.get("blur_ok") is False:
        flag_notes.append("slightly blurry")
    if quality_flags.get("brightness_ok") is False:
        flag_notes.append("brightness outside ideal range")
    if quality_flags.get("face_detected") is False:
        flag_notes.append("face detection not confirmed")

    global_score = scores["global_score"]
    global_meas = measurements["global"]
    per_region = measurements["per_region"]

    thresholds = scoring_cfg["thresholds"]
    global_buckets = {
        "area_ratio": _bucket(global_meas["area_ratio"], tuple(thresholds["area_ratio"])),
        "skeleton_length": _bucket(global_meas["skeleton_length"], tuple(thresholds["skeleton_length"])),
        "thickness": _bucket(global_meas["thickness"], tuple(thresholds["thickness"])),
        "density": _bucket(global_meas["density"], tuple(thresholds["density"])),
    }

    region_lines = []
    for region, vals in per_region.items():
        region_score = scores["per_region_scores"].get(region, 0.0)
        region_line = (
            f"{region}: score {region_score:.1f} from coverage {vals['area_ratio']*100:.2f}%, "
            f"density {vals['density']:.4f}, length {vals['skeleton_length']:.1f}, thickness {vals['thickness']:.2f}."
        )
        region_lines.append(region_line)

    top_regions = sorted(scores["per_region_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
    top_text = ", ".join([f"{name} ({score:.1f})" for name, score in top_regions])

    stab_cfg = scoring_cfg.get("stabilization", {})
    stab_note = ""
    if stab_cfg.get("enabled", False):
        stab_note = (
            f" Stabilization applied (aggregation={stab_cfg.get('aggregation', 'weighted_mean')}, "
            f"cap={stab_cfg.get('max_region_pct', 0.4):.2f}, "
            f"clamp={stab_cfg.get('max_abs_z', 3.0):.1f})."
        )
    feat_cfg = scoring_cfg.get("features", {}).get("under_eye_invariant", {})
    if feat_cfg.get("enabled", False):
        stab_note += " Under-eye invariant metric applied."
    feat_cfg = scoring_cfg.get("features", {}).get("crows_feet_invariant", {})
    if feat_cfg.get("enabled", False):
        stab_note += " Crow's feet invariant metric applied."

    parts = [
        "Quality gate passed." + (f" Flags: {', '.join(flag_notes)}." if flag_notes else "") + stab_note,
        (
            f"Global score {global_score:.1f} with area_ratio {global_meas['area_ratio']:.4f} ({global_buckets['area_ratio']}), "
            f"skeleton_length {global_meas['skeleton_length']:.1f} ({global_buckets['skeleton_length']}), "
            f"thickness {global_meas['thickness']:.2f} ({global_buckets['thickness']}), "
            f"density {global_meas['density']:.4f} ({global_buckets['density']})."
        ),
        f"Top regions by score: {top_text}.",
        "Per-region details: " + " ".join(region_lines),
    ]
    return " ".join(parts)
