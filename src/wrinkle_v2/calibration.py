from __future__ import annotations

from typing import Dict, List, Optional


def _apply_piecewise(value: float, anchors: List[List[float]]) -> float:
    if not anchors:
        return value
    anchors = sorted(anchors, key=lambda x: x[0])
    if value <= anchors[0][0]:
        return anchors[0][1]
    if value >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= value <= x1:
            t = (value - x0) / max(1e-6, (x1 - x0))
            return y0 + t * (y1 - y0)
    return value


class ScoreCalibrator:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def apply_scores(self, scores: Dict, age: Optional[int] = None, skin_type: Optional[str] = None) -> Dict:
        if not self.cfg.get("enabled", False):
            return scores
        anchors = self._adjusted_anchors(age, skin_type)
        global_score = _apply_piecewise(float(scores["global_score"]), anchors)
        per_region_scores = {
            k: _apply_piecewise(float(v), anchors) for k, v in scores["per_region_scores"].items()
        }
        scores["global_score"] = float(global_score)
        scores["per_region_scores"] = per_region_scores
        return scores

    def _adjusted_anchors(self, age: Optional[int], skin_type: Optional[str]) -> List[List[float]]:
        anchors = [list(a) for a in self.cfg.get("anchors", [])]
        delta = 0.0
        for bracket in self.cfg.get("age_brackets", []):
            if age is None:
                break
            if bracket["min"] <= age <= bracket["max"]:
                delta += float(bracket.get("delta", 0.0))
                break
        if skin_type:
            delta += float(self.cfg.get("skin_type_delta", {}).get(skin_type, 0.0))
        if delta == 0.0:
            return anchors
        for a in anchors:
            a[1] = max(0.0, min(100.0, a[1] + delta))
        return anchors
