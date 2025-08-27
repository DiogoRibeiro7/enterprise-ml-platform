from __future__ import annotations

"""Model explainability utilities."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import structlog

try:  # pragma: no cover - optional
    import shap
except Exception:  # pragma: no cover
    shap = None  # type: ignore

try:  # pragma: no cover - optional
    from lime.lime_tabular import LimeTabularExplainer
except Exception:  # pragma: no cover
    LimeTabularExplainer = None  # type: ignore

logger = structlog.get_logger()


@dataclass
class ModelExplainer:
    """Compute model explanations using SHAP or LIME."""

    def explain(
        self, model: Any, features: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config = config or {}
        method = config.get("method", "shap")
        if method == "shap":
            if shap is None:  # pragma: no cover - runtime check
                raise ImportError("shap is required for SHAP explanations")
            explainer = shap.Explainer(model, features)
            values = explainer(features)
            return {"shap_values": values.values}
        if method == "lime":
            if LimeTabularExplainer is None:  # pragma: no cover
                raise ImportError("lime is required for LIME explanations")
            explainer = LimeTabularExplainer(features)
            explanation = explainer.explain_instance(
                features[0], lambda x: model.predict(x)
            )
            return {"lime": explanation.as_list()}
        raise ValueError(f"Unknown explanation method: {method}")
