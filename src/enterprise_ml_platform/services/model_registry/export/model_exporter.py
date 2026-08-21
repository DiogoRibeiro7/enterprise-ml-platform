"""Model exporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelExporter:
    """Convert models into different serialisation formats.

    Not implemented. The previous version returned ``/tmp/exported_model.onnx``
    without converting anything, so callers could not tell a successful export
    from a no-op.
    """

    def export(self, model: Any, fmt: str) -> str:
        """Export ``model`` to format ``fmt`` and return the artifact path.

        Raises:
            NotImplementedError: Always. Use the format's own exporter
                (``skl2onnx``, ``mlflow.onnx``) until this is implemented.
        """
        raise NotImplementedError(
            f"exporting to {fmt!r} is not implemented; "
            "convert the model with the target format's own tooling"
        )
