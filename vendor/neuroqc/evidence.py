"""Small, JSON-safe evidence records; these are not complete QC reports."""
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__


def provenance(inputs, parameters):
    versions = {"neuroqc": __version__}
    for name in ("numpy", "scipy", "nibabel", "matplotlib", "surfplot", "trimesh", "python-fcl"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"inputs": {key: str(Path(value).resolve()) for key, value in inputs.items()
                       if value is not None}, "parameters": parameters, "versions": versions}


def write_json(path, record):
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def metric_result(metrics, *, failed=False, metric_status=None, missing=()):
    """Local checks can prove failure; they cannot establish a combined rule pass."""
    return {"status": "fail" if failed else "unknown", "evidence_kind": "metrics",
            "metricStatus": metric_status or ("fail" if failed else "pass"),
            "metrics": metrics, "missing_checks": list(missing)}
