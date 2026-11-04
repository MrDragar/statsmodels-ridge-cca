"""Run small numerical checks against R; optionally run the UCI candidate pilot.

Run after fetch_stage1: OPENBLAS_NUM_THREADS=1 python -m research.run_stage1
This is not nested CV, inference, a learning curve, or a final benchmark.
"""

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from statsmodels.multivariate.cancorr import CanCorr

from research.ridge import fit_ridge, paired_correlations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/stage1"
SEED = 20261104


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def duplicate_summary(z):
    _, inverse, counts = np.unique(z, axis=0, return_inverse=True, return_counts=True)
    groups = [np.flatnonzero(inverse == group).tolist() for group in np.flatnonzero(counts > 1)]
    return {"duplicate_rows_beyond_first": int(np.sum(counts - 1)),
            "groups_zero_based": groups}


def unique_paired_indices(x, y):
    """Keep the first original row per exact paired observation, in row order."""
    _, first = np.unique(np.column_stack([x, y]), axis=0, return_index=True)
    return np.sort(first)


def numerical_cases():
    rng = np.random.default_rng(SEED)
    cases = []
    for name, n, p, q, ax, ay in [
        ("classical", 72, 6, 4, 0., 0.),
        ("ridge_lowdim", 72, 6, 4, 0.3, 1.2),
        ("ridge_wide", 24, 40, 31, 0.3, 1.2),
    ]:
        x = rng.normal(size=(n, p))
        y = x[:, :2] @ rng.normal(size=(2, q)) + rng.normal(size=(n, q))
        start = perf_counter()
        fitted = fit_ridge(x, y, alpha_x=ax, alpha_y=ay, n_components=3)
        elapsed = perf_counter() - start
        case = {"name": name, "purpose": "assistant-selected numerical unit check; not proposal benchmark",
                "n": n, "p": p, "q": q, "alpha_x": ax, "alpha_y": ay,
                "lambda_x": fitted.lambda_x, "lambda_y": fitted.lambda_y,
                "penalized_roots": fitted.penalized_roots.tolist(),
                "train_correlations": fitted.train_correlations.tolist(),
                "fit_seconds": elapsed}
        if ax == ay == 0:
            case["statsmodels_max_root_error"] = float(np.max(abs(
                CanCorr(y, x).cancorr[:3] - fitted.penalized_roots)))
        fixture = ROOT / "statsmodels-checkout/statsmodels/multivariate/tests/results/ridge_cca" / name
        fixture.mkdir(parents=True, exist_ok=True)
        for filename, array in [("x.csv", x), ("y.csv", y)]:
            np.savetxt(fixture / filename, array, delimiter=",", fmt="%.17g")
        case["fixture_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(fixture.glob("*.csv"))}
        cases.append(case)
    return cases


def uci_audit_and_pilot():
    archive = ROOT / "data/raw/multiple_features.zip"
    source_manifest = json.loads((ROOT / "data/sources.json").read_text())
    expected = next(a["sha256"] for a in source_manifest["artifacts"]
                    if a["path"] == "data/raw/multiple_features.zip")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("UCI archive checksum mismatch")
    with ZipFile(archive) as z:
        xb, yb = z.read("mfeat-fac"), z.read("mfeat-pix")
        import io
        x, y = np.loadtxt(io.BytesIO(xb)), np.loadtxt(io.BytesIO(yb))
        (ROOT / "research/cache/mfeat.info").write_bytes(z.read("mfeat.info"))
        names = z.namelist()
    if x.shape != (2000, 216) or y.shape != (2000, 240):
        raise ValueError(f"Unexpected UCI dimensions: {x.shape}, {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("UCI contains missing or nonfinite values")
    labels = np.repeat(np.arange(10), 200)  # UCI's documented row order.
    audit = {"archive_sha256": expected, "archive_members": names,
             "x_shape": list(x.shape), "y_shape": list(y.shape),
             "x_file_sha256": hashlib.sha256(xb).hexdigest(),
             "y_file_sha256": hashlib.sha256(yb).hexdigest(),
             "nonfinite_values": 0,
             "constant_x_columns": np.flatnonzero(x.std(0) == 0).tolist(),
             "constant_y_columns": np.flatnonzero(y.std(0) == 0).tolist(),
             "x_duplicates": duplicate_summary(x), "y_duplicates": duplicate_summary(y),
             "paired_duplicates": duplicate_summary(np.column_stack([x, y])),
             "row_matching": "same row index per UCI documentation; no independent object IDs",
             "source_or_writer_ids_available": False,
             "independence": "not verified; exact-duplicate audit does not establish exchangeability"}
    unique = unique_paired_indices(x, y)
    audit["deduplication"] = {"policy": "keep first exact paired row before splitting",
                              "retained_rows": len(unique),
                              "dropped_original_indices": np.setdiff1d(np.arange(len(x)), unique).tolist()}
    split = StratifiedShuffleSplit(n_splits=1, test_size=400, random_state=SEED)
    pool_local, test_local = next(split.split(x[unique], labels[unique]))
    pool, test = unique[pool_local], unique[test_local]
    subsplit = StratifiedShuffleSplit(n_splits=1, train_size=100, random_state=SEED + 1)
    subtrain, _ = next(subsplit.split(x[pool], labels[pool]))
    train = pool[subtrain]
    # Train-only filtering; test never determines which columns are kept.
    keep_x, keep_y = x[train].std(0) > 0, y[train].std(0) > 0
    xt, yt = x[train][:, keep_x], y[train][:, keep_y]
    start = perf_counter()
    fitted = fit_ridge(xt, yt, n_components=3, scale=True)
    elapsed = perf_counter() - start
    u, v = fitted.transform(x[test][:, keep_x], y[test][:, keep_y])
    try:
        CanCorr(yt, xt)
        classical = "fit succeeded"
    except ValueError as exc:
        classical = f"ValueError: {exc}"
    pilot = {"seed": SEED, "subsample_seed": SEED + 1,
             "description": "single stratified split, fixed alpha=1; no tuning or inference",
             "n_train": len(train), "n_test": len(test),
             "remaining_pool_not_used": len(pool) - len(train),
             "deduplicated_before_splitting": True,
             "train_class_counts": np.bincount(labels[train], minlength=10).tolist(),
             "test_class_counts": np.bincount(labels[test], minlength=10).tolist(),
             "kept_x_columns": int(keep_x.sum()), "kept_y_columns": int(keep_y.sum()),
             "penalized_roots": fitted.penalized_roots.tolist(),
             "train_correlations": fitted.train_correlations.tolist(),
             "test_correlations": paired_correlations(u, v).tolist(),
             "fit_seconds": elapsed, "classical_statsmodels": classical}
    write_json(OUT / "uci_split.json", {"seed": SEED, "train": train.tolist(),
                                        "test": test.tolist(), "pool": pool.tolist()})
    return audit, pilot


def nutrimouse_checks():
    """Validate the proposal's high-dimensional real example against R rcc."""
    inputs = ROOT / "data/raw/nutrimouse"
    reference = OUT / "r_reference"
    x = np.loadtxt(inputs / "gene.csv", delimiter=",")
    y = np.loadtxt(inputs / "lipid.csv", delimiter=",")
    expected = np.loadtxt(reference / "nutrimouse.csv", delimiter=",", skiprows=1)
    fitted = fit_ridge(x, y, alpha_x=1, alpha_y=1, n_components=3, scale=True)
    np.testing.assert_allclose(fitted.penalized_roots, expected[:, 0], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(fitted.train_correlations, expected[:, 1], atol=1e-8, rtol=1e-6)
    cosines = {}
    for view, weights in [("x", fitted.x_weights), ("y", fitted.y_weights)]:
        r_weights = np.loadtxt(reference / f"nutrimouse_{view}_weights.csv", delimiter=",")
        qp, qr = np.linalg.qr(weights, mode="reduced")[0], np.linalg.qr(r_weights, mode="reduced")[0]
        values = np.linalg.svd(qp.T @ qr, compute_uv=False)
        np.testing.assert_allclose(values, np.ones(3), atol=1e-8, rtol=1e-6)
        cosines[view] = values.tolist()
    try:
        CanCorr(y, x)
        classical = "fit succeeded"
    except ValueError as exc:
        classical = f"ValueError: {exc}"
    return {"source": "CCA 1.2.2::nutrimouse exported by research/r_reference.R",
            "x_shape": list(x.shape), "y_shape": list(y.shape),
            "scale": "train sample standard deviation, ddof=1", "alpha": [1, 1],
            "description": "full-data numerical reference comparison; no held-out inference",
            "penalized_roots": fitted.penalized_roots.tolist(),
            "train_correlations": fitted.train_correlations.tolist(),
            "r_max_root_error": float(np.max(abs(fitted.penalized_roots - expected[:, 0]))),
            "r_max_correlation_error": float(np.max(abs(fitted.train_correlations - expected[:, 1]))),
            "subspace_principal_cosines": cosines,
            "classical_statsmodels": classical,
            "export_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(inputs.glob("*.csv"))}}


def main():
    parser = argparse.ArgumentParser(description="Numerical checks against R CCA::rcc; not the proposal benchmark")
    parser.add_argument("--uci-pilot", action="store_true", help="also run the optional UCI candidate pilot")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {p: metadata.version(p) for p in
                                ["numpy", "scipy", "pandas", "statsmodels",
                                 "scikit-learn", "pytest"]}}
    for obj in [CanCorr]:
        path = Path(inspect.getfile(obj))
        environment.setdefault("audited_installed_sources", []).append({
            "object": obj.__name__, "module": obj.__module__,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    # Invalidate stale run status before executing any reference calculations.
    environment["r_executed"] = False
    write_json(OUT / "environment.json", environment)
    cases = numerical_cases()
    write_json(OUT / "numerical_checks.json", cases)
    if args.uci_pilot:
        audit, pilot = uci_audit_and_pilot()
        write_json(OUT / "uci_audit.json", audit)
        write_json(OUT / "uci_pilot.json", pilot)
    subprocess.run(["Rscript", "research/r_reference.R"], cwd=ROOT, check=True)
    for case in cases:
        fixture = ROOT / "statsmodels-checkout/statsmodels/multivariate/tests/results/ridge_cca" / case["name"]
        r_results = np.loadtxt(fixture / "r_results.csv", delimiter=",", skiprows=1)
        np.testing.assert_allclose(case["penalized_roots"], r_results[:, 0], atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(case["train_correlations"], r_results[:, 1], atol=1e-8, rtol=1e-6)
        case["r_reference"] = {
            "max_root_error": float(np.max(abs(np.array(case["penalized_roots"]) - r_results[:, 0]))),
            "max_train_correlation_error": float(np.max(abs(
                np.array(case["train_correlations"]) - r_results[:, 1])))}
        case["fixture_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(fixture.glob("*.csv"))}
    write_json(OUT / "numerical_checks.json", cases)
    write_json(OUT / "nutrimouse_checks.json", nutrimouse_checks())
    environment["r_executed"] = True
    environment["r_version"] = subprocess.check_output(
        ["Rscript", "--version"], text=True).strip()
    write_json(OUT / "environment.json", environment)
    print(json.dumps({"numerical_unit_check_cases": len(cases), "r_reference": "passed",
                      "nutrimouse": "numerical fit comparison passed",
                      "uci_candidate_pilot_executed": args.uci_pilot}, indent=2))


if __name__ == "__main__":
    main()
