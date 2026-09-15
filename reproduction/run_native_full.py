"""Run the immutable full Deep SVDD and DROCC native matrices in a new root."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, load_plan, plan_path, sha256, write_json


def parity_gate(root, method):
    filename = "deep_parity.json" if method == "deep" else "drocc_parity.json"
    rows = json.loads((root / filename).read_text())
    if method == "deep":
        assert len(rows) == 6 and all(row["upstream_same_runtime_parity"] for row in rows)
    else:
        assert len(rows) == 2 and all(row["passed"] for row in rows)
    return sha256(root / filename)


def native_data_gate():
    """Require a fresh, content-matched audit of the immutable native inputs."""
    audit_path = ROOT / "artifacts/reproduction/fixtures/native_data_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["all_bounds_match"]
    assert audit["fit_population"] == "original training split only; no test images used for this audit"
    for relative, metadata in audit["files"].items():
        path = ROOT / relative
        assert path.is_file() and path.stat().st_size == metadata["bytes"]
        assert sha256(path) == metadata["sha256"]
    return audit_path


def run(command, failures, metadata):
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        failures.append({**metadata, "returncode": result.returncode, "command": command})


def run_deep(root, plan, failures):
    output = root / "deep_svdd" / plan["target"]
    for dataset in plan["datasets"]:
        for normal_class in plan["classes"]:
            for seed in plan["seeds"]:
                expected = [output / dataset / f"class_{normal_class}" / f"seed_{seed}" / objective / "result.json"
                            for objective in plan["objectives"]]
                if all(path.exists() and json.loads(path.read_text())["complete"] for path in expected):
                    continue
                run([
                    sys.executable, "-m", "reproduction.deep_svdd",
                    "--dataset", dataset, "--normal-class", str(normal_class),
                    "--seed", str(seed), "--output-root", str(output),
                ], failures, {"method": "deep", "dataset": dataset, "normal_class": normal_class, "seed": seed})
    return output


def run_drocc(root, plan, failures):
    output = root / "drocc" / plan["target"]
    for normal_class in plan["classes"]:
        for seed in plan["seeds"]:
            fixed = output / f"class_{normal_class}" / f"seed_{seed}" / "fixed_final_epoch_result.json"
            if fixed.exists():
                saved = json.loads(fixed.read_text())
                if saved["complete"] and saved["trained_epochs"] == plan["epochs"]:
                    continue
            run([
                sys.executable, "-m", "reproduction.drocc",
                "--normal-class", str(normal_class), "--seed", str(seed),
                "--output-root", str(output),
            ], failures, {"method": "drocc", "normal_class": normal_class, "seed": seed})
    return output


def result_count(root, pattern):
    return len(list(root.glob(pattern)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("deep", "drocc", "all"), default="all")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/native-full-4090-20260915",
        help="New full-schedule result root; existing historical output is never used.",
    )
    parser.add_argument("--parity-root", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_plan()
    assert plan_path().resolve() == (ROOT / "run_plan.json").resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit_path = native_data_gate()
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    shutil.copy2(audit_path, evidence / "native_data_audit.json")
    failures = []
    launch = {
        "run_plan": "run_plan.json",
        "run_plan_sha256": sha256(ROOT / "run_plan.json"),
        "source_lock_sha256": sha256(ROOT / "source_lock.json"),
        "started_unix": time.time(),
        "output": str(root.relative_to(ROOT)),
        "methods": args.method,
        "native_data_audit_sha256": sha256(audit_path),
    }
    if args.method in ("deep", "all"):
        launch["deep_parity_sha256"] = parity_gate(args.parity_root, "deep")
    if args.method in ("drocc", "all"):
        launch["drocc_parity_sha256"] = parity_gate(args.parity_root, "drocc")
    manifest_stem = "matrix" if args.method == "all" else args.method
    write_json(root / f"{manifest_stem}_launch.json", launch)
    outputs = {}
    if args.method in ("deep", "all"):
        outputs["deep"] = str(run_deep(root, plan["native"]["deep_svdd"], failures).relative_to(root))
    if args.method in ("drocc", "all"):
        outputs["drocc"] = str(run_drocc(root, plan["native"]["drocc"], failures).relative_to(root))
    counts = {
        "deep_result_rows": result_count(root, "deep_svdd/**/result.json"),
        "drocc_fixed_final_rows": result_count(root, "drocc/**/fixed_final_epoch_result.json"),
        "drocc_test_selected_rows": result_count(root, "drocc/**/test_selected_result.json"),
    }
    # Deep and DROCC can be launched as separate processes because their result
    # trees and manifests are disjoint.  Validate only the method owned by this
    # invocation; rows produced by the other process are observed, never treated
    # as an error.  This keeps the immutable matrix root safe for concurrent GPU
    # scheduling without weakening either method's completion gate.
    expected = {}
    if args.method in ("deep", "all"):
        expected["deep_result_rows"] = 400
    if args.method in ("drocc", "all"):
        expected["drocc_fixed_final_rows"] = 30
        expected["drocc_test_selected_rows"] = 30
    complete = not failures and all(counts[key] == value for key, value in expected.items())
    write_json(root / f"{manifest_stem}_completion.json", {
        "complete": complete, "failures": failures, "counts": counts, "expected": expected,
        "outputs": outputs, "finished_unix": time.time(),
    })
    if failures:
        write_json(root / f"{manifest_stem}_failures.json", failures)
    if not complete:
        raise SystemExit(f"Native full matrix is incomplete; see {manifest_stem}_completion.json")


if __name__ == "__main__":
    main()
