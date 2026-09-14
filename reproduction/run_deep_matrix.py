"""Execute all locked Deep SVDD rows, resuming persisted progress exactly."""

import json
import subprocess
import sys
import time

from .common import ROOT, load_plan, plan_path, sha256, write_json

plan = load_plan()["native"]["deep_svdd"]
parity_file = ROOT / "artifacts/reproduction/fixtures/deep_parity.json"
parity = json.loads(parity_file.read_text())
assert len(parity) == 6 and all(p["upstream_same_runtime_parity"] for p in parity)
root = ROOT / "results/native/deep_svdd" / plan["target"]
root.mkdir(parents=True, exist_ok=True)
write_json(
    root / "launch.json",
    {
        "run_plan_sha256": sha256(plan_path()),
        "parity_sha256": sha256(parity_file),
        "started_unix": time.time(),
        "full_matrix_objective_rows": 400,
    },
)
failures = []
for dataset in plan["datasets"]:
    for objective in plan["objectives"]:
        result = (
            ROOT
            / f"artifacts/reproduction/smoke/deep_svdd/{dataset}/class_0/seed_1/{objective}/result.json"
        )
        smoke = json.loads(result.read_text())
        assert smoke["complete"] and smoke["smoke"] and smoke["test_count"] == 10000
    for normal in plan["classes"]:
        for seed in plan["seeds"]:
            command = [
                sys.executable,
                "-m",
                "reproduction.deep_svdd",
                "--dataset",
                dataset,
                "--normal-class",
                str(normal),
                "--seed",
                str(seed),
            ]
            print("RUN", " ".join(command), flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                failures.append(
                    {
                        "dataset": dataset,
                        "normal_class": normal,
                        "seed": seed,
                        "returncode": result.returncode,
                        "command": command,
                    }
                )
                write_json(root / "failures.json", failures)
write_json(
    root / "matrix_exit.json",
    {"failure_count": len(failures), "finished_unix": time.time()},
)
sys.exit(bool(failures))
