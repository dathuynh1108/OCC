"""All locked CIFAR DROCC classes/seeds; both epoch-selection exports per trajectory."""

import json
import subprocess
import sys
import time

from .common import ROOT, load_plan, plan_path, sha256, write_json

config = load_plan()["native"]["drocc"]
parity = json.loads(
    (ROOT / "artifacts/reproduction/fixtures/drocc_parity.json").read_text()
)
assert len(parity) == 2 and all(r["passed"] for r in parity)
smoke = json.loads(
    (
        ROOT
        / "artifacts/reproduction/smoke/drocc/class_0/seed_0/fixed_final_epoch_result.json"
    ).read_text()
)
assert smoke["smoke"] and smoke["complete"] and smoke["test_count"] == 10000
out = ROOT / "results/native/drocc" / config["target"]
write_json(
    out / "launch.json",
    {
        "run_plan_sha256": sha256(plan_path()),
        "started_unix": time.time(),
        "trajectories": 30,
        "selection_exports": 60,
    },
)
failures = []
for normal in config["classes"]:
    for seed in config["seeds"]:
        result_path = out / f"class_{normal}/seed_{seed}/fixed_final_epoch_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (
                result["complete"]
                and not result["smoke"]
                and result["trained_epochs"] == config["epochs"]
            ):
                continue
        command = [
            sys.executable,
            "-m",
            "reproduction.drocc",
            "--normal-class",
            str(normal),
            "--seed",
            str(seed),
        ]
        print("RUN", " ".join(command), flush=True)
        run = subprocess.run(command, cwd=ROOT)
        if run.returncode:
            failures.append(
                {
                    "normal_class": normal,
                    "seed": seed,
                    "returncode": run.returncode,
                    "command": command,
                }
            )
            write_json(out / "failures.json", failures)
write_json(
    out / "matrix_exit.json", {"failures": failures, "finished_unix": time.time()}
)
sys.exit(bool(failures))
