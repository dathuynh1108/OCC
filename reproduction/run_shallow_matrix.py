"""Selected shallow source protocol with four isolated CPU fitting processes."""

import concurrent.futures
import itertools
import json
import subprocess
import sys
import time
from .common import ROOT, load_plan, plan_path, sha256, write_json

plan = load_plan()["native"]["shallow"]
fixture = json.loads(
    (ROOT / "artifacts/reproduction/fixtures/shallow_parity.json").read_text()
)
assert fixture["passed"]
out = ROOT / "results/native/shallow" / plan["target"]
write_json(
    out / "launch.json",
    {
        "started_unix": time.time(),
        "run_plan_sha256": sha256(plan_path()),
        "planned_rows": 400,
        "isolated_processes": 4,
        "BLAS_threads_per_process": 4,
    },
)


def run(key):
    seed, dataset, normal = key
    directory = out / dataset / f"class_{normal}" / f"seed_{seed}"
    if all((directory / f"nu_{nu}/result.json").exists() for nu in plan["nu"]):
        return None
    command = [
        sys.executable,
        "-m",
        "reproduction.shallow",
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
        return {
            "dataset": dataset,
            "seed": seed,
            "normal_class": normal,
            "returncode": result.returncode,
        }
    return None


failures = []
keys = itertools.product(plan["seeds"], plan["datasets"], plan["classes"])
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for result in executor.map(run, keys):
        if result:
            failures.append(result)
            write_json(out / "failures.json", failures)
write_json(
    out / "matrix_exit.json",
    {"failure_count": len(failures), "finished_unix": time.time()},
)
sys.exit(bool(failures))
