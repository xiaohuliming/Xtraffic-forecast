#!/usr/bin/env python3
"""Serial zoo queue for the 5080 box (Windows). Reads zoo_tasks_win.txt
(lines: <model> <region> <seed> [batch_size] [attempt]), skips runs whose
summary.json already exists, retries a failed run once at half batch.
Progress goes to stdout; per-run logs to logs/zoo_<model>_<region>_s<seed>.log
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "scripts" / "remote" / "windows" / "zoo_tasks_win.txt"


def out_dir(model: str, seed: str) -> Path:
    d = (
        ROOT
        / "outputs"
        / "protocol_v2_noleak"
        / "baselines"
        / f"zoo_{model}"
    )
    return d if seed == "42" else Path(str(d) + f"_seed{seed}")


def main() -> None:
    os.chdir(ROOT)
    (ROOT / "logs").mkdir(exist_ok=True)
    queue = []
    for ln in TASKS.read_text().split("\n"):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            parts = ln.split()
            queue.append(parts + ["32"] * (4 - len(parts)) + ["0"] * (5 - max(len(parts), 4)))
    env = dict(os.environ, PYTHONUTF8="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    i = 0
    while i < len(queue):
        model, region, seed, bs, att = queue[i][:5]
        i += 1
        od = out_dir(model, seed)
        summ = od / region / "summary.json"
        if summ.exists():
            print(f"SKIP {model} {region} s{seed} (summary exists)", flush=True)
            continue
        log = ROOT / "logs" / f"zoo_{model}_{region}_s{seed}.log"
        print(f"START {model} {region} s{seed} bs{bs} att{att} {time.strftime('%F %T')}",
              flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as lf:
            rc = subprocess.call(
                [sys.executable, "-X", "utf8", "scripts/train_baseline_zoo.py",
                 "--model", model, "--region", region, "--seed", seed,
                 "--batch-size", bs, "--out-dir", str(od)],
                stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        mins = (time.time() - t0) / 60
        if rc == 0 and summ.exists():
            mae = json.loads(summ.read_text()).get("all")
            print(f"DONE {model} {region} s{seed} all={mae:.3f} ({mins:.0f}min)", flush=True)
        elif int(att) < 1:
            nbs = str(max(8, int(bs) // 2))
            queue.append([model, region, seed, nbs, str(int(att) + 1)])
            print(f"FAIL {model} {region} s{seed} rc={rc} ({mins:.0f}min) "
                  f"-> requeued bs{nbs}", flush=True)
        else:
            print(f"GAVE-UP {model} {region} s{seed} rc={rc} ({mins:.0f}min)", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
