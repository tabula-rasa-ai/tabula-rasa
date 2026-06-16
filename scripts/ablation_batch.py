"""Ablation batch runner — queues experiments sequentially on CPU.
Patches config.py per run, launches train_specialist.py, logs to comparator.
"""
import sys, os, time, json, subprocess, shutil, re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE / 'config.py'
EXPERIMENTS_JSON = BASE / 'experiments.json'

def patch_config(**overrides):
    with open(CONFIG_PATH) as f:
        text = f.read()
    for key, val in overrides.items():
        if isinstance(val, str):
            new_line = f"    {key} = '{val}'"
        elif isinstance(val, bool):
            new_line = f"    {key} = {str(val)}"
        else:
            new_line = f"    {key} = {val}"
        pattern = rf"    {key} = .*"
        if re.search(pattern, text):
            text = re.sub(pattern, new_line, text)
        else:
            text = text.replace(
                "class Config:",
                f"class Config:\n{new_line}",
            )
        print(f"  [cfg] {key} = {val}")
    with open(CONFIG_PATH, 'w') as f:
        f.write(text)

def backup_config():
    shutil.copy(CONFIG_PATH, CONFIG_PATH.with_suffix('.py.bak'))

def restore_config():
    bak = CONFIG_PATH.with_suffix('.py.bak')
    if bak.exists():
        shutil.copy(bak, CONFIG_PATH)
        bak.unlink()

EXPERIMENTS = [
    {
        'name': 'ReLU + LR 1e-3',
        'op': 'add',
        'steps': 15000,
        'config': {'activation': 'relu', 'learning_rate': 1e-3},
        'note': 'ReLU with higher LR — checking if exp #6 was LR-starved',
    },
    {
        'name': 'GELU L4',
        'op': 'add',
        'steps': 15000,
        'config': {'activation': 'gelu', 'learning_rate': 5e-4},
        'note': 'GELU vs SwiGLU (51%) vs ReLU (76%) — best modern alt',
    },
    {
        'name': 'ReLU sub',
        'op': 'sub',
        'steps': 15000,
        'config': {'activation': 'relu', 'learning_rate': 5e-4},
        'note': 'ReLU on subtraction — does advantage transfer?',
    },
    {
        'name': 'ReLU L2',
        'op': 'add',
        'steps': 15000,
        'config': {'activation': 'relu', 'n_layers': 2, 'learning_rate': 5e-4},
        'note': 'ReLU at 2 layers — beats SwiGLU L2 (32%)?',
    },
]

log_file = BASE / 'ablation_batch.log'
backup_config()

results = []
for i, exp in enumerate(EXPERIMENTS):
    header = f"\n{'='*60}\n  [{i+1}/{len(EXPERIMENTS)}] {exp['name']} ({exp['op']}, {exp['steps']} steps)\n{'='*60}\n"
    print(header, flush=True)
    with open(log_file, 'a') as log:
        log.write(header)

    try:
        patch_config(**exp['config'])
    except Exception as e:
        err = f"  [ERR] Config patch failed: {e}"
        print(err, flush=True)
        with open(log_file, 'a') as log:
            log.write(err + '\n')
        results.append({'name': exp['name'], 'status': 'FAILED', 'error': str(e)})
        continue

    cmd = [
        sys.executable or 'python3',
        str(BASE / 'train_specialist.py'),
        exp['op'],
        '--steps', str(exp['steps']),
    ]
    print(f"  Cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(BASE))
    stdout, _ = proc.communicate()
    elapsed = time.time() - t0
    output = stdout.decode('utf-8', errors='replace')

    with open(log_file, 'a') as log:
        log.write(f"  Duration: {elapsed:.0f}s ({elapsed/60:.1f}m)\n")
        log.write(f"  Exit code: {proc.returncode}\n")
        log.write(output + '\n')

    if proc.returncode != 0:
        err = f"  [ERR] Training failed with code {proc.returncode}"
        print(err, flush=True)
        results.append({'name': exp['name'], 'status': 'FAILED', 'error': err})
        continue

    matches = re.findall(r'Done.*Best: ([0-9.]+)%', output)
    acc = float(matches[-1]) if matches else None

    if acc is not None:
        try:
            with open(EXPERIMENTS_JSON) as f:
                exps = json.load(f)
            if exps:
                exps[-1]['notes'] = exp['note']
                with open(EXPERIMENTS_JSON, 'w') as f:
                    json.dump(exps, f, indent=2)
        except Exception:
            pass

    result = {'name': exp['name'], 'status': 'OK', 'accuracy': acc, 'time_s': elapsed}
    results.append(result)
    print(f"  ✓ {exp['name']} — Best: {acc}% ({elapsed:.0f}s)", flush=True)
    time.sleep(2)

restore_config()

summary = f"\n{'='*60}\n  BATCH COMPLETE — {time.ctime()}\n{'='*60}\n"
for r in results:
    acc_str = f"{r.get('accuracy','?')}%" if r['status'] == 'OK' else r.get('error','FAILED')
    summary += f"  {r['name']}: {acc_str}\n"

print(summary, flush=True)
with open(log_file, 'a') as log:
    log.write(summary + '\n')
print(f"\n  Full log: {log_file}", flush=True)
