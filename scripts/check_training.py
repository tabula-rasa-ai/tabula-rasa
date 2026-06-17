"""Auto-check training health.
Usage: python3 scripts/check_training.py
       python3 scripts/check_training.py --watch   (poll every 60s)
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_PATH = 'specialists/math/add/training.log'
CURRENT_RUN_MAX_STEPS = 50000

# Expected milestones for scratchpad 1-digit addition
MILESTONES = [
    (2000,  1.2,   20,   'Warming up'),
    (5000,  0.8,   40,   'Learning phase'),
    (10000, 0.5,   60,   'Converging'),
    (20000, 0.25,  85,   'High accuracy'),
    (30000, 0.1,   95,   'Mastery'),
    (50000, 0.05,  98,   'Complete'),
]

def read_log():
    if not os.path.exists(LOG_PATH):
        return None, None, None, None, None
    with open(LOG_PATH) as f:
        text = f.read()

    # Parse only current run (max_steps=50000)
    steps, losses, accs = [], [], []
    best_acc = None
    latest_step, latest_loss, latest_speed = 0, None, None

    for line in text.split('\n'):
        m = re.search(r'Step\s+(\d+)/(\d+).*?loss=([\d.]+)', line)
        if m:
            total = int(m[2])
            if total != CURRENT_RUN_MAX_STEPS:
                continue
            step = int(m[1])
            loss = float(m[3])
            latest_step = step
            latest_loss = loss
            steps.append(step)
            losses.append(loss)
            sp = re.search(r'([\d.]+)\s+steps/s', line)
            if sp:
                latest_speed = float(sp.group(1))
        em = re.search(r'Eval:\s+([\d.]+)%\s+\(best:\s+([\d.]+)%', line)
        if em:
            acc = float(em[1])
            best = float(em[2])
            if best_acc is None or best > best_acc:
                best_acc = best
            accs.append((latest_step, acc))

    return latest_step, latest_loss, best_acc, latest_speed, (steps, losses, accs)

def check_health(step, loss, best_acc, speed):
    """Return (status, message, color) where status is 'good', 'warn', or 'bad'."""
    if step is None or step == 0:
        return 'wait', 'No training data yet. First log at step 500.', '\033[90m'

    # Find current milestone
    current_milestone = None
    next_milestone = None
    for i, (ms, ml, ma, name) in enumerate(MILESTONES):
        if step >= ms:
            current_milestone = (ms, ml, ma, name)
        if i + 1 < len(MILESTONES) and step < MILESTONES[i+1][0]:
            next_milestone = MILESTONES[i+1]
            break

    # Check loss vs expected
    if current_milestone:
        exp_loss = current_milestone[1]
        exp_acc = current_milestone[2]
        name = current_milestone[3]

        if loss is not None and loss > exp_loss * 2:
            return 'bad', f'Loss {loss:.4f} is much higher than expected {exp_loss:.2f} at step {step}. Check data format.', '\033[91m'
        elif loss is not None and loss > exp_loss * 1.3:
            return 'warn', f'Loss {loss:.4f} slightly above expected {exp_loss:.2f}. May need more capacity.', '\033[93m'
        elif best_acc is not None and best_acc < exp_acc * 0.5:
            return 'warn', f'Accuracy {best_acc:.0f}% below target {exp_acc}% at step {step}. Normal if early.', '\033[93m'
        else:
            eta = ''
            if speed and speed > 0 and next_milestone:
                remaining_steps = next_milestone[0] - step
                eta_secs = remaining_steps / speed
                eta = f' | Next milestone in {eta_secs/3600:.1f}h'
            return 'good', f'On track. Step {step} | Loss {loss:.4f} | Best {best_acc}%{eta}', '\033[92m'

    return 'info', f'Step {step} | Loss {loss} | Processing...', '\033[94m'

def check_once():
    step, loss, best_acc, speed, _ = read_log()
    status, msg, color = check_health(step, loss, best_acc, speed)

    # Color legend
    icons = {'good': '[OK]', 'warn': '[!]', 'bad': '[X]', 'wait': '[...]', 'info': '[i]'}

    print(f'{color}{icons[status]} {msg}\033[0m')

    # Show speed and progress bar
    if step and step > 0:
        pct = step / CURRENT_RUN_MAX_STEPS * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '#' * filled + '-' * (bar_len - filled)
        speed_str = f'{speed:.1f} st/s' if speed else '--'
        eta = ''
        if speed and speed > 0:
            remaining = (CURRENT_RUN_MAX_STEPS - step) / speed
            eta = f' | ETA: {remaining/3600:.1f}h'
        print(f'  [{bar}] {pct:.0f}% | {speed_str}{eta}')

    return status

def watch(interval=60):
    print(f'Training monitor started (checking every {interval}s). Ctrl+C to stop.')
    print(f'Looking for {CURRENT_RUN_MAX_STEPS}-step run in {LOG_PATH}')
    print()
    while True:
        try:
            status = check_once()
            if status == 'bad':
                print('  [!] Training may need attention.')
            print()
            time.sleep(interval)
        except KeyboardInterrupt:
            print('\nMonitor stopped.')
            break
        except Exception as e:
            print(f'  [ERR] {e}')
            time.sleep(interval)

def show_table():
    """Show the expected milestone table."""
    print('Expected milestones for 1-digit scratchpad:')
    print(f'  {"Step":>8} {"Loss":>8} {"Acc":>6}  Phase')
    print(f'  {"-"*8} {"-"*8} {"-"*6}  {"-"*15}')
    for ms, ml, ma, name in MILESTONES:
        print(f'  {ms:>8} {ml:>8.2f} {ma:>5}%  {name}')

if __name__ == '__main__':
    if '--watch' in sys.argv:
        watch()
    elif '--table' in sys.argv:
        show_table()
    else:
        check_once()
        print()
        show_table()
