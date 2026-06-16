"""Multi-computer specialist coordinator with worker heartbeats.

Each worker writes a heartbeat_<hostname>.json file every 5 minutes during
training. The coordinator monitors heartbeats and re-queues tasks from any
worker that goes silent for >15 minutes.

Usage:
    python3 specialist_network.py list               # Show all queued skills
    python3 specialist_network.py assign             # Assign unassigned skills
    python3 specialist_network.py sync D:/shared     # Import completed from network share

    # On worker computers:
    python3 specialist_network.py work math/subtraction  # Train with heartbeats

    # On coordinator (monitoring):
    python3 specialist_network.py watch D:/shared              # Poll for new zips
    python3 specialist_network.py watch D:/shared --heartbeats # + monitor heartbeats
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SPECIALISTS_DIR = Path("specialists")
QUEUED = ["math/sub", "math/mul", "math/div"]

# ─── Heartbeat Protocol ────────────────────────────────────────────
# Workers write heartbeat_<hostname>.json to a shared directory every
# HEARTBEAT_INTERVAL seconds. The coordinator checks heartbeats and
# marks tasks from stale workers as FAILED after STALE_TIMEOUT seconds.

HEARTBEAT_INTERVAL = 300  # 5 min
STALE_TIMEOUT = 900  # 15 min without heartbeat = dead worker


def get_hostname():
    """Get a stable worker identifier."""
    return socket.gethostname().lower().replace(" ", "_")


def write_heartbeat(task: str, status: str, share_path: str = None):
    """Write heartbeat file to signal worker is alive.

    Args:
        task: The skill being trained (e.g. 'math/subtraction').
        status: 'training', 'exporting', 'done', 'error'.
        share_path: Where to write the heartbeat. If None, writes to CWD.
    """
    hostname = get_hostname()
    data = {
        "worker": hostname,
        "task": task,
        "status": status,
        "last_seen": int(time.time()),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    dest = Path(share_path) if share_path else Path(".")
    dest.mkdir(parents=True, exist_ok=True)
    hb_file = dest / f"heartbeat_{hostname}.json"
    hb_file.write_text(json.dumps(data, indent=2))
    return hb_file


def check_heartbeats(share_path: str, stale_timeout: int = STALE_TIMEOUT):
    """Scan heartbeat files and return info about stale workers.

    Returns:
        List of (hostname, task, seconds_since_last_seen) for stale workers.
    """
    share = Path(share_path)
    if not share.exists():
        return []

    stale = []
    now = int(time.time())
    for hb_file in sorted(share.glob("heartbeat_*.json")):
        try:
            data = json.loads(hb_file.read_text())
            hostname = data.get("worker", hb_file.stem.replace("heartbeat_", ""))
            task = data.get("task", "unknown")
            status = data.get("status", "unknown")
            last_seen = data.get("last_seen", 0)
            age = now - last_seen
            if status not in ("done", "error") and age > stale_timeout:
                stale.append(
                    {
                        "hostname": hostname,
                        "task": task,
                        "status": status,
                        "age_seconds": age,
                        "file": str(hb_file),
                    }
                )
        except (json.JSONDecodeError, OSError) as e:
            print(f"[!] Bad heartbeat file {hb_file.name}: {e}")
    return stale


def heartbeat_daemon(task: str, share_path: str, interval: int = HEARTBEAT_INTERVAL):
    """Run heartbeat in a background thread. Call from work_on()."""
    import threading

    def _loop():
        while True:
            try:
                write_heartbeat(task, "training", share_path)
            except Exception as e:
                print(f"  [Heartbeat] Error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="heartbeat")
    t.start()
    return t


# ─── Network Share Setup ───────────────────────────────────────────


def find_network_shares():
    """Scan for accessible network locations."""
    candidates = []
    for drive in ["D:", "E:", "F:", "Z:"]:
        p = Path(f"{drive}/shared/")
        if p.exists():
            candidates.append(p)
    for host in ["localhost", "DESKTOP-", "LAPTOP-"]:
        p = Path(f"//{host}/shared/")
        if p.exists():
            candidates.append(p)
    return candidates


def ensure_share(share_path: str):
    """Ensure the shared folder exists."""
    p = Path(share_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─── Commands ──────────────────────────────────────────────────────


def list_skills():
    """Show status of all skills."""
    print(f'{"Skill":<25} {"Status":<12} {"Size":<10}')
    print("-" * 47)

    for skill in QUEUED:
        sp = SPECIALISTS_DIR / skill
        best = sp / "best.pt"
        final = sp / "final.pt"
        if best.exists():
            size = best.stat().st_size / 1e6
            status = "partial" if "add" in skill else "ready" if "general" in skill else "training"
            print(f"{skill:<25} {status:<12} {size:.1f} MB")
        else:
            print(f'{skill:<25} {"queued":<12} {"-":<10}')

    try:
        import json
        import urllib.request

        r = urllib.request.urlopen("http://127.0.0.1:8002/skills", timeout=3)
        data = json.loads(r.read())
        print()
        print("Tabula Rasa AI status:")
        for s in data["skills"]:
            print(f'  {s["status"]:>8}: {s["name"]}')
    except:
        print("\n  (Tabula Rasa AI not running on port 8002)")


def assign_skills():
    """Print skill assignments for each computer."""
    machines = [
        ("Main (Intel)", ["math/add"]),
        ("Worker 2", ["math/sub"]),
        ("Worker 3", ["math/mul"]),
        ("Worker 4", ["math/div"]),
    ]
    print(f"\nSkill Assignments:")
    print(f'{"Computer":<20} {"Skill":<20} {"Est. Time":<12}')
    print("-" * 52)
    for machine, skills in machines:
        for skill in skills:
            print(f'{machine:<20} {skill:<20} {"~8h":<12}')


# ─── Sync / Import ─────────────────────────────────────────────────


def sync_from_share(share_path: str):
    """Import completed specialists from a network share."""
    share = Path(share_path)
    if not share.exists():
        print(f"[!] Share not found: {share}")
        return

    zips = list(share.glob("specialist_*.zip"))
    if not zips:
        print(f"[*] No specialist zips found in {share}")
        print(f"    Place files like: specialist_math_subtraction.zip")
        return

    for zf in zips:
        print(f"[*] Found: {zf.name}")
        from export_specialist import import_specialist

        import_specialist(str(zf))
        imported_dir = share / "_imported"
        imported_dir.mkdir(exist_ok=True)
        zf.rename(imported_dir / zf.name)
        print(f"    Moved to {imported_dir / zf.name}")


# ─── Watch Daemon ──────────────────────────────────────────────────


def auto_watch(share_path: str, poll_interval: int = 60, monitor_heartbeats: bool = False):
    """Watch a network share for new specialists, with optional heartbeat monitoring.

    Args:
        share_path: Path to the shared NAS folder.
        poll_interval: Seconds between polls for new zips.
        monitor_heartbeats: If True, also checks for stale worker heartbeats.
    """
    print(f"[*] Watching {share_path} for completed specialists...")
    print(f"    Polling every {poll_interval}s. Ctrl+C to stop.")
    if monitor_heartbeats:
        print(f"    Heartbeat monitoring ON (stale timeout: {STALE_TIMEOUT}s)")
    last_files = set()

    # Track which tasks have been re-queued to avoid spam
    re_queued = set()

    try:
        while True:
            share = Path(share_path)
            if share.exists():
                # ── Check for new completed zips ──
                current = set(share.glob("specialist_*.zip"))
                new = current - last_files
                for zf in new:
                    print(f"\n[!] New specialist detected: {zf.name}")
                    from export_specialist import import_specialist

                    import_specialist(str(zf))
                    imported_dir = share / "_imported"
                    imported_dir.mkdir(exist_ok=True)
                    zf.rename(imported_dir / zf.name)
                last_files = current

                # ── Check heartbeats ──
                if monitor_heartbeats:
                    stale = check_heartbeats(share_path)
                    for worker in stale:
                        task = worker["task"]
                        hostname = worker["hostname"]
                        age_m = worker["age_seconds"] // 60
                        print(f"\n[!] Worker {hostname} STALE ({age_m}m ago) on task: {task}")

                        # Re-queue the task if not already handled
                        if task not in re_queued:
                            # Remove stale heartbeat so it's not re-detected
                            try:
                                Path(worker["file"]).unlink(missing_ok=True)
                            except OSError:
                                pass
                            # Write a failure flag for the coordinator
                            fail_file = share / f'failed_{hostname}_{task.replace("/", "_")}.json'
                            fail_file.write_text(
                                json.dumps(
                                    {
                                        "worker": hostname,
                                        "task": task,
                                        "failed_at": int(time.time()),
                                        "reason": f"No heartbeat for {age_m} min",
                                        "action": "re-queued",
                                    },
                                    indent=2,
                                )
                            )
                            print(f"    Re-queued {task} — written {fail_file.name}")
                            re_queued.add(task)

                    # Clean up stale re-queue tracking after 1 hour
                    now = int(time.time())
                    for f in share.glob("failed_*.json"):
                        try:
                            data = json.loads(f.read_text())
                            if now - data.get("failed_at", 0) > 3600:
                                f.unlink(missing_ok=True)
                        except (json.JSONDecodeError, OSError):
                            pass

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[*] Watcher stopped.")


# ─── Work On (with Heartbeats) ─────────────────────────────────────


def work_on(skill: str, share_path: str = None):
    """Train a specific specialist on this machine with heartbeats.

    Writes heartbeat_<hostname>.json to CWD (or share_path if given)
    every 5 minutes so the coordinator knows this worker is alive.
    """
    # Map skill paths to operation short names
    SKILL_TO_OP = {
        "math/add": "add",
        "math/addition": "add",
        "math/sub": "sub",
        "math/subtraction": "sub",
        "math/mul": "mul",
        "math/multiplication": "mul",
        "math/div": "div",
        "math/division": "div",
    }
    op = SKILL_TO_OP.get(skill, skill.split("/")[-1][:3])  # Fallback: first 3 chars

    if op not in ("add", "sub", "mul", "div"):
        print(f"[!] Unknown skill: {skill}. Expected math/add, math/sub, etc.")
        return

    print(f"[*] This machine assigned to: {skill} -> {op}")
    print(f"    Hostname: {get_hostname()}")

    # ── Write initial heartbeat ──
    if share_path:
        ensure_share(share_path)
    write_heartbeat(skill, "training", share_path or ".")
    print(f'    Heartbeat: {"OK" if share_path else "local only (no share)"}')

    # ── Start heartbeat daemon ──
    if share_path:
        hb_thread = heartbeat_daemon(skill, share_path)
        print(f"    Heartbeat daemon: every {HEARTBEAT_INTERVAL}s")

    # ── Run training ──
    print(f"[*] Starting training ({op})...")
    cmd = [sys.executable, "train_specialist.py", op]
    result = subprocess.run(cmd)

    # ── Final heartbeat ──
    if result.returncode == 0:
        write_heartbeat(skill, "done", share_path or ".")
        print(f"\n[*] Training complete! Exporting...")
        export_cmd = [sys.executable, "export_specialist.py", skill]
        write_heartbeat(skill, "exporting", share_path or ".")
        subprocess.run(export_cmd)
        write_heartbeat(skill, "done", share_path or ".")
    else:
        write_heartbeat(skill, "error", share_path or ".")
        print(f"\n[!] Training failed (exit code {result.returncode})")


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} list                 # Show all skills")
        print(f"  {sys.argv[0]} assign               # Show assignments")
        print(f"  {sys.argv[0]} work <skill> [share]  # Train with heartbeats")
        print(f"  {sys.argv[0]} sync <path>           # Import completed zips")
        print(f"  {sys.argv[0]} watch <path> [--h]    # Auto-import + heartbeat monitor")
        print()
        print("Example workflow:")
        print("  PC1> python3 specialist_network.py work math/addition //NAS/shared")
        print("  PC2> python3 specialist_network.py work math/subtraction //NAS/shared")
        print("  PC3> python3 specialist_network.py work math/multiplication //NAS/shared")
        print("  PC4> python3 specialist_network.py work math/division //NAS/shared")
        print("  Main> python3 specialist_network.py watch //NAS/shared --h")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "list":
        list_skills()
    elif cmd == "assign":
        assign_skills()
    elif cmd == "work":
        if len(sys.argv) < 3:
            print("Specify skill: python3 specialist_network.py work math/subtraction")
            sys.exit(1)
        skill = sys.argv[2]
        share = sys.argv[3] if len(sys.argv) > 3 else None
        work_on(skill, share)
    elif cmd == "sync":
        sync_from_share(sys.argv[2] if len(sys.argv) > 2 else "D:/shared")
    elif cmd == "watch":
        path = sys.argv[2] if len(sys.argv) > 2 else "D:/shared"
        monitor_hb = "--h" in sys.argv or "-h" in sys.argv or "--heartbeat" in sys.argv
        auto_watch(path, monitor_heartbeats=monitor_hb)
    else:
        print(f"Unknown command: {cmd}")
