"""Export a specialist as a portable package.

Usage:
    python3 export_specialist.py math/addition
    python3 export_specialist.py math/addition --out \\\\other-pc\\shared\\specialists\\
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shutil
import sys
import zipfile
from pathlib import Path

SPECIALISTS_DIR = Path("specialists")


def export_specialist(skill_path: str, output: str = None):
    """Package a specialist into a portable zip file."""
    src = SPECIALISTS_DIR / skill_path
    if not src.exists():
        print(f"[!] Specialist not found: {src}")
        sys.exit(1)

    if output:
        out_path = Path(output)
    else:
        out_path = Path(f'specialist_{skill_path.replace("/", "_")}.zip')

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src.rglob("*"):
            if f.is_file():
                arcname = str(f.relative_to(SPECIALISTS_DIR))
                zf.write(f, arcname)

    size = out_path.stat().st_size / 1e6
    print(f"  [*] Exported {skill_path} -> {out_path} ({size:.1f} MB)")
    print("  [*] Files:")
    with zipfile.ZipFile(out_path, "r") as zf:
        for info in zf.infolist():
            print(f"      {info.filename} ({info.file_size/1e3:.0f} KB)")

    return out_path


def import_specialist(zip_path: str, target: str = None):
    """Import a specialist from a zip file."""
    zf_path = Path(zip_path)
    if not zf_path.exists():
        print(f"[!] File not found: {zip_path}")
        sys.exit(1)

    with zipfile.ZipFile(zf_path, "r") as zf:
        # Determine target path from zip contents
        members = zf.infolist()
        if not members:
            print("[!] Empty zip")
            return

        # Get the base directory from the first entry
        first = members[0].filename.split("/")[0]
        if target:
            dest = SPECIALISTS_DIR / target
        else:
            dest = SPECIALISTS_DIR / first

        if dest.exists():
            print(f"[!] Target already exists: {dest}")
            yn = input("  Overwrite? (y/N): ")
            if yn.lower() != "y":
                print("  Cancelled.")
                return
            shutil.rmtree(dest)

        zf.extractall(SPECIALISTS_DIR)

    print(f"  [*] Imported to {dest}")
    for f in dest.rglob("*"):
        if f.is_file():
            print(f"      {f.relative_to(SPECIALISTS_DIR)}")
    print("  [*] Restart tabula_rasa.py to load the new specialist")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export/import a specialist as a portable package."
    )
    parser.add_argument(
        "skill_path", nargs="?", default=None, help="Path to specialist (e.g. math/addition)"
    )
    parser.add_argument("--out", default=None, help="Output directory for export")
    parser.add_argument(
        "--import-zip", dest="import_zip", default=None, help="Import a specialist from a zip file"
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[8, 4],
        default=None,
        help="Quantize model to 8-bit or 4-bit during export (requires bitsandbytes)",
    )
    args = parser.parse_args()

    if args.import_zip:
        import_specialist(args.import_zip, args.out)
    elif args.skill_path:
        export_specialist(args.skill_path, args.out)
    else:
        parser.print_help()
        sys.exit(1)
