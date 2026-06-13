"""Export a specialist as a portable package.

Usage:
    python3 export_specialist.py math/addition
    python3 export_specialist.py math/addition --out \\\\other-pc\\shared\\specialists\\
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import sys, json, zipfile, shutil
from pathlib import Path

SPECIALISTS_DIR = Path('specialists')


def export_specialist(skill_path: str, output: str = None):
    """Package a specialist into a portable zip file."""
    src = SPECIALISTS_DIR / skill_path
    if not src.exists():
        print(f'[!] Specialist not found: {src}')
        sys.exit(1)

    if output:
        out_path = Path(output)
    else:
        out_path = Path(f'specialist_{skill_path.replace("/", "_")}.zip')

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in src.rglob('*'):
            if f.is_file():
                arcname = str(f.relative_to(SPECIALISTS_DIR))
                zf.write(f, arcname)

    size = out_path.stat().st_size / 1e6
    print(f'  [*] Exported {skill_path} -> {out_path} ({size:.1f} MB)')
    print(f'  [*] Files:')
    with zipfile.ZipFile(out_path, 'r') as zf:
        for info in zf.infolist():
            print(f'      {info.filename} ({info.file_size/1e3:.0f} KB)')

    return out_path


def import_specialist(zip_path: str, target: str = None):
    """Import a specialist from a zip file."""
    zf_path = Path(zip_path)
    if not zf_path.exists():
        print(f'[!] File not found: {zip_path}')
        sys.exit(1)

    with zipfile.ZipFile(zf_path, 'r') as zf:
        # Determine target path from zip contents
        members = zf.infolist()
        if not members:
            print('[!] Empty zip')
            return

        # Get the base directory from the first entry
        first = members[0].filename.split('/')[0]
        if target:
            dest = SPECIALISTS_DIR / target
        else:
            dest = SPECIALISTS_DIR / first

        if dest.exists():
            print(f'[!] Target already exists: {dest}')
            yn = input('  Overwrite? (y/N): ')
            if yn.lower() != 'y':
                print('  Cancelled.')
                return
            shutil.rmtree(dest)

        zf.extractall(SPECIALISTS_DIR)

    print(f'  [*] Imported to {dest}')
    for f in dest.rglob('*'):
        if f.is_file():
            print(f'      {f.relative_to(SPECIALISTS_DIR)}')
    print(f'  [*] Restart tabula_rasa.py to load the new specialist')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:')
        print('  Export: python3 export_specialist.py math/addition')
        print('  Export: python3 export_specialist.py math/addition --out D:/backups/')
        print('  Import: python3 export_specialist.py --import specialist_math_addition.zip')
        sys.exit(1)

    if sys.argv[1] == '--import':
        import_specialist(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        out = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == '--out' else None
        export_specialist(sys.argv[1], out)
