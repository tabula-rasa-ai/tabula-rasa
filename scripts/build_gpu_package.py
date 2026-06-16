"""Build a GPU deployment package for Tabula Rasa.

Packages the essential project files into a deployable archive
for cloud GPU training (Colab, Lambda, Paperspace, etc.).

Usage:
    python3 scripts/build_gpu_package.py              # Build deploy.zip
    python3 scripts/build_gpu_package.py --output my_run.zip
    python3 scripts/build_gpu_package.py --include-specialists  # Include trained checkpoints
"""

import zipfile, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ESSENTIAL_PATTERNS = [
    # Source
    'src/tabula_rasa/*.py',
    # Config
    'config.py',
    # Core scripts
    'train_specialist.py',
    'auto_train.py',
    'api_server.py',
    # Egefalos modules
    'egefalos/*.py',
    # Scripts
    'scripts/socratic_benchmark.py',
    'scripts/validate_code_az.py',
    'scripts/train_text.py',
    'scripts/train_algebra.py',
    # Docs
    'README.md',
    'QUICKSTART.md',
    'ROADMAP.md',
    'GPU_BENCHMARKS.md',
    'pyproject.toml',
    'requirements.txt',
]

# Colab-specific wrapper
COLAB_SETUP_SCRIPT = """import sys, subprocess, os
from pathlib import Path

# Install dependencies
subprocess.run(['pip', 'install', '-q', 'torch', 'numpy', 'tqdm'], check=True)

# Mount Google Drive (optional, for saving checkpoints)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Google Drive mounted at /content/drive')
except Exception:
    print('Not in Colab or Drive mount failed — saving locally')

# Set up project
os.chdir('/content/tabula-rasa')
sys.path.insert(0, 'src')

print('\\nTabula Rasa GPU deployment ready!')
print(f'  CUDA: {__import__(\"torch\").cuda.is_available()}')
print(f'  GPU:  {__import__(\"torch\").cuda.get_device_name(0) if __import__(\"torch\").cuda.is_available() else \"None\"}')
print()
print('To run Socratic benchmark:')
print('  python3 scripts/socratic_benchmark.py --seeds 5 --steps 30000')
print()
print('To train a single specialist:')
print('  python3 train_specialist.py add --steps 30000 --batch 256')
print()
print('To test a quick run:')
print('  python3 train_specialist.py add --quick')
"""


def build_package(output_path: str = 'deploy.zip', include_specialists: bool = False):
    """Build a deployment zip for GPU training."""
    import glob

    output_path = ROOT / output_path
    files_added = 0

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add essential Python files
        for pattern in ESSENTIAL_PATTERNS:
            for f in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
                f_path = Path(f)
                arcname = str(f_path.relative_to(ROOT))
                zf.write(f_path, arcname)
                files_added += 1

        # Optionally include trained specialists
        if include_specialists:
            for ckpt in sorted((ROOT / 'specialists' / 'math').rglob('*.pt')):
                arcname = str(ckpt.relative_to(ROOT))
                zf.write(ckpt, arcname)
                files_added += 1
            # Include tokenizers
            for tok_file in (ROOT / 'specialists').rglob('tokenizer.json'):
                arcname = str(tok_file.relative_to(ROOT))
                zf.write(tok_file, arcname)
                files_added += 1

        # Add Colab setup script
        zf.writestr('colab_setup.py', COLAB_SETUP_SCRIPT)
        files_added += 1

        # Add simplified requirements for Colab
        zf.writestr('colab_requirements.txt', 'torch>=2.0.0\nnumpy>=1.24.0\ntqdm>=4.60.0\n')
        files_added += 1

        # Manifest
        manifest = {
            'built': __import__('datetime').datetime.now().isoformat(),
            'files': files_added,
            'has_specialists': include_specialists,
            'instructions': [
                '1. Upload deploy.zip to Colab /content/',
                '2. Run: !unzip -q deploy.zip -d /content/tabula-rasa',
                '3. Run: %run colab_setup.py',
                '4. Run: python3 scripts/socratic_benchmark.py --seeds 5 --steps 30000',
            ],
        }
        zf.writestr('manifest.json', json.dumps(manifest, indent=2))
        files_added += 1

    size_mb = output_path.stat().st_size / 1_000_000
    print(f'\n{"="*50}')
    print(f'  GPU Deployment Package')
    print(f'{"="*50}')
    print(f'  Output:    {output_path}')
    print(f'  Size:      {size_mb:.1f} MB')
    print(f'  Files:     {files_added}')
    print(f'  Specialists: {"Yes" if include_specialists else "No (smaller package)"}')
    print(f'{"="*50}')
    print(f'\n  Deployment instructions:')
    for step in manifest['instructions']:
        print(f'    {step}')

    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build GPU deployment package')
    parser.add_argument('--output', type=str, default='deploy.zip',
                        help='Output zip file (default: deploy.zip)')
    parser.add_argument('--include-specialists', action='store_true',
                        help='Include trained specialist checkpoints in package')
    args = parser.parse_args()
    build_package(output_path=args.output, include_specialists=args.include_specialists)
