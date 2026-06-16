"""Auto-install script — detects hardware and installs the correct PyTorch.

Detects: CUDA (NVIDIA), ROCm (AMD), MPS (Apple Silicon), CPU
Installs the optimal PyTorch + dependencies for the detected hardware.

Usage:
    python3 scripts/install.py                     # Auto-detect and install
    python3 scripts/install.py --cpu               # Force CPU install
    python3 scripts/install.py --cuda              # Force CUDA install
    python3 scripts/install.py --dev               # Include dev deps (pytest, mypy)
    python3 scripts/install.py --check-only        # Only print detected hardware
"""

import subprocess, sys, platform, os, re
from pathlib import Path


def detect_hardware() -> dict:
    """Detect available hardware accelerators."""
    info = {
        'os': platform.system(),
        'arch': platform.machine(),
        'python': sys.version.split()[0],
        'cuda': False,
        'cuda_version': None,
        'rocm': False,
        'mps': False,
        'cpu_cores': os.cpu_count() or 1,
    }

    # Check for NVIDIA CUDA via nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,cuda_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            info['cuda'] = True
            parts = result.stdout.strip().split(', ')
            info['gpu_name'] = parts[0] if len(parts) > 0 else 'NVIDIA GPU'
            info['cuda_version'] = parts[2] if len(parts) > 2 else '12'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for AMD ROCm
    if not info['cuda']:
        try:
            result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['rocm'] = True
        except FileNotFoundError:
            pass
        # Also check for /opt/rocm
        if Path('/opt/rocm').exists():
            info['rocm'] = True

    # Check for Apple MPS
    if platform.system() == 'Darwin' and platform.machine() in ('arm64', 'aarch64'):
        info['mps'] = True

    return info


def get_install_command(hardware: dict, dev: bool = False, force: str | None = None) -> list[str]:
    """Build the pip install command for the detected hardware."""
    base = [sys.executable, '-m', 'pip', 'install']

    if force == 'cpu' or (not hardware['cuda'] and not hardware['rocm'] and not hardware['mps']):
        # CPU-only PyTorch (smallest install)
        torch_pkg = 'torch torchvision --index-url https://download.pytorch.org/whl/cpu'
    elif force == 'cuda' or hardware['cuda']:
        # CUDA version
        cuda_ver = hardware.get('cuda_version', '12')
        major = cuda_ver.split('.')[0]
        index_urls = {'11': 'cu118', '12': 'cu121'}
        suffix = index_urls.get(major, 'cu121')
        torch_pkg = f'torch torchvision --index-url https://download.pytorch.org/whl/{suffix}'
    elif force == 'rocm' or hardware['rocm']:
        torch_pkg = 'torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2'
    elif hardware['mps']:
        # MPS uses default PyTorch (which includes MPS support on macOS)
        torch_pkg = 'torch torchvision'
    else:
        torch_pkg = 'torch torchvision --index-url https://download.pytorch.org/whl/cpu'

    # Core deps
    deps = 'numpy tqdm'
    dev_deps = ' pytest mypy pre-commit' if dev else ''

    return base + [f'{torch_pkg} {deps}{dev_deps}']


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Auto-install Tabula Rasa dependencies')
    parser.add_argument('--cpu', action='store_true', help='Force CPU-only install')
    parser.add_argument('--cuda', action='store_true', help='Force CUDA install')
    parser.add_argument('--rocm', action='store_true', help='Force ROCm install')
    parser.add_argument('--dev', action='store_true', help='Include dev dependencies')
    parser.add_argument('--check-only', action='store_true', help='Only detect and print hardware')
    args = parser.parse_args()

    hardware = detect_hardware()

    print(f'\n{"="*50}')
    print(f'  Tabula Rasa — Auto Installer')
    print(f'{"="*50}')
    print(f'  OS:      {hardware["os"]} ({hardware["arch"]})')
    print(f'  Python:  {hardware["python"]}')
    print(f'  Cores:   {hardware["cpu_cores"]}')
    print(f'  CUDA:    {"Yes (" + hardware.get("gpu_name", "") + " v" + hardware.get("cuda_version", "") + ")" if hardware["cuda"] else "No"}')
    print(f'  ROCm:    {"Yes" if hardware["rocm"] else "No"}')
    print(f'  MPS:     {"Yes" if hardware["mps"] else "No"}')

    if args.check_only:
        print(f'{"="*50}')
        return

    # Determine force
    force = None
    if args.cpu:
        force = 'cpu'
    elif args.cuda:
        force = 'cuda'
    elif args.rocm:
        force = 'rocm'

    cmd = get_install_command(hardware, dev=args.dev, force=force)
    pip_cmd = ' '.join(cmd)
    print(f'\n  Installing: {pip_cmd}')
    print(f'{"="*50}\n')

    result = subprocess.run(pip_cmd, shell=True)
    if result.returncode == 0:
        print(f'\n{"="*50}')
        print(f'  Installation complete!')
        print(f'{"="*50}')
    else:
        print(f'\n  [!] Installation failed (exit code {result.returncode})')
        print(f'  Try: {sys.executable} -m pip install torch numpy tqdm')
        sys.exit(1)


if __name__ == '__main__':
    main()
