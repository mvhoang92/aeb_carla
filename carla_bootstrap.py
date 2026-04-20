import glob
import os
import sys
from pathlib import Path


def _add_path(path):
    path = Path(path)
    if not path.exists():
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def bootstrap_carla():
    current_dir = Path(__file__).resolve().parent
    search_roots = [
        current_dir,
        current_dir.parent,
        Path.cwd(),
    ]

    platform_tag = 'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    fallback_tag = 'linux-x86_64' if platform_tag == 'win-amd64' else 'win-amd64'
    egg_patterns = []

    for root in search_roots:
        _add_path(root / 'WindowsNoEditor' / 'PythonAPI')
        _add_path(root / 'WindowsNoEditor' / 'PythonAPI' / 'carla')
        _add_path(root / 'PythonAPI')
        _add_path(root / 'PythonAPI' / 'carla')

        egg_patterns.extend([
            root / 'WindowsNoEditor' / 'PythonAPI' / 'carla' / 'dist' / f'carla-*-{platform_tag}.egg',
            root / 'PythonAPI' / 'carla' / 'dist' / f'carla-*-{platform_tag}.egg',
            root / 'WindowsNoEditor' / 'PythonAPI' / 'carla' / 'dist' / f'carla-*-{fallback_tag}.egg',
            root / 'PythonAPI' / 'carla' / 'dist' / f'carla-*-{fallback_tag}.egg',
        ])

    matches = []
    for pattern in egg_patterns:
        matches.extend(glob.glob(str(pattern)))

    for match in sorted(set(matches)):
        _add_path(match)

    if not matches:
        raise FileNotFoundError(
            'Khong tim thay CARLA Python API trong WindowsNoEditor/PythonAPI/carla/dist.'
        )
