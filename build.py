import PyInstaller.__main__
import os
import nicegui
from pathlib import Path

# Get the path to nicegui so we can include its static assets
nicegui_dir = Path(nicegui.__file__).parent

args = [
    'main.py',
    '--name=OpenYuGi',
    '--onedir',
    # Do not use windowed mode since we want NiceGUI to launch the browser and show logs
    '--console',
    f'--add-data={nicegui_dir}{os.pathsep}nicegui',
    '--clean',
    '--noconfirm',
]

# Add yolo26l-cls.pt if it exists in the project root
if os.path.exists('yolo26l-cls.pt'):
    args.append(f'--add-data=yolo26l-cls.pt{os.pathsep}.')

PyInstaller.__main__.run(args)
