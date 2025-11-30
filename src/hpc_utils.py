import os
import shutil
import subprocess
from pathlib import Path

def get_hpc_scratch_dir():
    return os.environ.get('BLACKHOLE', None)

def is_on_hpc():
    return get_hpc_scratch_dir() is not None

def get_data_dir(base_dir=None):
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    if is_on_hpc():
        scratch_dir = get_hpc_scratch_dir()
        return os.path.join(scratch_dir, 'data')
    else:
        return os.path.join(base_dir, 'data')

def get_results_dir(base_dir=None):
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    if is_on_hpc():
        scratch_dir = get_hpc_scratch_dir()
        return os.path.join(scratch_dir, 'results')
    else:
        return os.path.join(base_dir, 'results')

def setup_hpc_directories():
    if not is_on_hpc():
        return
    scratch_dir = get_hpc_scratch_dir()
    directories = ['data', 'results', 'results/models', 'results/plots', 'results/logs']
    for d in directories:
        directory = os.path.join(scratch_dir, d)
        os.makedirs(directory, exist_ok=True)

def sync_data_to_hpc(local_data_dir, hpc_data_dir=None):
    if not is_on_hpc():
        return
    if hpc_data_dir is None:
        hpc_data_dir = get_data_dir()
    os.makedirs(hpc_data_dir, exist_ok=True)
    try:
        subprocess.run(['rsync', '-av', '--exclude', '__pycache__', '--exclude', '*.pyc', f'{local_data_dir}/', f'{hpc_data_dir}/'], check=True)
    except subprocess.CalledProcessError:
        if os.path.exists(local_data_dir):
            for item in os.listdir(local_data_dir):
                src = os.path.join(local_data_dir, item)
                dst = os.path.join(hpc_data_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

def sync_results_from_hpc(hpc_results_dir=None, local_results_dir=None):
    if not is_on_hpc():
        return
    if hpc_results_dir is None:
        hpc_results_dir = get_results_dir()
    if local_results_dir is None:
        local_results_dir = os.path.join(Path(__file__).parent.parent, 'results')
    os.makedirs(local_results_dir, exist_ok=True)
    try:
        subprocess.run(['rsync', '-av', f'{hpc_results_dir}/', f'{local_results_dir}/'], check=True)
    except subprocess.CalledProcessError:
        pass
