import os
import subprocess
import sys
import time
import shutil
import ctypes
from pathlib import Path

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status_type="info"):
    timestamp = time.strftime("%H:%M:%S")
    if status_type == "info":
        print(f"{Colors.BLUE}[{timestamp}] INFO: {message}{Colors.ENDC}")
    elif status_type == "success":
        print(f"{Colors.GREEN}[{timestamp}] SUCCESS: {message}{Colors.ENDC}")
    elif status_type == "warning":
        print(f"{Colors.WARNING}[{timestamp}] WARNING: {message}{Colors.ENDC}")
    elif status_type == "error":
        print(f"{Colors.FAIL}[{timestamp}] ERROR: {message}{Colors.ENDC}")
    elif status_type == "header":
        print(f"\n{Colors.HEADER}{Colors.BOLD}[{timestamp}] {message}{Colors.ENDC}\n")

def verify_espeak_library(dll_path):
    try:
        lib = ctypes.CDLL(dll_path)
        if hasattr(lib, 'espeak_Initialize'):
            result = lib.espeak_Initialize(0, 0, None, 0)
            if result >= 0:
                print_status(f"Successfully initialized eSpeak library at {dll_path}", "success")
                return True
        print_status("Library found but initialization failed", "error")
        return False
    except Exception as e:
        print_status(f"Failed to load eSpeak library: {e}", "error")
        return False

def prepare_environment():
    print_status("Preparing environment...", "header")
    
    if 'VIRTUAL_ENV' not in os.environ:
        print_status("No virtual environment detected!", "error")
        return False

    venv_path = os.environ['VIRTUAL_ENV']
    scripts_dir = os.path.join(venv_path, 'Scripts')
    site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
    espeak_install = r"C:\Program Files\eSpeak NG"

    try:
        print_status("Installing phonemizer...")
        subprocess.run(['uv', 'pip', 'install', 'phonemizer==3.2.1'], check=True)
        
        phonemizer_dir = os.path.join(site_packages, 'phonemizer')
        bin_dir = os.path.join(phonemizer_dir, 'bin')
        os.makedirs(bin_dir, exist_ok=True)

        print_status("Copying eSpeak files...")
        files_to_copy = [
            ('espeak-ng.exe', 'executable'),
            ('libespeak-ng.dll', 'library'),
            ('libwinpthread-1.dll', 'dependency')
        ]
        
        for file_name, file_type in files_to_copy:
            src = os.path.join(espeak_install, file_name)
            if os.path.exists(src):
                for dest_dir in [scripts_dir, bin_dir]:
                    dest = os.path.join(dest_dir, file_name)
                    shutil.copy2(src, dest)
                    print_status(f"Copied {file_name} to {dest_dir}", "success")
            else:
                print_status(f"Warning: {file_name} not found in eSpeak installation", "warning")

        data_src = os.path.join(espeak_install, 'espeak-ng-data')
        if os.path.exists(data_src):
            for dest_dir in [scripts_dir, bin_dir]:
                data_dest = os.path.join(dest_dir, 'espeak-ng-data')
                if os.path.exists(data_dest):
                    shutil.rmtree(data_dest)
                shutil.copytree(data_src, data_dest)
                print_status(f"Copied espeak-ng-data to {dest_dir}", "success")
        
        dll_paths = [
            os.path.join(scripts_dir, 'libespeak-ng.dll'),
            os.path.join(bin_dir, 'libespeak-ng.dll')
        ]
        
        valid_dll = None
        for dll_path in dll_paths:
            if verify_espeak_library(dll_path):
                valid_dll = dll_path
                break
        
        if not valid_dll:
            print_status("Failed to verify eSpeak library in any location", "error")
            return False
        
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = valid_dll
        os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.dirname(valid_dll)
        os.environ['PATH'] = f"{os.path.dirname(valid_dll)}{os.pathsep}{os.environ['PATH']}"
        
        print_status("Testing phonemizer...")
        test_script = """
import sys
if sys.platform == 'win32': sys.stdout.reconfigure(encoding='utf-8')
import phonemizer
from phonemizer.backend import EspeakBackend
backend = EspeakBackend('en-us', language_switch='remove-flags')
print("Phonemizer test successful!")
"""
        result = subprocess.run([sys.executable, '-c', test_script],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print_status("Phonemizer test successful!", "success")
            return True
        else:
            print_status(f"Phonemizer test failed: {result.stderr}", "error")
            return False

    except Exception as e:
        print_status(f"Error during environment preparation: {e}", "error")
        return False

def install_packages():
    print_status("Starting package installation...", "header")
    
    try:
        print_status("Installing dependencies from requirements.txt and wheels...")
        # First install jaraco.functools which is a dependency
        subprocess.run(['uv', 'pip', 'install', 'jaraco.functools'], check=True)
        
        # Then install the rest of the requirements
        subprocess.run(['uv', 'pip', 'install', '--find-links=wheels/', '-r', 'requirements.txt'], check=True)
        
        print_status("Package installation completed", "success")

    except Exception as e:
        print_status(f"Error during package installation: {e}", "error")
        raise

def main():
    if sys.platform != 'win32':
        print_status("This script is for Windows only", "error")
        sys.exit(1)

    try:
        if not prepare_environment():
            print_status("Failed to prepare environment", "error")
            sys.exit(1)
        
        install_packages()
        print_status("Installation completed successfully", "success")
    except Exception as e:
        print_status(f"Installation failed: {str(e)}", "error")
        sys.exit(1)

if __name__ == "__main__":
    main()
