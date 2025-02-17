import os
import subprocess
import winreg
import sys
import time
import shutil

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

def get_espeak_ng_installation_path():
    """Find the espeak binary in the installation directory."""
    print_status("Searching for eSpeak NG installation...", "header")
    
    # First check common installation paths
    common_paths = [
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG",
    ]
    for path in common_paths:
        print_status(f"Checking path: {path}")
        dll_path = os.path.join(path, 'libespeak-ng.dll')
        if os.path.isfile(dll_path):
            print_status(f"Found eSpeak NG at: {path}", "success")
            return path
    
    print_status("eSpeak NG not found in common locations", "warning")
    return None

def install_espeak_ng():
    print_status("Starting eSpeak NG installation...", "header")
    try:
        print_status("Using winget to install eSpeak NG...")
        result = subprocess.run(
            ["winget", "install", "--id=eSpeak-NG.eSpeak-NG", "-e", "--silent",
             "--accept-source-agreements", "--accept-package-agreements"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        if result.returncode == 0:
            print_status("eSpeak NG installation completed successfully", "success")
        else:
            print_status(f"Installation output: {result.stdout}\nErrors: {result.stderr}", "warning")
            raise subprocess.CalledProcessError(result.returncode, result.args)
            
    except subprocess.CalledProcessError as e:
        print_status(f"Installation failed: {str(e)}", "error")
        raise RuntimeError("eSpeak NG installation failed")

def set_system_path(espeak_path):
    try:
        # Get the current system PATH
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_ALL_ACCESS)
        current_path = winreg.QueryValueEx(key, 'Path')[0]
        
        # Check if eSpeak path is already in PATH
        if espeak_path.lower() in current_path.lower():
            print_status("eSpeak NG already in system PATH", "success")
            return True
            
        # Add eSpeak path to PATH
        new_path = current_path + ';' + espeak_path
        winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
        winreg.CloseKey(key)
        print_status("Added eSpeak NG to system PATH", "success")
        
        # Notify about system update
        print_status("System PATH updated. You may need to restart your terminal or computer for changes to take effect.", "warning")
        return True
        
    except Exception as e:
        print_status(f"Failed to update system PATH: {str(e)}", "error")
        print_status("Please add the following path to your system PATH manually:", "warning")
        print_status(espeak_path, "info")
        return False

def setup_taproot_compatibility(espeak_dir):
    """Set up environment for taproot compatibility."""
    try:
        # Get virtual environment Scripts directory
        if 'VIRTUAL_ENV' not in os.environ:
            print_status("No virtual environment detected!", "error")
            return False

        scripts_dir = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts')
        bin_dir = os.path.join(os.environ['VIRTUAL_ENV'], 'bin')
        
        # Create bin directory if it doesn't exist
        os.makedirs(bin_dir, exist_ok=True)

        # Copy all necessary files to Scripts directory
        print_status("Copying eSpeak files to virtual environment...")
        
        # List of files to copy
        files_to_copy = [
            'espeak-ng.exe',
            'libespeak-ng.dll',
            'libwinpthread-1.dll'  # Common dependency
        ]
        
        for file_name in files_to_copy:
            src = os.path.join(espeak_dir, file_name)
            if os.path.exists(src):
                # Copy to Scripts
                dst_scripts = os.path.join(scripts_dir, file_name)
                shutil.copy2(src, dst_scripts)
                print_status(f"Copied {file_name} to Scripts directory", "success")
                
                # Copy to bin
                dst_bin = os.path.join(bin_dir, file_name)
                shutil.copy2(src, dst_bin)
                print_status(f"Copied {file_name} to bin directory", "success")

        # Copy espeak-ng-data directory
        data_src = os.path.join(espeak_dir, 'espeak-ng-data')
        if os.path.exists(data_src):
            data_dst_scripts = os.path.join(scripts_dir, 'espeak-ng-data')
            data_dst_bin = os.path.join(bin_dir, 'espeak-ng-data')
            
            if os.path.exists(data_dst_scripts):
                shutil.rmtree(data_dst_scripts)
            if os.path.exists(data_dst_bin):
                shutil.rmtree(data_dst_bin)
                
            shutil.copytree(data_src, data_dst_scripts)
            shutil.copytree(data_src, data_dst_bin)
            print_status("Copied espeak-ng-data directory", "success")

        # Create Unix-style symlink for taproot
        dll_path = os.path.join(scripts_dir, 'libespeak-ng.dll')
        unix_style_path = os.path.join(scripts_dir, 'libespeak.so.1')
        
        # Remove existing file if it exists
        if os.path.exists(unix_style_path):
            os.remove(unix_style_path)
        
        # Copy DLL to Unix-style name
        shutil.copy2(dll_path, unix_style_path)
        print_status("Created Unix-style library link for taproot", "success")

        # Set environment variables
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = dll_path
        os.environ['PHONEMIZER_ESPEAK_PATH'] = scripts_dir
        os.environ['TAPROOT_ESPEAK_LIBRARY'] = unix_style_path
        
        print_status("Environment variables set:", "success")
        print_status(f"PHONEMIZER_ESPEAK_LIBRARY={dll_path}")
        print_status(f"PHONEMIZER_ESPEAK_PATH={scripts_dir}")
        print_status(f"TAPROOT_ESPEAK_LIBRARY={unix_style_path}")

        return True
    except Exception as e:
        print_status(f"Error setting up taproot compatibility: {e}", "error")
        return False

def verify_installation():
    """Verify eSpeak installation and environment setup."""
    try:
        result = subprocess.run(['espeak-ng', '--version'], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            print_status("eSpeak NG installation verified", "success")
            print_status(f"Version: {result.stdout.strip()}")
            return True
        else:
            print_status("Failed to verify eSpeak NG installation", "error")
            return False
    except Exception as e:
        print_status(f"Error verifying installation: {e}", "error")
        return False

def main():
    if not sys.platform == 'win32':
        print_status("This script is for Windows only", "error")
        sys.exit(1)

    try:
        print_status("Starting Step 1: eSpeak NG Installation", "header")
        
        # Check if eSpeak is already installed
        espeak_path = get_espeak_ng_installation_path()
        if not espeak_path:
            print_status("eSpeak NG not found, starting installation...")
            install_espeak_ng()
            # Check again after installation
            espeak_path = get_espeak_ng_installation_path()
            if not espeak_path:
                raise RuntimeError("Failed to find eSpeak NG after installation")
        
        # Set system PATH
        path_success = set_system_path(espeak_path)
        
        # Set up taproot compatibility
        taproot_success = setup_taproot_compatibility(espeak_path)
        
        if path_success and taproot_success:
            # Verify installation
            if verify_installation():
                print_status("\nStep 1 completed successfully!", "success")
                print_status("Please follow these steps:", "info")
                print_status("1. Close this terminal", "info")
                print_status("2. Open a new terminal", "info")
                print_status("3. Run step2-install-packages.py", "info")
            else:
                print_status("Installation verification failed", "error")
        else:
            print_status("Step 1 completed with warnings. Please check the messages above.", "warning")
            
    except Exception as e:
        print_status(f"Installation failed: {str(e)}", "error")
        sys.exit(1)

if __name__ == "__main__":
    main()