"""
Utility functions for the OARC dependency upgrade process.

This module provides helper functions for checking, fixing, and upgrading
project dependencies, including platform-specific handling of packages.
"""

import subprocess
import re
import sys
import shutil
import json
import tomli
import tomli_w
import os
from pathlib import Path
from typing import Dict, Tuple, List
from oarc.utils.log import log

def ensure_package_installed(package_name: str) -> bool:
    """
    Ensure a package is installed in the current environment.
    
    Args:
        package_name: Name of the package to install if needed
        
    Returns:
        bool: True if package is available, False if installation failed
    """
    if package_name == "uv" and shutil.which("uv"):
        log.info("UV package manager found.")
        return True
    
    log.info(f"{package_name} not found. Installing...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        log.info(f"{package_name} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to install {package_name}: {e}")
        return False

def fix_toml_file(pyproject_path: str) -> bool:
    """
    Fix common syntax issues in pyproject.toml file.
    
    Args:
        pyproject_path: Path to the pyproject.toml file
        
    Returns:
        bool: True if file is valid or was fixed successfully
    """
    try:
        # Convert to Path object and check existence
        path = Path(pyproject_path)
        if not path.exists():
            log.error(f"TOML file not found at: {path}")
            return False
        
        log.info(f"Found pyproject.toml at: {path}")
        
        # First try to parse with tomli to see if it's valid
        try:
            with open(path, "rb") as f:
                tomli.load(f)
            log.info("TOML file is valid. No fixes needed.")
            return True
        except tomli.TOMLDecodeError as e:
            log.warning(f"TOML syntax error detected: {e}")
            
        # If parsing failed, apply manual fixes
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Apply a series of common fixes
        # 1. Fix double commas
        content = re.sub(r'",\s*,', '",', content)
        
        # 2. Fix missing commas between dependency lines
        lines = content.split('\n')
        in_deps_section = False
        for i in range(len(lines) - 1):
            if "dependencies = [" in lines[i]:
                in_deps_section = True
                continue
            
            if in_deps_section:
                if "]" in lines[i]:
                    in_deps_section = False
                    continue
                    
                current_line = lines[i].strip()
                next_line = lines[i+1].strip()
                
                # If current line is a dependency and next line is also a dependency
                if (current_line.startswith('"') and 
                    next_line.startswith('"') and 
                    not current_line.endswith(',')):
                    
                    # Handle lines with comments
                    if '#' in current_line:
                        comment_pos = current_line.find('#')
                        lines[i] = current_line[:comment_pos] + ',' + current_line[comment_pos:]
                    else:
                        lines[i] = current_line + ','
        
        content = '\n'.join(lines)
        
        # Write back the fixed content
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            
        # Verify if our fixes worked
        try:
            with open(path, "rb") as f:
                tomli.load(f)
            log.info("TOML file fixed successfully.")
            return True
        except tomli.TOMLDecodeError as e:
            log.error(f"TOML file still has syntax errors after fixes: {e}")
            return False
            
    except Exception as e:
        log.error(f"Error fixing TOML file: {e}")
        return False

def get_outdated_packages(use_uv: bool = True) -> Dict[str, Tuple[str, str]]:
    """
    Get list of outdated packages.
    
    Args:
        use_uv: Whether to use UV package manager instead of pip
    
    Returns:
        Dict mapping package names to (current_version, latest_version) tuples
    """
    log.info("Checking for outdated packages...")
    outdated_packages = {}
    
    try:
        if use_uv:
            result = subprocess.run(
                ["uv", "pip", "list", "--outdated"],
                capture_output=True,
                text=True
            )
            # Parse UV output
            lines = result.stdout.strip().split('\n')
            if len(lines) > 2:  # Skip header lines
                for line in lines[2:]:  # Skip header lines
                    parts = line.split()
                    if len(parts) >= 3:
                        package = parts[0].strip()
                        current = parts[1].strip()
                        latest = parts[2].strip()
                        outdated_packages[package.lower()] = (current, latest)
        else:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True
            )
            try:
                packages_json = json.loads(result.stdout)
                for pkg in packages_json:
                    package = pkg.get('name', '').lower()
                    current = pkg.get('version', '')
                    latest = pkg.get('latest_version', '')
                    outdated_packages[package] = (current, latest)
            except json.JSONDecodeError:
                log.error("Could not parse pip JSON output")
        
        # Log the outdated packages
        if outdated_packages:
            log.info(f"Found {len(outdated_packages)} outdated packages:")
            for pkg, (current, latest) in outdated_packages.items():
                log.info(f"  {pkg}: {current} -> {latest}")
        else:
            log.info("No outdated packages found.")
            
        return outdated_packages
    except subprocess.SubprocessError as e:
        log.error(f"Error checking outdated packages: {e}")
        return {}

def upgrade_project_dependencies(use_uv: bool = True) -> bool:
    """
    Upgrade all project dependencies.
    
    Args:
        use_uv: Whether to use UV package manager instead of pip
        
    Returns:
        bool: True if upgrade was successful
    """
    log.info("Upgrading dependencies...")
    try:
        # Get the project directory (where pyproject.toml is located)
        project_root = None
        
        # Try to find the project root
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").exists():
            project_root = cwd
        
        # If not found in current directory, try a few levels up
        if not project_root:
            current_dir = Path(__file__).resolve().parent
            for _ in range(5):  # Try up to 5 levels up
                if (current_dir / "pyproject.toml").exists():
                    project_root = current_dir
                    break
                parent = current_dir.parent
                if parent == current_dir:  # Reached filesystem root
                    break
                current_dir = parent
        
        if not project_root:
            # Last resort, try the repo root (3 levels up from utils/setup)
            project_root = Path(__file__).resolve().parents[3]
        
        log.info(f"Using project directory: {project_root}")
        
        # Change to project directory for installation
        old_cwd = os.getcwd()
        os.chdir(str(project_root))
        
        try:
            cmd = ["uv", "pip", "install", "--upgrade", "-e", "."] if use_uv else [sys.executable, "-m", "pip", "install", "--upgrade", "-e", "."]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Check for specific error about oarc.exe being in use
                if "oarc.exe" in result.stderr and "process cannot access the file" in result.stderr:
                    log.error("Cannot upgrade while oarc.exe is in use. Please try one of the following:")
                    log.error("1. Exit any running OARC processes and try again")
                    log.error("2. Run the upgrade from a regular command prompt instead of through the oarc command")
                    log.error("3. Temporarily rename the oarc.exe file, upgrade, then rename it back")
                    return False
                else:
                    log.error(f"Upgrade failed with error: {result.stderr}")
                    return False
            
            log.info("Dependencies upgraded successfully!")
            return True
        finally:
            # Always restore original working directory
            os.chdir(old_cwd)
            
    except subprocess.CalledProcessError as e:
        log.error(f"Error upgrading dependencies: {e}")
        if e.stderr and "oarc.exe" in e.stderr.decode():
            log.error("Cannot upgrade while oarc.exe is in use. Try running upgrade from a regular command prompt.")
        return False
    except Exception as e:
        log.error(f"Unexpected error during dependency upgrade: {e}")
        return False

def get_current_package_versions() -> Dict[str, str]:
    """
    Get current versions of all installed packages.
    
    Returns:
        Dict mapping package names to their versions
    """
    log.info("Getting current package versions...")
    current_versions = {}
    
    try:
        pip_list = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        packages_json = json.loads(pip_list.stdout)
        for pkg in packages_json:
            current_versions[pkg['name'].lower()] = pkg['version']
        
        log.info(f"Found {len(current_versions)} installed packages")
        return current_versions
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        log.error(f"Error getting current package versions: {e}")
        return {}

def verify_upgrades(before_packages: Dict[str, Tuple[str, str]]) -> bool:
    """
    Verify that packages were actually upgraded.
    
    Args:
        before_packages: Dict of outdated packages before upgrade
        
    Returns:
        True if all upgrades were successful, False otherwise
    """
    log.info("Verifying package upgrades...")
    
    # Get current versions of previously outdated packages
    current_versions = get_current_package_versions()
    
    # Check if packages were upgraded
    upgrade_issues = []
    for package, (old_version, target_version) in before_packages.items():
        if package in current_versions:
            current = current_versions[package]
            if current == old_version:
                upgrade_issues.append(f"{package} was not upgraded: still at {current}")
                log.warning(f"{package} was not upgraded: still at {current}")
            elif current != target_version:
                log.info(f"{package} was upgraded from {old_version} to {current} (target was {target_version})")
            else:
                log.info(f"{package} was successfully upgraded to {current}")
    
    if upgrade_issues:
        log.warning(f"Found {len(upgrade_issues)} upgrade issues:")
        for issue in upgrade_issues:
            log.warning(f"  - {issue}")
        return False
    
    log.info("All packages were successfully upgraded!")
    return True

def update_pyproject_toml(pyproject_path: str) -> bool:
    """
    Update versions in pyproject.toml based on installed packages.
    
    Args:
        pyproject_path: Path to the pyproject.toml file
        
    Returns:
        bool: True if update was successful
    """
    log.info("Updating pyproject.toml with latest versions...")
    
    try:
        # Ensure the path exists
        path = Path(pyproject_path)
        if not path.exists():
            log.error(f"pyproject.toml not found at: {path}")
            return False
            
        # Read the current pyproject.toml
        with open(path, "rb") as f:
            pyproject_data = tomli.load(f)
        
        # Get current versions directly from pip
        deps_dict = get_current_package_versions()
        
        # Update dependency versions in pyproject.toml
        if "project" in pyproject_data and "dependencies" in pyproject_data["project"]:
            dependencies = pyproject_data["project"]["dependencies"]
            
            for i, dep in enumerate(dependencies):
                # Handle dependencies in string format
                if isinstance(dep, str):
                    # Parse package name from dependency string
                    match = re.match(r'^([a-zA-Z0-9_-]+)(?:\[.*\])?', dep)
                    if match:
                        pkg_name = match.group(1).lower()
                        if pkg_name in deps_dict:
                            # Keep any markers or extras in the dependency string
                            extras_match = re.search(r'(\[.*?\])', dep)
                            extras = extras_match.group(1) if extras_match else ""
                            
                            marker_match = re.search(r';\s*(.*?)$', dep)
                            marker = f"; {marker_match.group(1)}" if marker_match else ""
                            
                            # Update with new version
                            new_dep = f"{pkg_name}{extras}>={deps_dict[pkg_name]}{marker}"
                            dependencies[i] = new_dep
            
            # Write back updated pyproject.toml
            with open(path, "wb") as f:
                tomli_w.dump(pyproject_data, f)
            
            log.info("pyproject.toml updated successfully!")
            return True
        else:
            log.warning("Could not find dependencies in pyproject.toml")
            return False
            
    except Exception as e:
        log.error(f"Error updating pyproject.toml: {e}")
        return False

def fix_platform_specific_dependencies(pyproject_path: str) -> bool:
    """
    Handle platform-specific dependency issues.
    
    Args:
        pyproject_path: Path to the pyproject.toml file
        
    Returns:
        bool: True if fixes were applied or no fixes were needed
    """
    log.info("Checking for platform-specific dependency issues...")
    
    platform = sys.platform
    
    try:
        # Read the current pyproject.toml
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomli.load(f)
        
        if "project" in pyproject_data and "dependencies" in pyproject_data["project"]:
            dependencies = pyproject_data["project"]["dependencies"]
            modified = False
            
            # Handle Windows-specific issues
            if platform.startswith('win'):
                for i, dep in enumerate(dependencies):
                    if isinstance(dep, str):
                        # Fix PyQt5-QT5 issues on Windows
                        if 'pyqt5-qt5' in dep.lower():
                            log.warning("Detected PyQt5-QT5 which is incompatible with Windows.")
                            # Replace with a platform-specific marker to exclude Windows
                            if '; ' in dep:
                                # There's already a marker, enhance it
                                dependencies[i] = dep.replace('; ', '; sys_platform != "win32" and ')
                            else:
                                # Add a new marker
                                dependencies[i] = f"{dep}; sys_platform != \"win32\""
                            
                            # Add PyQt5 main package if it's not already there
                            if not any('pyqt5' in d.lower() and not 'pyqt5-qt5' in d.lower() for d in dependencies):
                                dependencies.append('"PyQt5>=5.15.0"')
                            
                            modified = True
                            log.info("Modified PyQt5-QT5 dependency to be excluded on Windows.")
            
            if modified:
                # Write back updated pyproject.toml
                with open(pyproject_path, "wb") as f:
                    tomli_w.dump(pyproject_data, f)
                log.info("Updated pyproject.toml with platform-specific fixes.")
                return True
            else:
                log.info("No platform-specific fixes needed.")
                return True
    except Exception as e:
        log.error(f"Error fixing platform-specific dependencies: {e}")
        return False

def handle_problematic_dependencies() -> bool:
    """
    Handle packages known to cause issues during installation.
    
    Returns:
        bool: True if handling was successful
    """
    log.info("Checking for problematic dependencies...")
    
    # List of packages that might need special handling
    problematic_packages = {
        # Package name: (handling_method, message)
        "pyqt5-qt5": ("skip", "Known to cause issues on Windows."),
        "tensorflow-intel": ("manual", "May need separate installation steps.")
    }
    
    platform = sys.platform
    
    for pkg_name, (action, message) in problematic_packages.items():
        # Check if the problematic package is installed
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:  # Package is installed
                if platform.startswith('win') and pkg_name == "pyqt5-qt5":
                    log.warning(f"Detected {pkg_name} which is {message}")
                    # For PyQt5-Qt5 on Windows, we try to install regular PyQt5
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", "PyQt5"],
                        check=False  # Don't fail if this doesn't work
                    )
                elif action == "manual":
                    log.warning(f"Package {pkg_name} {message}")
        except subprocess.SubprocessError:
            pass  # Ignore errors
    
    return True

def check_invalid_packages(pyproject_path: str) -> bool:
    """
    Check for packages that don't exist in PyPI or have incorrect names.
    
    Args:
        pyproject_path: Path to the pyproject.toml file
        
    Returns:
        bool: True if corrections were made or no corrections were needed
    """
    log.info("Checking for invalid package names...")
    
    # Known package name corrections
    package_corrections = {
        "pycpuinfo": "py-cpuinfo",
        # Add more corrections here as needed
    }
    
    try:
        # Read the current pyproject.toml
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomli.load(f)
        
        if "project" in pyproject_data and "dependencies" in pyproject_data["project"]:
            dependencies = pyproject_data["project"]["dependencies"]
            modified = False
            
            for i, dep in enumerate(dependencies):
                if isinstance(dep, str):
                    # Extract package name
                    match = re.match(r'^"?([a-zA-Z0-9_-]+)', dep)
                    if match:
                        pkg_name = match.group(1).lower()
                        
                        # Check if this package needs correction
                        if pkg_name in package_corrections:
                            correct_name = package_corrections[pkg_name]
                            log.warning(f"Found invalid package name: {pkg_name} -> should be {correct_name}")
                            
                            # Replace the package name while keeping version and markers
                            new_dep = dep.replace(pkg_name, correct_name, 1)
                            dependencies[i] = new_dep
                            modified = True
                            log.info(f"Corrected package name from {pkg_name} to {correct_name}")
            
            if modified:
                # Write back updated pyproject.toml
                with open(pyproject_path, "wb") as f:
                    tomli_w.dump(pyproject_data, f)
                log.info("Updated pyproject.toml with package name corrections.")
                return True
            else:
                log.info("No package name corrections needed.")
                return True
    except Exception as e:
        log.error(f"Error checking invalid packages: {e}")
        return False
