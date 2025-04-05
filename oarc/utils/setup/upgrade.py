#!/usr/bin/env python3
"""
OARC Upgrade Module

This module provides functionality to upgrade and maintain the OARC project dependencies.
It includes tools to check for outdated packages, fix dependency issues in pyproject.toml,
handle platform-specific dependencies, and update package versions.

When run as a script, it performs a complete dependency upgrade process.
It can also be imported and used programmatically through the main() function.
"""

import sys
from pathlib import Path
import importlib.util
import subprocess
import re
import os

# Ensure required dependencies are installed first
def ensure_initial_dependencies():
    """Ensure critical dependencies needed for upgrading are installed."""
    for package_name in ["tomli", "tomli_w"]:
        if importlib.util.find_spec(package_name) is None:
            print(f"Installing required package {package_name}...")
            import subprocess
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_name],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError:
                print(f"Failed to install {package_name}. Exiting.")
                sys.exit(1)

# Call this before any other imports that might depend on these packages
ensure_initial_dependencies()

# Now we can safely import other modules
from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.setup.setup import main as setup_main
from oarc.utils.setup.upgrade_utils import (
    ensure_package_installed,
    fix_toml_file,
    get_outdated_packages,
    upgrade_project_dependencies,
    verify_upgrades,
    update_pyproject_toml,
    fix_platform_specific_dependencies,
    handle_problematic_dependencies,
    check_invalid_packages
)

def handle_dependency_conflicts():
    """Handle known dependency conflicts by pinning specific versions."""
    log.info("Handling dependency conflicts...")
    conflicts = [
        # Format: (package_name, constraint, reason)
        ("numpy", "<2.0.0,>=1.26.0", "Required by tensorflow-intel 2.15.1"),
        ("tensorboard", "<2.16,>=2.15", "Required by tensorflow-intel 2.15.1")
    ]
    
    for package, constraint, reason in conflicts:
        log.info(f"Pinning {package} to {constraint} ({reason})")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"{package}{constraint}", "--force-reinstall"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to pin {package}: {e}")
            log.debug(f"Stderr: {e.stderr.decode() if e.stderr else 'No error output'}")
            return False
    return True

def check_dependency_conflicts():
    """Check for dependency conflicts by running pip check."""
    log.info("Checking for dependency conflicts...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,  # Don't raise exception on non-zero exit
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            log.warning("Dependency conflicts detected:")
            for line in result.stdout.splitlines():
                log.warning(f"  {line}")
            return False
        return True
    except Exception as e:
        log.error(f"Error checking dependencies: {e}")
        return False

def main():
    """
    Run the complete dependency setup and upgrade process.
    
    This function first runs the setup process to ensure all required
    dependencies are installed, then proceeds with upgrading all packages
    to their latest versions.
    
    Returns:
        bool: True if the setup and upgrade were successful, False otherwise
    """
    try:
        # First run the setup process to ensure all dependencies are installed
        log.info("Starting OARC setup process...")
        setup_success = setup_main()
        
        if not setup_success:
            log.error("Setup process failed. Cannot proceed with upgrade.")
            return False
        
        log.info("Setup completed successfully. Proceeding with upgrade...")
        
        # Get project root path from Paths API
        paths = Paths()
        project_root = Path(paths._paths['base']['project_root'])
        pyproject_path = project_root / "pyproject.toml"
        
        log.info(f"Starting upgrade process using {pyproject_path}")
        
        # Check if pyproject.toml exists
        if not pyproject_path.exists():
            log.error(f"pyproject.toml not found at {pyproject_path}")
            return False
        
        # Ensure UV package manager is installed
        use_uv = ensure_package_installed("uv")
        
        # Fix any syntax issues in the TOML file
        if not fix_toml_file(str(pyproject_path)):
            log.error("Could not fix TOML file. Exiting.")
            return False
        
        # Check for and correct invalid package names
        check_invalid_packages(str(pyproject_path))
        
        # Apply platform-specific dependency fixes
        fix_platform_specific_dependencies(str(pyproject_path))
        
        # Get a list of outdated packages before upgrading
        outdated_before = get_outdated_packages(use_uv)
        
        # Handle known problematic dependencies
        handle_problematic_dependencies()
        
        # Upgrade all project dependencies
        upgrade_success = upgrade_project_dependencies(use_uv)
        if not upgrade_success:
            log.error("Failed to upgrade dependencies.")
            return False
        
        # Handle specific dependency conflicts
        if not handle_dependency_conflicts():
            log.warning("Some dependency conflicts could not be resolved automatically.")
        
        # Verify the upgrades were successful
        verify_result = verify_upgrades(outdated_before)
        
        # Update the pyproject.toml file with current versions
        update_pyproject_toml(str(pyproject_path))
        
        # Apply platform-specific fixes again after updates
        fix_platform_specific_dependencies(str(pyproject_path))
        
        # Check for any remaining dependency conflicts
        conflicts_check = check_dependency_conflicts()
        if not conflicts_check:
            log.warning("There are still dependency conflicts. Manual intervention may be required.")
        
        log.info("Package upgrade process completed!")
        return verify_result and conflicts_check
        
    except Exception as e:
        log.error(f"Unexpected error during upgrade process: {str(e)}")
        return False

# Make the module directly runnable
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)