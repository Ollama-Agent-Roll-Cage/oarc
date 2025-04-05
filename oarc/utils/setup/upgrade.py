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

def main():
    """
    Run the complete dependency setup and upgrade process.
    
    This function first runs the setup process to ensure all required
    dependencies are installed, then proceeds with upgrading all packages
    to their latest versions.
    
    Returns:
        bool: True if the setup and upgrade were successful, False otherwise
    """
    # First run the setup process to ensure all dependencies are installed
    log.info("Starting OARC setup process...")
    setup_success = setup_main()
    
    if not setup_success:
        log.error("Setup process failed. Cannot proceed with upgrade.")
        return False
    
    log.info("Setup completed successfully. Proceeding with upgrade...")
    
    # Get the default pyproject.toml path or from command line
    pyproject_path = "pyproject.toml"
    if len(sys.argv) > 1:
        pyproject_path = sys.argv[1]
    
    log.info(f"Starting upgrade process using {pyproject_path}")
    
    # Ensure UV package manager is installed
    use_uv = ensure_package_installed("uv")
    
    # Fix any syntax issues in the TOML file
    if not fix_toml_file(pyproject_path):
        log.error("Could not fix TOML file. Exiting.")
        return False
    
    # Check for and correct invalid package names
    check_invalid_packages(pyproject_path)
    
    # Apply platform-specific dependency fixes
    fix_platform_specific_dependencies(pyproject_path)
    
    # Get a list of outdated packages before upgrading
    outdated_before = get_outdated_packages(use_uv)
    
    # Handle known problematic dependencies
    handle_problematic_dependencies()
    
    # Upgrade all project dependencies
    if not upgrade_project_dependencies(use_uv):
        log.error("Failed to upgrade dependencies.")
        return False
    
    # Verify the upgrades were successful
    verify_result = verify_upgrades(outdated_before)
    
    # Update the pyproject.toml file with current versions
    update_pyproject_toml(pyproject_path)
    
    # Apply platform-specific fixes again after updates
    fix_platform_specific_dependencies(pyproject_path)
    
    log.info("Package upgrade process completed!")
    return verify_result

# Make the module directly runnable
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)