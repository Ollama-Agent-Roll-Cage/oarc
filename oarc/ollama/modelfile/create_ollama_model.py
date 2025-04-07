"""
Python script to create an Ollama model from a modelfile.
Replaces the original create_ollama_model_from_modelfile.cmd batch file.
"""

import os
import subprocess
import argparse

from oarc.utils.log import log
from oarc.utils.paths import Paths

def create_ollama_model(model_name, modelfile_path=None):
    """
    Create an Ollama model using a modelfile.
    
    Args:
        model_name: Name of the model to create
        modelfile_path: Path to the modelfile. If None, uses './ModelFile' in the model directory
    
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    log.info(f"Creating Ollama model: {model_name}")
    
    try:
        # Get paths from the singleton
        paths = Paths.get_instance()
        ignored_agents_dir = os.path.join(paths.get_model_dir(), "AgentFiles", "Ignored_Agents")
        
        if modelfile_path is None:
            modelfile_path = os.path.join(ignored_agents_dir, model_name, "ModelFile")
        
        # Check if the modelfile exists
        if not os.path.exists(modelfile_path):
            log.error(f"Modelfile not found at: {modelfile_path}")
            return False
            
        # Execute the ollama create command
        log.info(f"Running ollama create {model_name} with modelfile at {modelfile_path}")
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            check=True,
            capture_output=True,
            text=True
        )
        
        log.info(f"{model_name} agent has been successfully created!")
        log.debug(f"Command output: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        log.error(f"Error creating Ollama model: {e}")
        log.error(f"Command output: {e.stderr}")
        return False
    except Exception as e:
        log.error(f"Unexpected error creating Ollama model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an Ollama model from a modelfile")
    parser.add_argument("model_name", help="Name of the model to create")
    parser.add_argument("--modelfile", help="Path to the modelfile (default: ./ModelFile in the model directory)")
    
    args = parser.parse_args()
    success = create_ollama_model(args.model_name, args.modelfile)
    
    exit(0 if success else 1)
