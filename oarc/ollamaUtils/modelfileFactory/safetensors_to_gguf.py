"""
Python script to convert Hugging Face safetensors to GGUF format using llamacpp.
Replaces the original llamacpp_safetensors_to_GGUF.cmd batch file.
"""

import os
import subprocess
import argparse
from oarc.utils.log import log
from oarc.utils.paths import Paths

def convert_safetensors_to_gguf(model_dir, model_name, quantization="q8_0"):
    """
    Convert a Hugging Face model to GGUF format using llama.cpp conversion tools.
    
    Args:
        model_dir: Directory containing the model and llama.cpp tools
        model_name: Name of the HuggingFace model to convert
        quantization: Quantization level (default: q8_0)
    
    Returns:
        bool: True if conversion was successful, False otherwise
        str: Path to the converted model if successful, None otherwise
    """
    log.info(f"Converting {model_name} to GGUF format with {quantization} quantization")
    
    try:
        # Ensure the converted directory exists
        converted_dir = os.path.join(model_dir, "converted")
        os.makedirs(converted_dir, exist_ok=True)
        
        # Full path to the output file
        output_file = os.path.join(converted_dir, f"{model_name}-{quantization}.gguf")
        
        # Path to the conversion script
        convert_script = os.path.join(model_dir, "llama.cpp", "convert-hf-to-gguf.py")
        
        if not os.path.exists(convert_script):
            log.error(f"Conversion script not found at: {convert_script}")
            return False, None
        
        # Build the command
        cmd = [
            "python",
            convert_script,
            f"--outtype", quantization,
            f"--model-name", f"{model_name}-{quantization}",
            f"--outfile", output_file,
            model_name
        ]
        
        log.info(f"Running conversion command: {' '.join(cmd)}")
        
        # Execute the conversion
        result = subprocess.run(
            cmd,
            check=True,
            cwd=model_dir,  # Run from the model directory
            capture_output=True,
            text=True
        )
        
        log.info(f"Successfully converted model to: {output_file}")
        log.debug(f"Command output: {result.stdout}")
        
        return True, output_file
        
    except subprocess.CalledProcessError as e:
        log.error(f"Error converting model: {e}")
        log.error(f"Command output: {e.stderr}")
        return False, None
    except Exception as e:
        log.error(f"Unexpected error during conversion: {e}")
        return False, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face safetensors to GGUF format")
    parser.add_argument("model_dir", help="Directory containing the model and llama.cpp tools")
    parser.add_argument("model_name", help="Name of the HuggingFace model to convert")
    parser.add_argument("--quantization", default="q8_0", help="Quantization level (default: q8_0)")
    
    args = parser.parse_args()
    success, output_path = convert_safetensors_to_gguf(args.model_dir, args.model_name, args.quantization)
    
    if success:
        print(f"CONVERTED: {args.model_name} to {output_path}")
    
    exit(0 if success else 1)
