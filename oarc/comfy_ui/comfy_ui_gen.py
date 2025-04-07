"""
ComfyUI Generator Class for Image Generation with Flux Models
This class provides methods to set up the environment, import custom nodes,
and generate images using ComfyUI with Flux models."""

import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


class ComfyGenerator:
    def __init__(self):
        """Initialize the ComfyGenerator class and set up the environment."""
        # Initialize sys paths and import necessary components
        self._setup_environment()
        self._import_custom_nodes()
        
        # Import node mappings
        try:
            from nodes import NODE_CLASS_MAPPINGS
            self.NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
            print(f"Found {len(NODE_CLASS_MAPPINGS)} nodes in the node mappings")
        except Exception as e:
            print(f"Error loading NODE_CLASS_MAPPINGS: {e}")
            raise RuntimeError(f"Failed to load node mappings: {e}")

    def _get_value_at_index(self, obj: Union[Sequence, Mapping], index: int) -> Any:
        """Returns the value at the given index of a sequence or mapping."""
        try:
            return obj[index]
        except KeyError:
            return obj["result"][index]

    def _find_path(self, name: str, path: str = None) -> str:
        """Recursively looks for a file/directory in parent folders."""
        # If no path is given, use the current working directory
        if path is None:
            path = os.getcwd()

        # Check if the current directory contains the name
        if name in os.listdir(path):
            path_name = os.path.join(path, name)
            return path_name

        # Get the parent directory
        parent_directory = os.path.dirname(path)

        # If the parent directory is the same as the current directory, we've reached the root
        if parent_directory == path:
            return None

        # Recursively call the function with the parent directory
        return self._find_path(name, parent_directory)

    def _setup_environment(self):
        """Set up the environment by adding ComfyUI paths."""
        # Get ComfyUI path from environment variable
        comfyui_path = os.environ.get("COMFYUI_PATH")
        
        if not comfyui_path:
            # Fallback to recursive search if environment variable is not set
            comfyui_path = self._find_path("ComfyUI")
        
        if comfyui_path is not None and os.path.isdir(comfyui_path):
            # Add ComfyUI directory to the system path first
            sys.path.append(comfyui_path)
            
            # Add ComfyUI's 'comfy' subdirectory to path if it exists
            comfy_dir = os.path.join(comfyui_path, "comfy")
            if os.path.isdir(comfy_dir):
                sys.path.append(comfy_dir)
                
            # Add ComfyUI's 'web' subdirectory to path if it exists
            web_dir = os.path.join(comfyui_path, "web")
            if os.path.isdir(web_dir):
                sys.path.append(web_dir)
                
            print(f"ComfyUI found at '{comfyui_path}' and added to sys.path")
        else:
            raise RuntimeError("ComfyUI directory not found. Please set the COMFYUI_PATH environment variable or make sure ComfyUI exists in a parent directory.")

        # Add extra model paths if available
        try:
            extra_model_paths = os.environ.get("EXTRA_MODEL_PATHS")
            if not extra_model_paths:
                extra_model_paths = self._find_path("extra_model_paths.yaml")
                
            if extra_model_paths is not None:
                try:
                    # We need to import this after setting up the path
                    from comfy.cli_args import load_extra_path_config
                except ImportError:
                    try:
                        from main import load_extra_path_config
                    except ImportError:
                        try:
                            from utils.extra_config import load_extra_path_config
                        except ImportError:
                            print("Warning: Could not import load_extra_path_config")
                load_extra_path_config(extra_model_paths)
                print(f"Loaded extra model paths from {extra_model_paths}")
        except Exception as e:
            print(f"Note: Could not load extra_model_paths.yaml: {e}")

        # Add this to your _setup_environment method
        custom_nodes_dir = os.path.join(comfyui_path, "custom_nodes")
        if os.path.isdir(custom_nodes_dir):
            sys.path.append(custom_nodes_dir)
            print(f"Added custom_nodes directory: {custom_nodes_dir}")

    def _import_custom_nodes(self):
        """Import custom nodes from ComfyUI without using server components."""
        try:
            # Make sure we can get the basic node mappings first
            import folder_paths
            from nodes import NODE_CLASS_MAPPINGS, load_custom_node, get_module_name
            
            print("Note: Server components not imported - not needed for generation only")
            
            # Import available objects from nodes module to see what we're working with
            from nodes import NODE_CLASS_MAPPINGS, load_custom_node, get_module_name
            print(f"Available objects in nodes: {', '.join(dir())}")
            
            # Manually import custom nodes without server
            custom_nodes_path = os.path.join(os.environ.get("COMFYUI_PATH", self._find_path("ComfyUI")), "custom_nodes")
            if os.path.exists(custom_nodes_path):
                if hasattr(sys.modules["nodes"], "init_custom_nodes"):
                    print("Initializing custom nodes using built-in function")
                    sys.modules["nodes"].init_custom_nodes()
                else:
                    print("No initialization function found - continuing with already loaded nodes")
                
            # Check if nodes are loaded successfully
            if NODE_CLASS_MAPPINGS:
                print(f"Successfully loaded {len(NODE_CLASS_MAPPINGS)} node types")
                # Check for Flux nodes specifically
                flux_nodes = ["DualCLIPLoaderGGUF", "UnetLoaderGGUF"]
                found_flux_nodes = [node for node in flux_nodes if node in NODE_CLASS_MAPPINGS]
                if found_flux_nodes:
                    print(f"Found Flux nodes: {found_flux_nodes}")
                else:
                    print("Warning: Flux nodes not found in NODE_CLASS_MAPPINGS")
                return True
            else:
                raise RuntimeError("NODE_CLASS_MAPPINGS is empty - no nodes were loaded")
                    
        except ImportError as e:
            print(f"Import error: {e}")
            print(f"Current sys.path: {sys.path}")
            raise RuntimeError(f"Failed to import ComfyUI modules. Make sure COMFYUI_PATH is set correctly: {e}")
        except Exception as e:
            print(f"Unexpected error during node import: {str(e)}")
            raise RuntimeError(f"Failed to import custom nodes: {e}")

    def check_flux_installation(self):
        """Check if Flux nodes are installed correctly."""
        comfyui_path = os.environ.get("COMFYUI_PATH", self._find_path("ComfyUI"))
        if not comfyui_path:
            return False
            
        # Check for Flux custom nodes
        flux_path = os.path.join(comfyui_path, "custom_nodes", "ComfyUI-Flux")
        flux_exists = os.path.exists(flux_path)
        
        if flux_exists:
            print(f"Flux installation found at: {flux_path}")
            
            # Check for model files
            models_path = os.path.join(comfyui_path, "models", "flux")
            if os.path.exists(models_path):
                files = os.listdir(models_path)
                print(f"Flux models found: {files}")
                if "flux1-dev-Q2_K.gguf" in files and "t5-v1_1-xxl-encoder-Q3_K_S.gguf" in files:
                    print("Required Flux model files found!")
                    return True
                else:
                    print("Warning: Required Flux model files not found in models/flux directory")
            else:
                print("Warning: Flux models directory not found")
        else:
            print("Flux installation not found. Please install ComfyUI-Flux in your custom_nodes directory")
            
        return False

    def generate_image(self, 
                       prompt: str = "neon, horrific armored ogre boss knight holding battle axe, ready to fight",
                       negative_prompt: str = "2D",
                       width: int = 568,
                       height: int = 320,
                       steps: int = 30,
                       cfg: float = 1.0,
                       seed: int = None,
                       sampler_name: str = "heun",  # Changed to match fastFlux
                       scheduler: str = "simple",    # Changed to match fastFlux
                       filename_prefix: str = "generated"):
        """
        Generate an image using ComfyUI with Flux models.
        
        Args:
            prompt: The text prompt for image generation
            negative_prompt: Negative prompt to avoid certain aspects in generation
            width: Width of the generated image
            height: Height of the generated image
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            seed: Random seed (if None, a random seed will be used)
            sampler_name: Name of the sampler to use
            scheduler: Name of the scheduler to use
            filename_prefix: Prefix for the saved image filename
            
        Returns:
            Path to the generated image
        """
        import folder_paths
        
        if seed is None:
            seed = random.randint(1, 2**64)
        
        print(f"Available nodes: {list(self.NODE_CLASS_MAPPINGS.keys())}")
            
        with torch.inference_mode():
            # Create empty latent image
            emptylatentimage = self.NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
            emptylatentimage_out = emptylatentimage.generate(
                width=width, height=height, batch_size=1
            )

            # Check if Flux nodes are available
            if "DualCLIPLoaderGGUF" in self.NODE_CLASS_MAPPINGS and "UnetLoaderGGUF" in self.NODE_CLASS_MAPPINGS:
                # Using Flux models (GGUF format)
                dualcliploadergguf = self.NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
                dualcliploadergguf_out = dualcliploadergguf.load_clip(
                    clip_name1="t5-v1_1-xxl-encoder-Q3_K_S.gguf",
                    clip_name2="clip_l.safetensors",
                    type="flux",
                )
                
                # Encode text prompts with Flux
                cliptextencode = self.NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
                cliptextencode_pos = cliptextencode.encode(
                    text=prompt,
                    clip=self._get_value_at_index(dualcliploadergguf_out, 0),
                )

                cliptextencode_neg = cliptextencode.encode(
                    text=negative_prompt, 
                    clip=self._get_value_at_index(dualcliploadergguf_out, 0)
                )
                
                # Load Flux UNet
                unetloadergguf = self.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
                unetloadergguf_out = unetloadergguf.load_unet(unet_name="flux1-dev-Q2_K.gguf")
                
                # Load VAE
                vaeloader = self.NODE_CLASS_MAPPINGS["VAELoader"]()
                vaeloader_out = vaeloader.load_vae(vae_name="ae.safetensors")
                
                # Setup sampler and decoder
                ksampler = self.NODE_CLASS_MAPPINGS["KSampler"]()
                vaedecode = self.NODE_CLASS_MAPPINGS["VAEDecode"]()
                saveimage = self.NODE_CLASS_MAPPINGS["SaveImage"]()

                # Run sampling with Flux UNet
                ksampler_out = ksampler.sample(
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=1.0,
                    model=self._get_value_at_index(unetloadergguf_out, 0),
                    positive=self._get_value_at_index(cliptextencode_pos, 0),
                    negative=self._get_value_at_index(cliptextencode_neg, 0),
                    latent_image=self._get_value_at_index(emptylatentimage_out, 0),
                )

                # Decode the image with the loaded VAE
                vaedecode_out = vaedecode.decode(
                    samples=self._get_value_at_index(ksampler_out, 0),
                    vae=self._get_value_at_index(vaeloader_out, 0),
                )

            else:
                # Fallback to standard checkpoint loading if Flux nodes aren't available
                print("Flux nodes not available, falling back to standard checkpoint loading")
                checkpointloader = self.NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
                model_filenames = folder_paths.get_filename_list('checkpoints')
                preferred_models = ["sd_xl_base_1.0.safetensors", "v1-5-pruned.safetensors", "sd3.5_medium.safetensors"]
                model_name = None
                for preferred in preferred_models:
                    if preferred in model_filenames:
                        model_name = preferred
                        break
                if not model_name:
                    model_name = model_filenames[0] if model_filenames else "model.ckpt"
                print(f"Selected model: {model_name}")
                
                checkpointloader_out = checkpointloader.load_checkpoint(
                    ckpt_name=model_name
                )

                # Encode text prompts
                cliptextencode = self.NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
                cliptextencode_pos = cliptextencode.encode(
                    text=prompt,
                    clip=self._get_value_at_index(checkpointloader_out, 1),
                )

                cliptextencode_neg = cliptextencode.encode(
                    text=negative_prompt, 
                    clip=self._get_value_at_index(checkpointloader_out, 1)
                )

                # Setup sampling
                ksampler = self.NODE_CLASS_MAPPINGS["KSampler"]()
                ksampler_out = ksampler.sample(
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=1.0,
                    model=self._get_value_at_index(checkpointloader_out, 0),
                    positive=self._get_value_at_index(cliptextencode_pos, 0),
                    negative=self._get_value_at_index(cliptextencode_neg, 0),
                    latent_image=self._get_value_at_index(emptylatentimage_out, 0),
                )

                # Decode the image
                vaedecode = self.NODE_CLASS_MAPPINGS["VAEDecode"]()
                vaedecode_out = vaedecode.decode(
                    samples=self._get_value_at_index(ksampler_out, 0),
                    vae=self._get_value_at_index(checkpointloader_out, 2),
                )

            # Save the image (same for both workflows)
            saveimage = self.NODE_CLASS_MAPPINGS["SaveImage"]()
            saveimage_out = saveimage.save_images(
                filename_prefix=filename_prefix,
                images=self._get_value_at_index(vaedecode_out, 0),
            )
            
            # Return the results
            return saveimage_out


# Simple usage example
if __name__ == "__main__":
    generator = ComfyGenerator()
    
    # Generate with default parameters
    generator.generate_image()
    
    # Generate with custom parameters
    generator.generate_image(
        prompt="cyberpunk cityscape, neon lights, rainy night, dramatic lighting",
        negative_prompt="blurry, low quality",
        width=768,
        height=512,
        steps=20,
        filename_prefix="cyberpunk_scene"
    )
    
# # Simple usage of the ComfyGenerator class

# from comfy_generator import ComfyGenerator

# # Create an instance of the generator
# generator = ComfyGenerator()

# # Example 1: Basic usage with just a prompt
# result = generator.generate_image(
#     prompt="beautiful mountain landscape with a lake at sunset",
#     filename_prefix="mountain_sunset"
# )
# print(f"Generated image saved at: {result}")

# # Example 2: Advanced usage with more parameters
# result = generator.generate_image(
#     prompt="cyberpunk cityscape, neon lights, rainy night, dramatic lighting",
#     negative_prompt="blurry, low quality, cartoon, sketch",
#     width=768,
#     height=512,
#     steps=20,
#     cfg=7.0,
#     seed=42,  # fixed seed for reproducibility
#     filename_prefix="cyberpunk_scene"
# )
# print(f"Generated image saved at: {result}")

# # Example 3: Batch generation with different prompts
# prompts = [
#     "fantasy castle on a floating island",
#     "futuristic spaceship interior with holographic displays",
#     "underwater ancient ruins with glowing crystals"
# ]

# for i, prompt in enumerate(prompts):
#     result = generator.generate_image(
#         prompt=prompt,
#         filename_prefix=f"batch_generation_{i}"
#     )
#     print(f"Generated image {i+1} saved at: {result}")