import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import asyncio


class ComfyGenerator:
    def __init__(self):
        """Initialize the ComfyGenerator class and set up the environment."""
        # Initialize sys paths and import necessary components
        self._setup_environment()
        self._import_custom_nodes()
        
        # Import node mappings
        from nodes import NODE_CLASS_MAPPINGS
        self.NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS

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
        # Add ComfyUI to path
        comfyui_path = self._find_path("ComfyUI")
        if comfyui_path is not None and os.path.isdir(comfyui_path):
            sys.path.append(comfyui_path)
            print(f"ComfyUI found at '{comfyui_path}' and added to sys.path")
        else:
            raise RuntimeError("ComfyUI directory not found. Please make sure it exists in a parent directory.")

        # Add extra model paths if available
        try:
            extra_model_paths = self._find_path("extra_model_paths.yaml")
            if extra_model_paths is not None:
                try:
                    from main import load_extra_path_config
                except ImportError:
                    from utils.extra_config import load_extra_path_config
                load_extra_path_config(extra_model_paths)
                print(f"Loaded extra model paths from {extra_model_paths}")
        except Exception as e:
            print(f"Note: Could not load extra_model_paths.yaml: {e}")

    def _import_custom_nodes(self):
        """Import custom nodes from ComfyUI."""
        try:
            import execution
            from nodes import init_extra_nodes
            import server

            # Creating a new event loop and setting it as the default loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Creating an instance of PromptServer with the loop
            server_instance = server.PromptServer(loop)
            execution.PromptQueue(server_instance)

            # Initializing custom nodes
            init_extra_nodes()
            print("Successfully imported custom nodes")
        except Exception as e:
            raise RuntimeError(f"Failed to import custom nodes: {e}")

    def generate_image(self, 
                       prompt: str = "neon, horrific armored ogre boss knight holding battle axe, ready to fight",
                       negative_prompt: str = "2D",
                       width: int = 568,
                       height: int = 320,
                       steps: int = 6,
                       cfg: float = 1.0,
                       seed: int = None,
                       sampler_name: str = "heun",
                       scheduler: str = "simple",
                       filename_prefix: str = "generated",
                       clip_name1: str = "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
                       clip_name2: str = "clip_l.safetensors",
                       unet_name: str = "flux1-dev-Q2_K.gguf",
                       vae_name: str = "ae.safetensors"):
        """
        Generate an image using ComfyUI with the specified parameters.
        
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
            clip_name1: First CLIP model filename
            clip_name2: Second CLIP model filename
            unet_name: UNet model filename
            vae_name: VAE model filename
            
        Returns:
            Path to the generated image
        """
        if seed is None:
            seed = random.randint(1, 2**64)
            
        with torch.inference_mode():
            # Create empty latent image
            emptylatentimage = self.NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
            emptylatentimage_out = emptylatentimage.generate(
                width=width, height=height, batch_size=1
            )

            # Load CLIP models
            dualcliploadergguf = self.NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
            dualcliploadergguf_out = dualcliploadergguf.load_clip(
                clip_name1=clip_name1,
                clip_name2=clip_name2,
                type="flux",
            )

            # Encode text prompts
            cliptextencode = self.NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cliptextencode_pos = cliptextencode.encode(
                text=prompt,
                clip=self._get_value_at_index(dualcliploadergguf_out, 0),
            )

            cliptextencode_neg = cliptextencode.encode(
                text=negative_prompt, 
                clip=self._get_value_at_index(dualcliploadergguf_out, 0)
            )

            # Load UNet model
            unetloadergguf = self.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unetloadergguf_out = unetloadergguf.load_unet(unet_name=unet_name)

            # Load VAE model
            vaeloader = self.NODE_CLASS_MAPPINGS["VAELoader"]()
            vaeloader_out = vaeloader.load_vae(vae_name=vae_name)

            # Setup sampling, decoding and saving nodes
            ksampler = self.NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = self.NODE_CLASS_MAPPINGS["VAEDecode"]()
            saveimage = self.NODE_CLASS_MAPPINGS["SaveImage"]()

            # Run sampling
            ksampler_out = ksampler.sample(
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=1,
                model=self._get_value_at_index(unetloadergguf_out, 0),
                positive=self._get_value_at_index(cliptextencode_pos, 0),
                negative=self._get_value_at_index(cliptextencode_neg, 0),
                latent_image=self._get_value_at_index(emptylatentimage_out, 0),
            )

            # Decode the image
            vaedecode_out = vaedecode.decode(
                samples=self._get_value_at_index(ksampler_out, 0),
                vae=self._get_value_at_index(vaeloader_out, 0),
            )

            # Save the image
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