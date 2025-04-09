from diffusers import StableDiffusionXLPipeline
import torch
import os
import asyncio  # Add this import
from PIL import Image
import logging
import gc  # For garbage collection

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only log warnings and errors

class SDXLGenerator:
    """Stable Diffusion XL image generation module"""
    
    def __init__(self, model_path=None):
        # Default model path if not provided
        self.model_path = model_path or os.path.join("M:\\", "SDXL_MOD", "randommaxxArtMerge_v10.safetensors")
        self.pipe = None
        
    def load_model(self):
        """Load the SDXL model into memory with optimizations"""
        try:
            # Clear CUDA cache and run garbage collection first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            # Verify the file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
                
            logger.info(f"Loading SDXL model from: {self.model_path}")
            
            # Load the model with memory optimizations
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,  # Use half precision
                use_safetensors=True,
                variant="fp16"  # Explicitly request fp16 variant
            )
            
            # Enable memory efficient attention
            self.pipe.enable_attention_slicing(slice_size="auto")
            
            # Enable VAE slicing for memory efficiency
            self.pipe.enable_vae_slicing()
            
            # Move to GPU with specific options
            self.pipe.to("cuda")
            
            logger.info("SDXL model loaded successfully with memory optimizations!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading SDXL model: {e}")
            return False
            
    def unload_model(self):
        """Unload model to free up VRAM"""
        try:
            if self.pipe is not None:
                self.pipe = None
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("SDXL model unloaded to free memory")
            return True
        except Exception as e:
            logger.error(f"Error unloading SDXL model: {e}")
            return False
    
    def generate_image(self, prompt, negative_prompt=None, width=1024, height=1024, 
                      steps=28, guidance_scale=7.5, output_path=None, callback=None):
        """Generate an image with the specified parameters"""
        try:
            # Load model if not already loaded
            if self.pipe is None:
                if not self.load_model():
                    raise Exception("Failed to load SDXL model")
            
            # Set default negative prompt if not provided
            if negative_prompt is None:
                negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy"
                
            # Generate the image
            logger.info(f"Generating SDXL image with prompt: {prompt[:50]}...")
            
            # Clear cache before generation
            torch.cuda.empty_cache()
            gc.collect()
            
            # For large images, use tiling
            use_tiling = max(width, height) > 768
            
            # Define callback for tracking progress with reduced frequency
            def progress_callback(step, timestep, latents):
                if callback:
                    # Only report at 0%, 25%, 50%, 75%, and 100%
                    report_steps = [0, round(steps/4), round(steps/2), round(3*steps/4), steps-1]
                    if step in report_steps or step == steps-1:  # Report at start and end for sure
                        progress = int((step / steps) * 100)
                        asyncio.create_task(callback(f"Generation progress: {progress}% complete (Step {step+1}/{steps})"))
                return True
            
            # Enable XFormers memory efficient attention if width/height is large
            if use_tiling:
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Using xformers memory efficient attention")
                except:
                    logger.info("Xformers not available, using standard generation")
            
            # Generate image with reduced memory usage
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    callback=progress_callback if callback else None,
                    callback_steps=5  # Update every 5 steps
                )
            
            # Save the image if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                result.images[0].save(output_path)
                logger.info(f"Image saved to: {output_path}")
            
            # Clear cache after generation
            torch.cuda.empty_cache()
            gc.collect()
                
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Error generating SDXL image: {e}")
            # Try to free GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            raise
            
    def __del__(self):
        """Destructor to ensure memory is freed"""
        if hasattr(self, 'pipe') and self.pipe is not None:
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()