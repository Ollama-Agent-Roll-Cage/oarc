"""image-generator-class.py
    
    This module provides a class for generating images using an AI/ML API.

Author: @BorcherdingL
Date: 4/4/2024
"""

import os
import base64
import logging
import asyncio
from io import BytesIO
from typing import Optional, Dict, Any, List, Union

import requests
import aiohttp
from PIL import Image

class ImageGenerator:
    """
    A versatile image generation class that can be integrated with any AI/ML API.
    Designed to work with aimlapi.com and other providers with similar interfaces.
    Visit https://aimlapi.com/models/flux-1-1-pro-api for more details on api keys.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        api_url: str = "https://api.aimlapi.com/v1/images/generations",
        env_key_name: str = "AIML_API_KEY"
    ):
        """
        Initialize the image generator with API credentials.
        
        Args:
            api_key (str, optional): API key for the image generation service.
                If not provided, will look for it in environment variables.
            api_url (str, optional): Base URL for the API. 
                Defaults to "https://api.aimlapi.com/v1/images/generations".
            env_key_name (str, optional): Name of environment variable containing the API key.
                Defaults to "AIML_API_KEY".
        """
        self.api_key = api_key or os.environ.get(env_key_name)
        if not self.api_key:
            raise ValueError(f"API key not provided and not found in environment variable {env_key_name}")
        
        self.api_url = api_url
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up logging for the image generator."""
        logger = logging.getLogger('ImageGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate(
        self,
        prompt: str,
        model: str = "flux-pro/v1.1",
        size: Optional[str] = None,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        return_format: str = "json",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images based on the provided prompt (synchronous version).
        
        Args:
            prompt (str): The text prompt to generate images from.
            model (str, optional): The model to use for generation. Defaults to "flux-pro/v1.1".
            size (str, optional): Size of the generated image (e.g., "1024x1024").
            num_images (int, optional): Number of images to generate. Defaults to 1.
            negative_prompt (str, optional): What not to include in the generated image.
            return_format (str, optional): Format to return the results in. Options:
                - "json": Raw JSON response (default)
                - "image": PIL Image object(s)
                - "b64": Base64 encoded string(s)
                - "url": URL string(s)
            **kwargs: Additional provider-specific parameters.
        
        Returns:
            Dict[str, Any] or List[Image.Image] or List[str]: Generated images in the requested format.
        
        Raises:
            requests.RequestException: If the API request fails.
        """
        self.logger.info(f"Generating image with prompt: {prompt[:50]}...")
        
        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "model": model,
            "n": num_images
        }
        
        # Add optional parameters
        if size:
            payload["size"] = size
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
            
        # Add any additional kwargs to the payload
        payload.update(kwargs)
        
        # Make the API request
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload
        )
        
        # Handle response
        response.raise_for_status()
        result = response.json()
        
        # Process the result based on return_format
        if return_format == "json":
            return result
        elif return_format == "image":
            return self._process_images(result)
        elif return_format == "b64":
            return self._extract_b64(result)
        elif return_format == "url":
            return self._extract_urls(result)
        else:
            raise ValueError(f"Invalid return_format: {return_format}")
    
    async def generate_async(
        self,
        prompt: str,
        model: str = "flux-pro/v1.1",
        size: Optional[str] = None,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        return_format: str = "json",
        **kwargs
    ) -> Union[Dict[str, Any], List[Image.Image], List[str]]:
        """
        Generate images based on the provided prompt (asynchronous version).
        
        Args:
            prompt (str): The text prompt to generate images from.
            model (str, optional): The model to use for generation. Defaults to "flux-pro/v1.1".
            size (str, optional): Size of the generated image (e.g., "1024x1024").
            num_images (int, optional): Number of images to generate. Defaults to 1.
            negative_prompt (str, optional): What not to include in the generated image.
            return_format (str, optional): Format to return the results in. Options:
                - "json": Raw JSON response (default)
                - "image": PIL Image object(s)
                - "b64": Base64 encoded string(s)
                - "url": URL string(s)
            **kwargs: Additional provider-specific parameters.
        
        Returns:
            Dict[str, Any] or List[Image.Image] or List[str]: Generated images in the requested format.
        
        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        self.logger.info(f"Generating image asynchronously with prompt: {prompt[:50]}...")
        
        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "model": model,
            "n": num_images
        }
        
        # Add optional parameters
        if size:
            payload["size"] = size
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
            
        # Add any additional kwargs to the payload
        payload.update(kwargs)
        
        # Make the API request asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
        
        # Process the result based on return_format
        if return_format == "json":
            return result
        elif return_format == "image":
            return await self._process_images_async(result)
        elif return_format == "b64":
            return self._extract_b64(result)  # This is fast enough to not need async
        elif return_format == "url":
            return self._extract_urls(result)  # This is fast enough to not need async
        else:
            raise ValueError(f"Invalid return_format: {return_format}")
    
    def _process_images(self, result: Dict[str, Any]) -> List[Image.Image]:
        """Convert API result to PIL Images (synchronous version)."""
        images = []
        
        # Handle different API response formats
        if "data" in result and isinstance(result["data"], list):
            items = result["data"]
        elif "images" in result and isinstance(result["images"], list):
            items = result["images"]
        else:
            items = [result]  # Fallback for unknown format
            
        for item in items:
            if "url" in item:
                # Download from URL
                response = requests.get(item["url"])
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                images.append(img)
            elif "b64_json" in item:
                # Decode from base64
                img_data = base64.b64decode(item["b64_json"])
                img = Image.open(BytesIO(img_data))
                images.append(img)
            elif "image" in item and isinstance(item["image"], str):
                # Handle base64 directly in 'image' field
                if self._is_base64(item["image"]):
                    img_data = base64.b64decode(item["image"])
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                # Handle URL in 'image' field
                elif item["image"].startswith(("http://", "https://")):
                    response = requests.get(item["image"])
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    images.append(img)
        
        return images
        
    async def _process_images_async(self, result: Dict[str, Any]) -> List[Image.Image]:
        """Convert API result to PIL Images (asynchronous version)."""
        images = []
        tasks = []
        
        # Handle different API response formats
        if "data" in result and isinstance(result["data"], list):
            items = result["data"]
        elif "images" in result and isinstance(result["images"], list):
            items = result["images"]
        else:
            items = [result]  # Fallback for unknown format
        
        async with aiohttp.ClientSession() as session:
            for item in items:
                if "url" in item:
                    # Download from URL asynchronously
                    task = self._download_image_async(session, item["url"])
                    tasks.append(task)
                elif "b64_json" in item:
                    # Decode from base64 (no need for async here as it's CPU-bound)
                    img_data = base64.b64decode(item["b64_json"])
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                elif "image" in item and isinstance(item["image"], str):
                    # Handle base64 directly in 'image' field
                    if self._is_base64(item["image"]):
                        img_data = base64.b64decode(item["image"])
                        img = Image.open(BytesIO(img_data))
                        images.append(img)
                    # Handle URL in 'image' field
                    elif item["image"].startswith(("http://", "https://")):
                        task = self._download_image_async(session, item["image"])
                        tasks.append(task)
            
            # Wait for all download tasks to complete
            if tasks:
                downloaded_images = await asyncio.gather(*tasks)
                images.extend(downloaded_images)
        
        return images
    
    async def _download_image_async(self, session: aiohttp.ClientSession, url: str) -> Image.Image:
        """Download an image asynchronously."""
        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            return Image.open(BytesIO(content))
    
    def _extract_b64(self, result: Dict[str, Any]) -> List[str]:
        """Extract base64 strings from API result."""
        b64_strings = []
        
        if "data" in result and isinstance(result["data"], list):
            items = result["data"]
        elif "images" in result and isinstance(result["images"], list):
            items = result["images"]
        else:
            items = [result]
            
        for item in items:
            if "b64_json" in item:
                b64_strings.append(item["b64_json"])
            elif "url" in item:
                # Convert URL to base64
                response = requests.get(item["url"])
                response.raise_for_status()
                b64_string = base64.b64encode(response.content).decode('utf-8')
                b64_strings.append(b64_string)
        
        return b64_strings
    
    def _extract_urls(self, result: Dict[str, Any]) -> List[str]:
        """Extract URLs from API result."""
        urls = []
        
        if "data" in result and isinstance(result["data"], list):
            items = result["data"]
        elif "images" in result and isinstance(result["images"], list):
            items = result["images"]
        else:
            items = [result]
            
        for item in items:
            if "url" in item:
                urls.append(item["url"])
        
        return urls
    
    def _is_base64(self, s: str) -> bool:
        """Check if a string is base64 encoded."""
        try:
            return base64.b64encode(base64.b64decode(s)) == s.encode()
        except (base64.binascii.Error, ValueError):
            return False
    
    def save_images(self, images: List[Image.Image], output_dir: str = "output", filename_prefix: str = "image", format: str = "png"):
        """
        Save generated images to disk (synchronous version).
        
        Args:
            images (List[Image.Image]): List of PIL Image objects to save.
            output_dir (str, optional): Directory to save images to. Defaults to "output".
            filename_prefix (str, optional): Prefix for filenames. Defaults to "image".
            format (str, optional): Image format to save as. Defaults to "png".
        
        Returns:
            List[str]: Paths to saved images.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, img in enumerate(images):
            path = os.path.join(output_dir, f"{filename_prefix}_{i}.{format}")
            img.save(path, format=format.upper())
            saved_paths.append(path)
            self.logger.info(f"Saved image to {path}")
        
        return saved_paths
        
    async def save_images_async(self, images: List[Image.Image], output_dir: str = "output", filename_prefix: str = "image", format: str = "png"):
        """
        Save generated images to disk asynchronously.
        
        Args:
            images (List[Image.Image]): List of PIL Image objects to save.
            output_dir (str, optional): Directory to save images to. Defaults to "output".
            filename_prefix (str, optional): Prefix for filenames. Defaults to "image".
            format (str, optional): Image format to save as. Defaults to "png".
        
        Returns:
            List[str]: Paths to saved images.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        tasks = []
        
        # Create save tasks
        for i, img in enumerate(images):
            path = os.path.join(output_dir, f"{filename_prefix}_{i}.{format}")
            task = self._save_image_async(img, path, format)
            tasks.append((path, task))
        
        # Execute all save tasks concurrently
        for path, task in tasks:
            await task
            saved_paths.append(path)
            self.logger.info(f"Saved image to {path}")
        
        return saved_paths
    
    async def _save_image_async(self, img: Image.Image, path: str, format: str):
        """Save a single image asynchronously using an executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            lambda: img.save(path, format=format.upper())
        )
        
    async def generate_batch_async(
        self,
        prompts: List[str],
        model: str = "flux-pro/v1.1",
        size: Optional[str] = None,
        return_format: str = "image",
        **kwargs
    ) -> List[Union[Dict[str, Any], List[Image.Image], List[str]]]:
        """
        Generate multiple images from different prompts concurrently.
        
        Args:
            prompts (List[str]): List of prompts to generate images from.
            model (str, optional): The model to use for generation.
            size (str, optional): Size of the generated images.
            return_format (str, optional): Format to return the results in.
            **kwargs: Additional parameters to pass to each generate call.
            
        Returns:
            List of results, one for each prompt.
        """
        tasks = []
        for prompt in prompts:
            task = self.generate_async(
                prompt=prompt,
                model=model,
                size=size,
                return_format=return_format,
                **kwargs
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

# Example usage
if __name__ == "__main__":
    # Synchronous example
    def run_sync_example():
        # Create generator (API key should be set in environment variable or passed directly)
        generator = ImageGenerator(api_key="YOUR_API_KEY")
        
        # Generate images
        result = generator.generate(
            prompt="A jellyfish in the ocean with bioluminescent glow",
            model="flux-pro/v1.1",
            num_images=2,
            return_format="image"  # Returns PIL Images
        )
        
        # Save the generated images
        if isinstance(result, list):
            generator.save_images(result, filename_prefix="jellyfish")
    
    # Asynchronous example
    async def run_async_example():
        # Create generator
        generator = ImageGenerator(api_key="YOUR_API_KEY")
        
        # Generate multiple images concurrently
        tasks = [
            generator.generate_async(
                prompt=f"A {subject} in a fantasy landscape",
                return_format="image"
            )
            for subject in ["dragon", "unicorn", "phoenix"]
        ]
        
        # Wait for all generations to complete
        results = await asyncio.gather(*tasks)
        
        # Save all images
        for i, images in enumerate(results):
            await generator.save_images_async(images, filename_prefix=f"fantasy_{i}")
        
        print("All images generated and saved!")
    
    # Run async example
    # asyncio.run(run_async_example())
    
    # Run sync example (default)
    run_sync_example()
