# 🖼️ AI ML Image Generator

A versatile Python class for generating images using AI/ML APIs with both synchronous and asynchronous capabilities.

## Features

- 🚀 **Simple Interface**: Easy to use with any AI image generation API
- ⚡ **Async Support**: Generate multiple images concurrently
- 🔄 **Multiple Return Formats**: Get results as JSON, PIL Images, base64 strings, or URLs
- 💾 **Image Saving**: Helper methods to save generated images to disk
- 🛠️ **Flexible Configuration**: Support for model selection, image size, negative prompts
- 📊 **Error Handling**: Comprehensive error handling and logging

## Installation

```bash
pip install pillow requests aiohttp
```

## Quick Start

### Basic Usage

```python
from image_generator_class import ImageGenerator

# Initialize with API key
generator = ImageGenerator(api_key="your_api_key_here")

# Generate an image
result = generator.generate(
    prompt="A sunset over a mountain lake with reflections",
    model="flux-pro/v1.1",
    return_format="image"  # Returns PIL Images
)

# Save the generated images
generator.save_images(result, filename_prefix="sunset")
```

### Using Environment Variables

```python
import os

# Set API key in environment variable
os.environ["AIML_API_KEY"] = "your_api_key_here"

# Initialize without explicitly providing API key
generator = ImageGenerator()
```

## Async Examples

### Generate Multiple Images Concurrently

```python
import asyncio
from image_generator_class import ImageGenerator

async def main():
    generator = ImageGenerator(api_key="your_api_key_here")
    
    # Generate multiple images concurrently
    results = await generator.generate_batch_async(
        prompts=[
            "A futuristic city with flying cars",
            "An underwater civilization with bioluminescent architecture",
            "A steampunk airship flying through clouds"
        ],
        return_format="image"
    )
    
    # Save all images
    for i, images in enumerate(results):
        await generator.save_images_async(images, filename_prefix=f"scene_{i}")
    
    print("All images generated and saved!")

# Run the async function
asyncio.run(main())
```

### Streaming Example

```python
import asyncio
from image_generator_class import ImageGenerator

async def main():
    generator = ImageGenerator(api_key="your_api_key_here")
    
    # Process images as they're completed
    prompts = [f"Portrait of a {character}" for character in [
        "wizard", "knight", "dragon rider", "elf archer"
    ]]
    
    for i, prompt in enumerate(prompts):
        image = await generator.generate_async(
            prompt=prompt,
            return_format="image"
        )
        
        # Process each image as it completes
        await generator.save_images_async(image, filename_prefix=f"character_{i}")
        print(f"Generated and saved image {i+1}/{len(prompts)}")

# Run the async function
asyncio.run(main())
```

## API Reference

### Initialization

```python
generator = ImageGenerator(
    api_key="your_api_key",             # Optional: API key (can use env var instead)
    api_url="https://api.example.com",  # Optional: API endpoint URL
    env_key_name="API_KEY_ENV_NAME"     # Optional: Name of env var containing API key
)
```

### Generate Images (Sync)

```python
result = generator.generate(
    prompt="Image description",         # Required: Text prompt
    model="flux-pro/v1.1",             # Optional: Model to use
    size="1024x1024",                   # Optional: Image size
    num_images=1,                       # Optional: Number of images to generate
    negative_prompt="What to avoid",    # Optional: Negative prompt
    return_format="json"                # Optional: Format to return (json/image/b64/url)
)
```

### Generate Images (Async)

```python
result = await generator.generate_async(
    prompt="Image description",
    model="flux-pro/v1.1",
    size="1024x1024",
    num_images=1,
    negative_prompt="What to avoid",
    return_format="json"
)
```

### Generate Multiple Images (Async)

```python
results = await generator.generate_batch_async(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    model="flux-pro/v1.1",
    size="1024x1024",
    return_format="image"
)
```

### Save Images

```python
# Synchronous
paths = generator.save_images(
    images,                     # List of PIL Image objects
    output_dir="output",        # Directory to save images to
    filename_prefix="image",    # Prefix for filenames
    format="png"                # Image format
)

# Asynchronous
paths = await generator.save_images_async(
    images,
    output_dir="output",
    filename_prefix="image",
    format="png"
)
```

## Return Formats

The `return_format` parameter accepts the following values:

- `"json"`: Returns the raw JSON response from the API
- `"image"`: Returns a list of PIL Image objects
- `"b64"`: Returns a list of base64-encoded image strings
- `"url"`: Returns a list of URLs to the generated images

## Error Handling

The class includes comprehensive error handling for:

- Missing API keys
- API request failures
- Invalid response formats
- Image processing errors

## Example Implementation

```python
import asyncio
from image_generator_class import ImageGenerator

async def generate_art_series():
    generator = ImageGenerator(api_key="your_api_key_here")
    
    # Generate a series of related images
    prompts = [
        "A peaceful mountain landscape at sunrise",
        "A peaceful mountain landscape at noon",
        "A peaceful mountain landscape at sunset",
        "A peaceful mountain landscape at night"
    ]
    
    results = await generator.generate_batch_async(
        prompts=prompts,
        model="flux-pro/v1.1",
        size="1024x1024",
        return_format="image"
    )
    
    # Save the time series
    for i, images in enumerate(results):
        await generator.save_images_async(
            images, 
            output_dir="mountain_series",
            filename_prefix=f"time_{i}",
            format="png"
        )
    
    print("Art series complete!")

# Run the async function
if __name__ == "__main__":
    asyncio.run(generate_art_series())
```

## API Compatibility

This class is designed to work with aimlapi.com by default but can be adapted to work with any image generation API that follows a similar interface pattern.

Visit [aimlapi.com/models/flux-1-1-pro-api](https://aimlapi.com/models/flux-1-1-pro-api) for more details on API keys and available models.

## License

MIT

---

Created by @BorcherdingL
