# OllamaCommands Documentation

The `OllamaCommands` class provides a comprehensive set of methods for interacting with the Ollama library. It simplifies working with local large language models by wrapping the Ollama API with a consistent interface for both synchronous and asynchronous operations.

## Table of Contents

- [Installation](#installation)
- [Initialization](#initialization)
- [Core Methods](#core-methods)
  - [Model Information Methods](#model-information-methods)
  - [Model Management Methods](#model-management-methods)
  - [Generation Methods](#generation-methods)
  - [Utility Methods](#utility-methods)
- [Examples](#examples)
  - [Synchronous Usage](#synchronous-usage)
  - [Asynchronous Usage](#asynchronous-usage)
- [Error Handling](#error-handling)
- [Logging](#logging)

## Installation

To use the `OllamaCommands` class, you need to install the Ollama Python library:

```bash
pip install ollama
```

Make sure that Ollama is installed and running on your system. You can download it from [Ollama's website](https://ollama.com/download).

## Initialization

```python
from ollama_commands import OllamaCommands

# Default initialization (async mode, localhost)
commands = OllamaCommands()

# Custom initialization
commands = OllamaCommands(
    host="http://localhost:11434",  # Ollama server URL
    async_mode=True                 # Whether to use async methods by default
)
```

## Core Methods

### Model Information Methods

#### `async ollama_show_modelfile(user_input_model_select: str) -> Dict[str, Any]`

Get the full model information for a specified model.

```python
model_info = await commands.ollama_show_modelfile("llama3.2")
print(model_info)
```

#### `async ollama_show_template(user_input_model_select: str) -> str`

Get just the template for a specified model.

```python
template = await commands.ollama_show_template("llama3.2")
print(template)
```

#### `async ollama_show_license(user_input_model_select: str) -> str`

Get just the license information for a specified model.

```python
license_info = await commands.ollama_show_license("llama3.2")
print(license_info)
```

#### `async ollama_show_loaded_models() -> List[Dict[str, Any]]`

Get information about currently running models.

```python
loaded_models = await commands.ollama_show_loaded_models()
for model in loaded_models:
    print(f"Model {model['name']} is running")
```

#### `async ollama_list() -> List[str]`

Get list of available models.

```python
available_models = await commands.ollama_list()
print("Available models:", available_models)
```

### Model Management Methods

#### `async ollama_pull(model_name: str) -> Dict[str, Any]`

Pull a model from the Ollama library.

```python
result = await commands.ollama_pull("llama3.2")
print("Pull result:", result)
```

#### `async ollama_delete(model_name: str) -> Dict[str, Any]`

Delete a model from local storage.

```python
result = await commands.ollama_delete("unused-model")
print("Delete result:", result)
```

#### `async ollama_copy(source_model: str, target_model: str) -> Dict[str, Any]`

Copy a model to a new name.

```python
result = await commands.ollama_copy("llama3.2", "my-custom-llama")
print("Copy result:", result)
```

#### `async ollama_create(model_name: str, from_model: str, system_prompt: str = None, modelfile: str = None) -> Dict[str, Any]`

Create a new model based on an existing one.

```python
result = await commands.ollama_create(
    model_name="my-assistant",
    from_model="llama3.2",
    system_prompt="You are a helpful assistant that speaks like Shakespeare."
)
print("Create result:", result)
```

### Generation Methods

#### `async ollama_chat(model_name: str, messages: List[Dict[str, str]], stream: bool = False, options: Dict[str, Any] = None) -> Union[Dict[str, Any], AsyncGenerator]`

Chat with a model using a conversation history.

```python
# Single response
response = await commands.ollama_chat(
    model_name="llama3.2",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print("Chat response:", response['message']['content'])

# Streaming response
async for chunk in await commands.ollama_chat(
    model_name="llama3.2",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

#### `async ollama_generate(model_name: str, prompt: str, stream: bool = False, options: Dict[str, Any] = None) -> Union[Dict[str, Any], AsyncGenerator]`

Generate a response from a model with a single prompt.

```python
# Single response
response = await commands.ollama_generate(
    model_name="llama3.2",
    prompt="The capital of France is"
)
print("Generated response:", response['response'])

# Streaming response
async for chunk in await commands.ollama_generate(
    model_name="llama3.2",
    prompt="Write a poem about mountains",
    stream=True
):
    print(chunk['response'], end='', flush=True)
```

#### `async ollama_embed(model_name: str, input_text: Union[str, List[str]]) -> Dict[str, Any]`

Get embeddings for text.

```python
# Single text embedding
embeddings = await commands.ollama_embed(
    model_name="llama3.2",
    input_text="The sky is blue"
)
print("Embedding:", embeddings['embedding'])

# Batch embeddings
batch_embeddings = await commands.ollama_embed(
    model_name="llama3.2",
    input_text=["The sky is blue", "Grass is green"]
)
print("Batch embeddings:", batch_embeddings['embeddings'])
```

### Utility Methods

#### `async get_model_details(model_name: str) -> Dict[str, Any]`

Get comprehensive details about a model including parameters, size, and more.

```python
details = await commands.get_model_details("llama3.2")
print("Model details:", details)
```

#### `async get_models_summary() -> Dict[str, Any]`

Get a summary of all available and running models.

```python
summary = await commands.get_models_summary()
print(f"Available models: {summary['available_count']}")
print(f"Running models: {summary['running_count']}")
```

#### `is_model_running(model_name: str, running_models: List[Dict[str, Any]] = None) -> bool`

Check if a specific model is currently running.

```python
if commands.is_model_running("llama3.2"):
    print("The model is running")
else:
    print("The model is not running")
```

#### `async ensure_model_available(model_name: str, auto_pull: bool = False) -> bool`

Ensure a model is available locally, optionally pulling it if not.

```python
# Check if available, pull if not
available = await commands.ensure_model_available("llama3.2", auto_pull=True)
if available:
    print("Model is available and ready to use")
else:
    print("Failed to make model available")
```

## Examples

### Synchronous Usage

```python
import asyncio
from ollama_commands import OllamaCommands

# Initialize in sync mode
commands = OllamaCommands(async_mode=False)

# Run async methods in a loop
loop = asyncio.new_event_loop()
models = loop.run_until_complete(commands.ollama_list())
loop.close()

print("Available models:", models)

if models:
    # Use the first available model
    model = models[0]
    
    # Generate a response using direct ollama API
    import ollama
    response = ollama.generate(model=model, prompt="Hello, how are you?")
    print(f"Response from {model}:", response.get('response', ''))
```

### Asynchronous Usage

```python
import asyncio
from ollama_commands import OllamaCommands

async def main():
    # Initialize in async mode (default)
    commands = OllamaCommands()
    
    # Get available models
    models = await commands.ollama_list()
    print("Available models:", models)
    
    if models:
        # Use the first available model
        model = models[0]
        
        # Ensure model is available
        await commands.ensure_model_available(model, auto_pull=True)
        
        # Chat with the model
        response = await commands.ollama_chat(
            model_name=model,
            messages=[{"role": "user", "content": "What's the meaning of life?"}]
        )
        print("Response:", response['message']['content'])

# Run the async function
asyncio.run(main())
```

## Error Handling

The `OllamaCommands` class includes comprehensive error handling and logging. All methods catch exceptions and log errors, making it easier to debug issues. The class returns appropriate default values (like empty lists or dictionaries) when errors occur.

```python
try:
    response = await commands.ollama_chat(
        model_name="non-existent-model",
        messages=[{"role": "user", "content": "Hello"}]
    )
except ollama.ResponseError as e:
    if e.status_code == 404:
        print("Model not found, pulling it now...")
        await commands.ollama_pull("non-existent-model")
    else:
        print("Error:", e.error)
```

## Logging

The class uses Python's standard logging module. You can configure logging to see what's happening:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to logging.DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now use the class
commands = OllamaCommands()
```

This configuration will show informational messages about API calls, model loading, and other operations. Set the level to `logging.DEBUG` for even more detailed logs.

---

The `OllamaCommands` class provides a convenient and robust interface to the Ollama API, making it easy to work with local large language models in your Python applications.
