# ✨ GroqMagic

A unified interface for working with Groq AI models including text, vision, audio, moderation, and code.

![GroqMagic](https://via.placeholder.com/800x200?text=GroqMagic)

## Features

- 🧠 **Comprehensive Model Support**: Access all Groq models (Llama, Whisper, Llama Guard, etc.)
- 📷 **Vision Models**: Process and analyze images easily
- 🔊 **Audio Transcription**: Transcribe audio with Whisper models
- 🛡️ **Content Moderation**: Check content with Llama Guard
- 💻 **Code Generation**: Create code with specialized models
- 🔄 **Sync & Async APIs**: Choose between synchronous or asynchronous operation
- 📱 **CLI Tool**: Powerful command-line interface
- 📝 **Conversation History**: Built-in message history management
- 📊 **Streaming Support**: Stream responses in real-time

## Quick Start

### Set up your API key

```python
import os
os.environ["GROQ_API_KEY"] = "your-api-key-here"

# Or pass it directly
from groq_magic import GroqMagic
client = GroqMagic(api_key="your-api-key-here")
```

### Chat with a text model

```python
from groq_magic import GroqMagic, GroqModel

client = GroqMagic()

response = client.chat(
    model=GroqModel.LLAMA_3_70B,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
)

print(response)
```

### Analyze an image

```python
response = client.chat(
    model=GroqModel.LLAMA_3_2_11B_VISION,
    messages=[
        {
            "role": "user",
            "content": "What's in this image?",
            "image_path": "path/to/your/image.jpg"
        }
    ],
)

print(response)
```

### Transcribe audio

```python
transcription = client.transcribe(
    audio_file="path/to/your/audio.mp3",
    model=GroqModel.WHISPER_LARGE_V3_TURBO,
)

print(transcription)
```

### Generate code

```python
code = client.generate_code(
    prompt="Create a Python function that calculates the Fibonacci sequence",
    model=GroqModel.DEEPSEEK_DISTILL_LLAMA_70B,
    temperature=0.1,
)

print(code)
```

### Use the async API

```python
import asyncio
from groq_magic import GroqMagic

async def main():
    client = GroqMagic()
    
    # Run multiple requests concurrently
    tasks = [
        client.chat_async(
            model="llama-3-8b",
            messages=[{"role": "user", "content": "Tell me about Mars."}],
        ),
        client.chat_async(
            model="llama-3-8b",
            messages=[{"role": "user", "content": "Tell me about Venus."}],
        ),
    ]
    
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

# Run the async function
asyncio.run(main())
```

## Command-Line Interface

GroqMagic includes a powerful CLI for easy access to Groq models:

```bash
# Chat with a model
groq-magic chat --model llama-3-8b --prompt "Tell me a joke"

# Interactive chat mode
groq-magic chat --model llama-3-70b

# Analyze an image
groq-magic vision --image photo.jpg --prompt "What's in this image?"

# Transcribe audio
groq-magic transcribe --file audio.mp3

# Moderate content
groq-magic moderate --content "Check if this content is appropriate"

# Generate code
groq-magic code --prompt "Create a Python function to sort a list"

# List all available models
groq-magic list
```

## Advanced Usage

### Conversation tracking

```python
# Start a conversation
client.chat(
    model=GroqModel.LLAMA_3_8B,
    messages=[{"role": "user", "content": "What is machine learning?"}],
    conversation_id="my_conversation",
)

# Continue the conversation
response = client.chat(
    model=GroqModel.LLAMA_3_8B,
    messages=[{"role": "user", "content": "Give me an example application."}],
    conversation_id="my_conversation",
)

# Clear conversation history
client.clear_history("my_conversation")
```

### Streaming responses

```python
def process_chunk(chunk):
    # Process each chunk as it arrives
    print(chunk, end="")

response = client.chat(
    model=GroqModel.LLAMA_3_70B,
    messages=[{"role": "user", "content": "Write a short story about time travel."}],
    stream=True,
    callback=process_chunk,
)
```

## Available Models

GroqMagic provides easy access to all Groq models through enums:

```python
from groq_magic import GroqModel

# Text models
GroqModel.LLAMA_3_8B
GroqModel.LLAMA_3_70B
GroqModel.LLAMA_3_1_8B
GroqModel.LLAMA_3_1_70B
GroqModel.LLAMA_3_2_11B
GroqModel.LLAMA_3_3_70B_VERSATILE

# Vision models
GroqModel.LLAMA_3_1_8B_VISION
GroqModel.LLAMA_3_1_70B_VISION
GroqModel.LLAMA_3_2_11B_VISION

# Moderation models
GroqModel.LLAMA_GUARD_3_8B

# Audio models
GroqModel.WHISPER_LARGE_V3
GroqModel.WHISPER_LARGE_V3_TURBO

# Code models
GroqModel.QWEN_CODER
GroqModel.DEEPSEEK_DISTILL_LLAMA_70B
```

## License

MIT

---

Made with ✨ by [Your Name]
