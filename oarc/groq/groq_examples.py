"""
✨ GroqMagic Examples
This file contains examples of how to use the GroqMagic library for various tasks.
"""
import asyncio
from groq_magic import GroqMagic, GroqModel, ResponseFormat

# Set your API key in environment variables
# os.environ["GROQ_API_KEY"] = "your-api-key-here"


def example_chat():
    """Example: Chat with a text model."""
    print("🧠 Chat Example")
    
    # Initialize the client
    client = GroqMagic()
    
    # Simple chat without history
    response = client.chat(
        model=GroqModel.LLAMA_3_8B,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        temperature=0.7,
        max_tokens=500,
    )
    
    print(f"Response: {response}\n")
    
    # Chat with streaming
    print("Streaming response:")
    response = client.chat(
        model=GroqModel.LLAMA_3_8B,
        messages=[{"role": "user", "content": "Tell me a short joke about programming."}],
        stream=True,
    )
    print("\n")
    
    # Chat with conversation history
    conversation_id = "example_convo"
    
    client.chat(
        model=GroqModel.LLAMA_3_8B,
        messages=[{"role": "user", "content": "What is machine learning?"}],
        conversation_id=conversation_id,
    )
    
    response = client.chat(
        model=GroqModel.LLAMA_3_8B,
        messages=[{"role": "user", "content": "Give me an example application."}],
        conversation_id=conversation_id,
    )
    
    print(f"Follow-up response: {response}")


def example_vision():
    """Example: Analyze images with a vision model."""
    print("📷 Vision Example")
    
    # Initialize the client
    client = GroqMagic()
    
    # Analyze an image
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
    
    print(f"Image analysis: {response}")


def example_transcribe():
    """Example: Transcribe audio with Whisper."""
    print("🔊 Transcription Example")
    
    # Initialize the client
    client = GroqMagic()
    
    # Transcribe an audio file
    transcription = client.transcribe(
        audio_file="path/to/your/audio.mp3",
        model=GroqModel.WHISPER_LARGE_V3_TURBO,
    )
    
    print(f"Transcription: {transcription}")
    
    # Transcribe with more options
    transcription_json = client.transcribe(
        audio_file="path/to/your/audio.mp3",
        response_format=ResponseFormat.VERBOSE_JSON,
        language="en",
    )
    
    print(f"Detailed transcription: {transcription_json}")


def example_moderation():
    """Example: Moderate content with Llama Guard."""
    print("🛡️ Moderation Example")
    
    # Initialize the client
    client = GroqMagic()
    
    # Moderate some content
    content_to_check = "I want to learn how to make a website for my business."
    result = client.moderate(content=content_to_check)
    
    print(f"Moderation result: {result}")


def example_code_generation():
    """Example: Generate code with specialized models."""
    print("💻 Code Generation Example")
    
    # Initialize the client
    client = GroqMagic()
    
    # Generate code
    prompt = "Create a Python function that calculates the Fibonacci sequence recursively."
    code = client.generate_code(
        prompt=prompt,
        model=GroqModel.DEEPSEEK_DISTILL_LLAMA_70B,
        temperature=0.1,
    )
    
    print(f"Generated code: \n{code}")


async def example_async():
    """Example: Using the async API."""
    print("🚀 Async API Example")
    
    # Initialize the client
    client = GroqMagic()
    
    # Run multiple requests concurrently
    tasks = [
        client.chat_async(
            model=GroqModel.LLAMA_3_8B,
            messages=[{"role": "user", "content": "Tell me about Mars."}],
        ),
        client.chat_async(
            model=GroqModel.LLAMA_3_8B,
            messages=[{"role": "user", "content": "Tell me about Venus."}],
        ),
        client.chat_async(
            model=GroqModel.LLAMA_3_8B,
            messages=[{"role": "user", "content": "Tell me about Jupiter."}],
        ),
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Print the results
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result[:100]}...\n")


if __name__ == "__main__":
    # Run the examples
    example_chat()
    # example_vision()  # Uncomment to run (requires an image file)
    # example_transcribe()  # Uncomment to run (requires an audio file)
    example_moderation()
    example_code_generation()
    
    # Run async example
    # asyncio.run(example_async())
