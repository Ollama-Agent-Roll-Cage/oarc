"""
groq-magic.py

A unified interface for working with Groq AI models, providing support for text generation,
vision analysis, audio transcription, content moderation, and code generation.

Features:
- Sync and async APIs for all operations
- Streaming support for real-time results
- Built-in message history management
- Automatic image encoding for vision models
- Command-line interface
- Comprehensive error handling

Author: Your Name
Version: 1.0.0
"""

# Standard library imports
import os
import sys
import json
import base64
import asyncio
import logging
import argparse
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Union, Optional, BinaryIO, Callable
from io import BytesIO

# Third-party imports
from PIL import Image

# Emoji constants for CLI and logging
EMOJI = {
    "sparkles": "✨",
    "brain": "🧠",
    "rocket": "🚀",
    "camera": "📷",
    "shield": "🛡️",
    "sound": "🔊",
    "code": "💻",
    "check": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "clock": "⏱️",
    "star": "⭐",
    "light": "💡",
}

try:
    from groq import Groq
    from groq.types.chat import ChatCompletionMessageParam
except ImportError:
    raise ImportError(
        f"{EMOJI['error']} The Groq Python package is not installed. "
        "Please install it using: pip install groq"
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GroqMagic")


class ModelType(Enum):
    """Enum for categorizing different types of Groq models."""
    
    TEXT = "text"
    VISION = "vision"
    MODERATION = "moderation"
    AUDIO = "audio"
    CODE = "code"


class GroqModel(Enum):
    """Enum for all available Groq models with their types and capabilities."""
    
    # Text models
    LLAMA_3_8B = ("llama-3-8b", ModelType.TEXT, "Fast and efficient text generation")
    LLAMA_3_70B = ("llama-3-70b", ModelType.TEXT, "High-quality text generation")
    LLAMA_3_1_8B = ("llama-3.1-8b", ModelType.TEXT, "Updated 8B text model")
    LLAMA_3_1_70B = ("llama-3.1-70b", ModelType.TEXT, "Updated 70B text model")
    LLAMA_3_2_11B = ("llama-3.2-11b", ModelType.TEXT, "Latest 11B text model")
    LLAMA_3_3_70B_VERSATILE = ("llama-3.3-70b-versatile", ModelType.TEXT, "Most capable text model")
    
    # Vision models
    LLAMA_3_1_8B_VISION = ("llama-3.1-8b-vision", ModelType.VISION, "Efficient multimodal model")
    LLAMA_3_1_70B_VISION = ("llama-3.1-70b-vision", ModelType.VISION, "High-quality multimodal model")
    LLAMA_3_2_11B_VISION = ("llama-3.2-11b-vision-preview", ModelType.VISION, "Latest multimodal model")
    
    # Moderation models
    LLAMA_GUARD_3_8B = ("llama-guard-3-8b", ModelType.MODERATION, "Content moderation model")
    
    # Audio models
    WHISPER_LARGE_V3 = ("whisper-large-v3", ModelType.AUDIO, "High-quality audio transcription")
    WHISPER_LARGE_V3_TURBO = ("whisper-large-v3-turbo", ModelType.AUDIO, "Fast audio transcription")
    
    # Code models
    QWEN_CODER = ("qwen-coder", ModelType.CODE, "Specialized code generation")
    DEEPSEEK_DISTILL_LLAMA_70B = ("deepseek-r1-distill-llama-70b", ModelType.CODE, "Powerful code model")
    
    def __init__(self, id: str, model_type: ModelType, description: str):
        self.id = id
        self.type = model_type
        self.description = description


class MessageRole(Enum):
    """Enum for different roles in a chat conversation."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ResponseFormat(Enum):
    """Enum for different response formats."""
    
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"


class GroqMagic:
    """
    ✨ GroqMagic: A unified interface for working with Groq AI models.
    
    This class provides a simple, consistent API for working with all Groq models:
    - Text generation models (Llama 3 family)
    - Vision models (Llama Vision models)
    - Moderation models (Llama Guard)
    - Audio transcription (Whisper)
    - Code generation (Qwen Coder, DeepSeek)
    
    Features:
    - Sync and async APIs for all operations
    - Streaming support for real-time results
    - Built-in message history management
    - Automatic image encoding for vision models
    - Command-line interface
    - Comprehensive error handling
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        """
        Initialize the GroqMagic client.
        
        Args:
            api_key (str, optional): Groq API key. If not provided, will try to get from GROQ_API_KEY environment variable.
            timeout (int, optional): Request timeout in seconds. Defaults to 60.
        
        Raises:
            ValueError: If API key is not provided and not found in environment variables.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                f"{EMOJI['error']} API key not provided and not found in GROQ_API_KEY environment variable."
            )
        
        self.client = Groq(api_key=self.api_key, timeout=timeout)
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logger
    
    def _prepare_messages(
        self, 
        messages: List[Dict[str, str]], 
        conversation_id: Optional[str] = None
    ) -> List[ChatCompletionMessageParam]:
        """
        Prepare messages for the API, handling image encoding for vision models.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.
            conversation_id (str, optional): ID for conversation history tracking.
            
        Returns:
            List[ChatCompletionMessageParam]: Prepared messages for the API.
        """
        prepared_messages = []
        
        # Add conversation history if provided
        if conversation_id and conversation_id in self.history:
            prepared_messages.extend(self.history[conversation_id])
        
        # Process new messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Handle image content for vision models
            if isinstance(content, list) or "image" in message or "image_path" in message:
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                
                # Add image from URL
                if "image" in message and message["image"].startswith(("http://", "https://")):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": message["image"]}
                    })
                
                # Add image from path
                if "image_path" in message:
                    image_path = message["image_path"]
                    with open(image_path, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                
                # Add image from PIL Image
                if "pil_image" in message and isinstance(message["pil_image"], Image.Image):
                    buffer = BytesIO()
                    message["pil_image"].save(buffer, format="JPEG")
                    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            
            prepared_message = {"role": role, "content": content}
            
            # Add name if provided
            if "name" in message:
                prepared_message["name"] = message["name"]
                
            prepared_messages.append(prepared_message)
        
        return prepared_messages
    
    def chat(
        self,
        model: Union[str, GroqModel],
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        conversation_id: Optional[str] = None,
        response_format: Optional[Union[str, ResponseFormat]] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a chat completion using any Groq text or vision model.
        
        Args:
            model (Union[str, GroqModel]): Model to use for generation.
            messages (List[Dict[str, Any]]): List of messages for the conversation.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            top_p (float, optional): Nucleus sampling parameter. Defaults to 0.95.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            stop (Union[str, List[str]], optional): Stop sequences to end generation.
            conversation_id (str, optional): ID for conversation history tracking.
            response_format (Union[str, ResponseFormat], optional): Format of the response.
            seed (int, optional): Random seed for deterministic generation.
            callback (Callable, optional): Function to call for each chunk when streaming.
            
        Returns:
            Union[str, Dict[str, Any]]: Generated response text or full response object.
            
        Raises:
            ValueError: If an invalid model or parameter is provided.
        """
        # Convert model enum to string if needed
        model_id = model.id if isinstance(model, GroqModel) else model
        self.logger.info(f"{EMOJI['brain']} Using model: {model_id}")
        
        # Convert response format enum to string if needed
        if isinstance(response_format, ResponseFormat):
            response_format_val = {"type": response_format.value}
        elif isinstance(response_format, str):
            response_format_val = {"type": response_format}
        else:
            response_format_val = None
            
        # Prepare messages
        prepared_messages = self._prepare_messages(messages, conversation_id)
        
        # Create completion
        try:
            completion = self.client.chat.completions.create(
                model=model_id,
                messages=prepared_messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                stop=stop,
                response_format=response_format_val,
                seed=seed,
            )
            
            # Handle streaming response
            if stream:
                full_text = ""
                for chunk in completion:
                    content = chunk.choices[0].delta.content or ""
                    full_text += content
                    
                    if callback and content:
                        callback(content)
                    elif content:
                        print(content, end="", flush=True)
                
                # Store in conversation history if tracking
                if conversation_id:
                    if conversation_id not in self.history:
                        self.history[conversation_id] = []
                    
                    # Add user message to history
                    for msg in messages:
                        # Only add text content to history
                        if isinstance(msg.get("content"), str):
                            self.history[conversation_id].append({
                                "role": msg.get("role", "user"),
                                "content": msg.get("content", "")
                            })
                    
                    # Add assistant response to history
                    self.history[conversation_id].append({
                        "role": "assistant",
                        "content": full_text
                    })
                
                return full_text
            
            # Handle regular response
            response_text = completion.choices[0].message.content
            
            # Store in conversation history if tracking
            if conversation_id:
                if conversation_id not in self.history:
                    self.history[conversation_id] = []
                
                # Add user message to history
                for msg in messages:
                    # Only add text content to history
                    if isinstance(msg.get("content"), str):
                        self.history[conversation_id].append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                
                # Add assistant response to history
                self.history[conversation_id].append({
                    "role": "assistant",
                    "content": response_text
                })
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"{EMOJI['error']} Error generating completion: {str(e)}")
            raise
    
    async def chat_async(
        self,
        model: Union[str, GroqModel],
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        conversation_id: Optional[str] = None,
        response_format: Optional[Union[str, ResponseFormat]] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Asynchronously generate a chat completion using any Groq text or vision model.
        
        Note: This is a convenience wrapper around the synchronous method since
        the official Groq Python client doesn't support async natively yet.
        """
        # Use asyncio to run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                stop=stop,
                conversation_id=conversation_id,
                response_format=response_format,
                seed=seed,
                callback=callback,
            )
        )
    
    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        model: Union[str, GroqModel] = GroqModel.WHISPER_LARGE_V3_TURBO,
        response_format: Union[str, ResponseFormat] = ResponseFormat.TEXT,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Transcribe audio using Groq's Whisper models.
        
        Args:
            audio_file (Union[str, BinaryIO]): Path to audio file or file-like object.
            model (Union[str, GroqModel], optional): Whisper model to use.
            response_format (Union[str, ResponseFormat], optional): Format of the response.
            prompt (str, optional): Optional prompt for transcription context.
            language (str, optional): Language code (e.g., "en") for transcription.
            
        Returns:
            Union[str, Dict[str, Any]]: Transcription text or full response object.
            
        Raises:
            ValueError: If an invalid model or parameter is provided.
            FileNotFoundError: If the audio file doesn't exist.
        """
        # Convert model enum to string if needed
        model_id = model.id if isinstance(model, GroqModel) else model
        self.logger.info(f"{EMOJI['sound']} Transcribing with model: {model_id}")
        
        # Convert response format enum to string if needed
        format_val = response_format.value if isinstance(response_format, ResponseFormat) else response_format
        
        try:
            # Handle string file path
            if isinstance(audio_file, str):
                file_path = audio_file
                with open(file_path, "rb") as file:
                    file_content = file.read()
                    file_name = os.path.basename(file_path)
                    
                    transcription = self.client.audio.transcriptions.create(
                        file=(file_name, file_content),
                        model=model_id,
                        response_format=format_val,
                        prompt=prompt,
                        language=language,
                    )
            # Handle file-like object
            else:
                # If a file-like object was passed, read it and use a default name
                file_content = audio_file.read()
                file_name = getattr(audio_file, "name", "audio.mp3")
                
                transcription = self.client.audio.transcriptions.create(
                    file=(file_name, file_content),
                    model=model_id,
                    response_format=format_val,
                    prompt=prompt,
                    language=language,
                )
            
            # Return text for simple format, full response for JSON formats
            if format_val == "text":
                return transcription.text
            else:
                return transcription
                
        except FileNotFoundError:
            self.logger.error(f"{EMOJI['error']} Audio file not found: {audio_file}")
            raise
        except Exception as e:
            self.logger.error(f"{EMOJI['error']} Error transcribing audio: {str(e)}")
            raise
    
    async def transcribe_async(
        self,
        audio_file: Union[str, BinaryIO],
        model: Union[str, GroqModel] = GroqModel.WHISPER_LARGE_V3_TURBO,
        response_format: Union[str, ResponseFormat] = ResponseFormat.TEXT,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Asynchronously transcribe audio using Groq's Whisper models.
        
        This is a convenience wrapper around the synchronous method.
        """
        # Use asyncio to run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.transcribe(
                audio_file=audio_file,
                model=model,
                response_format=response_format,
                prompt=prompt,
                language=language,
            )
        )
    
    def moderate(
        self,
        content: str,
        model: Union[str, GroqModel] = GroqModel.LLAMA_GUARD_3_8B,
        temperature: float = 0.0,  # Lower temperature for deterministic moderation
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Moderate content using Llama Guard.
        
        Args:
            content (str): Content to moderate.
            model (Union[str, GroqModel], optional): Moderation model to use.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 512.
            
        Returns:
            Dict[str, Any]: Moderation result with flagged categories and reasoning.
        """
        # Convert model enum to string if needed
        model_id = model.id if isinstance(model, GroqModel) else model
        self.logger.info(f"{EMOJI['shield']} Moderating content with model: {model_id}")
        
        system_prompt = """You are a content moderator. Your task is to identify content that falls into 
        the following categories:
        
        1. Violence and Hate
        2. Sexual Content
        3. Criminal Planning
        4. Guns and Illegal Weapons
        5. Regulated or Controlled Substances
        6. Self-Harm
        
        For each category, indicate whether the content is flagged (true) or not flagged (false).
        Provide a brief explanation for each category. Format your response as JSON.
        """
        
        try:
            # Prepare messages for moderation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please moderate the following content: {content}"}
            ]
            
            # Use JSON response format
            completion = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            
            # Parse the JSON response
            response_text = completion.choices[0].message.content
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                return {
                    "error": "Invalid JSON response from moderation model",
                    "raw_response": response_text
                }
                
        except Exception as e:
            self.logger.error(f"{EMOJI['error']} Error moderating content: {str(e)}")
            raise
    
    async def moderate_async(
        self,
        content: str,
        model: Union[str, GroqModel] = GroqModel.LLAMA_GUARD_3_8B,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Asynchronously moderate content using Llama Guard.
        
        This is a convenience wrapper around the synchronous method.
        """
        # Use asyncio to run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.moderate(
                content=content,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    
    def generate_code(
        self,
        prompt: str,
        model: Union[str, GroqModel] = GroqModel.DEEPSEEK_DISTILL_LLAMA_70B,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        stream: bool = True,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate code using specialized code models.
        
        Args:
            prompt (str): Description of the code to generate.
            model (Union[str, GroqModel], optional): Code model to use.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 4096.
            stream (bool, optional): Whether to stream the response. Defaults to True.
            callback (Callable, optional): Function to call for each chunk when streaming.
            
        Returns:
            str: Generated code.
        """
        # Convert model enum to string if needed
        model_id = model.id if isinstance(model, GroqModel) else model
        self.logger.info(f"{EMOJI['code']} Generating code with model: {model_id}")
        
        # Craft a specific system prompt for code generation
        system_prompt = """You are an expert programmer. Your task is to write clean, efficient, and well-documented code 
        based on the user's requirements. Include comments to explain complex parts and provide a brief explanation of how 
        the code works at the beginning."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            callback=callback,
        )
    
    async def generate_code_async(
        self,
        prompt: str,
        model: Union[str, GroqModel] = GroqModel.DEEPSEEK_DISTILL_LLAMA_70B,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        stream: bool = True,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Asynchronously generate code using specialized code models.
        
        This is a convenience wrapper around the synchronous method.
        """
        # Use asyncio to run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_code(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                callback=callback,
            )
        )
    
    def clear_history(self, conversation_id: Optional[str] = None):
        """
        Clear conversation history.
        
        Args:
            conversation_id (str, optional): ID of conversation to clear.
                If None, clears all conversation history.
        """
        if conversation_id:
            if conversation_id in self.history:
                del self.history[conversation_id]
                self.logger.info(f"{EMOJI['check']} Cleared history for conversation: {conversation_id}")
        else:
            self.history = {}
            self.logger.info(f"{EMOJI['check']} Cleared all conversation history")
    
    def list_models(self) -> Dict[ModelType, List[GroqModel]]:
        """
        List all available models organized by type.
        
        Returns:
            Dict[ModelType, List[GroqModel]]: Models organized by type.
        """
        models_by_type = {}
        
        for model in GroqModel:
            if model.type not in models_by_type:
                models_by_type[model.type] = []
            
            models_by_type[model.type].append(model)
        
        return models_by_type
    
    @staticmethod
    def format_model_list(models_by_type: Dict[ModelType, List[GroqModel]]) -> str:
        """
        Format the model list as a string for display.
        
        Args:
            models_by_type (Dict[ModelType, List[GroqModel]]): Models organized by type.
            
        Returns:
            str: Formatted string of models.
        """
        output = f"\n{EMOJI['rocket']} Available Groq Models:\n\n"
        
        for model_type, models in models_by_type.items():
            type_emoji = {
                ModelType.TEXT: EMOJI["brain"],
                ModelType.VISION: EMOJI["camera"],
                ModelType.MODERATION: EMOJI["shield"],
                ModelType.AUDIO: EMOJI["sound"],
                ModelType.CODE: EMOJI["code"],
            }.get(model_type, "")
            
            output += f"{type_emoji} {model_type.value.upper()} MODELS:\n"
            
            for model in models:
                output += f"  - {model.id}: {model.description}\n"
            
            output += "\n"
        
        return output


def cli():
    """Command-line interface for GroqMagic."""
    parser = argparse.ArgumentParser(
        description=f"{EMOJI['sparkles']} GroqMagic - A unified interface for Groq AI models"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with a Groq model")
    chat_parser.add_argument("--model", "-m", default="llama-3-8b", help="Model ID to use")
    chat_parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    chat_parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling parameter")
    chat_parser.add_argument("--stream", "-s", action="store_true", help="Stream the response")
    chat_parser.add_argument("--prompt", "-p", help="Chat prompt (uses interactive mode if not provided)")
    chat_parser.add_argument("--system", help="System message")
    chat_parser.add_argument("--conversation", "-c", help="Conversation ID for history tracking")
    
    # Vision command
    vision_parser = subparsers.add_parser("vision", help="Analyze an image with a Groq vision model")
    vision_parser.add_argument("--model", "-m", default="llama-3.2-11b-vision-preview", help="Vision model ID")
    vision_parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    vision_parser.add_argument("--image", "-i", required=True, help="Path to image file")
    vision_parser.add_argument("--prompt", "-p", required=True, help="Text prompt about the image")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio with Whisper")
    transcribe_parser.add_argument("--model", "-m", default="whisper-large-v3-turbo", help="Whisper model ID")
    transcribe_parser.add_argument("--file", "-f", required=True, help="Path to audio file")
    transcribe_parser.add_argument("--format", choices=["text", "json", "verbose_json"], default="text", 
                                  help="Response format")
    transcribe_parser.add_argument("--language", "-l", help="Language code (e.g., 'en')")
    
    # Moderate command
    moderate_parser = subparsers.add_parser("moderate", help="Moderate content with Llama Guard")
    moderate_parser.add_argument("--model", "-m", default="llama-guard-3-8b", help="Moderation model ID")
    moderate_parser.add_argument("--content", "-c", help="Content to moderate (uses stdin if not provided)")
    moderate_parser.add_argument("--file", "-f", help="Path to file containing content to moderate")
    
    # Code command
    code_parser = subparsers.add_parser("code", help="Generate code with specialized models")
    code_parser.add_argument("--model", "-m", default="deepseek-r1-distill-llama-70b", help="Code model ID")
    code_parser.add_argument("--temperature", "-t", type=float, default=0.1, help="Sampling temperature")
    code_parser.add_argument("--prompt", "-p", help="Code description (uses interactive mode if not provided)")
    code_parser.add_argument("--file", "-f", help="Save generated code to file")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available Groq models")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Display version information")
    
    args = parser.parse_args()
    
    # Set up GroqMagic client
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print(f"{EMOJI['error']} GROQ_API_KEY environment variable not set.")
            print("Please set it using: export GROQ_API_KEY=your_api_key")
            sys.exit(1)
            
        client = GroqMagic(api_key=api_key)
        
        # Process commands
        if args.command == "chat":
            if args.prompt:
                # Use provided prompt
                messages = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                messages.append({"role": "user", "content": args.prompt})
                
                client.chat(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    stream=args.stream,
                    conversation_id=args.conversation,
                )
                print()  # Add newline after streaming output
            else:
                # Interactive chat mode
                print(f"{EMOJI['sparkles']} GroqMagic Chat Mode - Model: {args.model}")
                print("Type 'exit' or 'quit' to end the conversation.")
                print("Type 'clear' to clear conversation history.")
                
                conversation_id = args.conversation or datetime.now().strftime("%Y%m%d%H%M%S")
                
                # Add system message if provided
                if args.system:
                    print(f"{EMOJI['light']} Using system message: {args.system}")
                    client.chat(
                        model=args.model,
                        messages=[{"role": "system", "content": args.system}],
                        conversation_id=conversation_id,
                        stream=False,
                    )
                
                while True:
                    try:
                        user_input = input(f"\n{EMOJI['rocket']} You: ")
                        
                        if user_input.lower() in ["exit", "quit"]:
                            print(f"{EMOJI['check']} Ending conversation.")
                            break
                        elif user_input.lower() == "clear":
                            client.clear_history(conversation_id)
                            print(f"{EMOJI['check']} Conversation history cleared.")
                            continue
                        
                        print(f"\n{EMOJI['brain']} Assistant: ", end="")
                        
                        client.chat(
                            model=args.model,
                            messages=[{"role": "user", "content": user_input}],
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            top_p=args.top_p,
                            stream=True,
                            conversation_id=conversation_id,
                        )
                        print()  # Add newline after streaming output
                        
                    except KeyboardInterrupt:
                        print(f"\n{EMOJI['check']} Ending conversation.")
                        break
                    except Exception as e:
                        print(f"\n{EMOJI['error']} Error: {str(e)}")
        
        elif args.command == "vision":
            if not os.path.exists(args.image):
                print(f"{EMOJI['error']} Image file not found: {args.image}")
                sys.exit(1)
                
            print(f"{EMOJI['camera']} Analyzing image with model: {args.model}")
            
            messages = [
                {
                    "role": "user",
                    "content": args.prompt,
                    "image_path": args.image
                }
            ]
            
            response = client.chat(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                stream=True,
            )
            print()  # Add newline after streaming output
        
        elif args.command == "transcribe":
            if not os.path.exists(args.file):
                print(f"{EMOJI['error']} Audio file not found: {args.file}")
                sys.exit(1)
                
            print(f"{EMOJI['sound']} Transcribing audio with model: {args.model}")
            
            result = client.transcribe(
                audio_file=args.file,
                model=args.model,
                response_format=args.format,
                language=args.language,
            )
            
            if args.format == "text":
                print(f"\n{EMOJI['check']} Transcription:\n")
                print(result)
            else:
                print(f"\n{EMOJI['check']} Transcription result:\n")
                print(json.dumps(result, indent=2))
        
        elif args.command == "moderate":
            content = None
            
            if args.content:
                content = args.content
            elif args.file:
                if not os.path.exists(args.file):
                    print(f"{EMOJI['error']} File not found: {args.file}")
                    sys.exit(1)
                    
                with open(args.file, "r") as f:
                    content = f.read()
            else:
                print(f"{EMOJI['shield']} Enter content to moderate (Ctrl+D to finish):")
                content = sys.stdin.read()
            
            if not content or content.strip() == "":
                print(f"{EMOJI['error']} No content provided for moderation.")
                sys.exit(1)
            
            print(f"{EMOJI['shield']} Moderating content with model: {args.model}")
            
            result = client.moderate(
                content=content,
                model=args.model,
            )
            
            print(f"\n{EMOJI['check']} Moderation result:\n")
            print(json.dumps(result, indent=2))
        
        elif args.command == "code":
            if args.prompt:
                prompt = args.prompt
            else:
                print(f"{EMOJI['code']} Enter code description (Ctrl+D to finish):")
                prompt = sys.stdin.read()
            
            if not prompt or prompt.strip() == "":
                print(f"{EMOJI['error']} No code description provided.")
                sys.exit(1)
            
            print(f"{EMOJI['code']} Generating code with model: {args.model}")
            
            code = client.generate_code(
                prompt=prompt,
                model=args.model,
                temperature=args.temperature,
                stream=True,
            )
            
            if args.file:
                with open(args.file, "w") as f:
                    f.write(code)
                print(f"\n{EMOJI['check']} Code saved to: {args.file}")
        
        elif args.command == "list":
            models = client.list_models()
            print(client.format_model_list(models))
        
        elif args.command == "version":
            print(f"{EMOJI['sparkles']} GroqMagic v1.0.0")
            print("A unified interface for Groq AI models")
            print("https://github.com/yourusername/groq-magic")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"{EMOJI['error']} Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli()