# Comprehensive Guide to Ollama Integration in Python Applications

This guide provides patterns and best practices for integrating Ollama into your Python applications, with a focus on robust handling of API responses and structured data.

## Table of Contents

- [Basic Setup](#basic-setup)
- [Core Integration Patterns](#core-integration-patterns)
  - [Synchronous Requests](#synchronous-requests)
  - [Asynchronous (Threaded) Requests](#asynchronous-threaded-requests)
  - [PyQt Integration](#pyqt-integration)
- [Handling Responses](#handling-responses)
  - [Basic Response Extraction](#basic-response-extraction)
  - [Structured Data Extraction](#structured-data-extraction)
  - [JSON Extraction Patterns](#json-extraction-patterns)
- [Advanced Techniques](#advanced-techniques)
  - [Conversation Context Management](#conversation-context-management)
  - [Prompt Engineering for Structured Data](#prompt-engineering-for-structured-data)
  - [Error Recovery Strategies](#error-recovery-strategies)
- [Complete Implementation Examples](#complete-implementation-examples)
  - [CLI Application](#cli-application)
  - [GUI Application (PyQt)](#gui-application-pyqt)
  - [Web Application (FastAPI)](#web-application-fastapi)
- [Best Practices](#best-practices)

## Basic Setup

First, install the Ollama Python client:

```bash
pip install ollama
```

Ensure you have the Ollama server running locally or specify a remote endpoint.

```python
import ollama

# Optional: Configure a custom endpoint
# ollama.set_host("http://localhost:11434")
```

## Core Integration Patterns

### Synchronous Requests

The simplest way to use Ollama (but will block your application):

```python
import ollama

# Simple generation
response = ollama.generate(model="llama3", prompt="Explain quantum computing")
print(response['response'])

# Chat with history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I make pancakes?"}
]
response = ollama.chat(model="llama3", messages=messages)
print(response['message']['content'])
```

### Asynchronous (Threaded) Requests

For applications that need to stay responsive:

```python
import threading
import ollama

class OllamaThread(threading.Thread):
    def __init__(self, model="llama3", prompt="", context=None, callback=None):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.context = context or []
        self.callback = callback
        self.response = None
        self.error = None
        
    def run(self):
        try:
            if self.context:
                # Using chat with context
                self.response = ollama.chat(
                    model=self.model,
                    messages=self.context + [{"role": "user", "content": self.prompt}]
                )
            else:
                # Using generate without context
                self.response = ollama.generate(model=self.model, prompt=self.prompt)
                
            if self.callback:
                self.callback(self.response)
                
        except Exception as e:
            self.error = str(e)
            print(f"Ollama error: {e}")
            if self.callback:
                self.callback({"error": str(e)})

# Example usage:
def handle_response(response):
    if "error" in response:
        print(f"Error: {response['error']}")
        return
        
    content = response.get('message', {}).get('content', '') or response.get('response', '')
    print(f"Response: {content}")

# Start a request
thread = OllamaThread(
    model="llama3", 
    prompt="What is the capital of France?",
    callback=handle_response
)
thread.start()
```

### PyQt Integration

For desktop applications using PyQt:

```python
from PyQt6.QtCore import QThread, pyqtSignal

class OllamaThread(QThread):
    response_ready = pyqtSignal(object)
    
    def __init__(self, model="llama3", prompt="", context=None):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.context = context or []
        
    def run(self):
        try:
            if self.context:
                # Using chat with context
                response = ollama.chat(
                    model=self.model,
                    messages=self.context + [{"role": "user", "content": self.prompt}]
                )
                self.response_ready.emit(response)
            else:
                # Using generate without context
                response = ollama.generate(model=self.model, prompt=self.prompt)
                self.response_ready.emit(response)
        except Exception as e:
            print(f"Ollama error: {e}")
            self.response_ready.emit({"error": str(e)})

# Usage in a PyQt class:
def generate_content(self):
    self.ollama_thread = OllamaThread(model="llama3", prompt="Tell me a joke")
    self.ollama_thread.response_ready.connect(self.handle_response)
    self.ollama_thread.start()

def handle_response(self, response):
    if "error" in response:
        self.display_error(response["error"])
        return
        
    content = response.get('message', {}).get('content', '') or response.get('response', '')
    self.display_content(content)
```

## Handling Responses

### Basic Response Extraction

The response structure differs between the `generate` and `chat` APIs:

```python
def extract_content(response):
    """Extract content from either generate or chat API response"""
    if "message" in response:
        # Chat API response format
        return response.get('message', {}).get('content', '')
    else:
        # Generate API response format
        return response.get('response', '')
```

### Structured Data Extraction

When you need the LLM to return structured data (like JSON):

```python
def extract_structured_data(response, format="json"):
    """Extract and parse structured data from LLM response"""
    content = extract_content(response)
    
    if not content:
        return None
        
    if format.lower() == "json":
        return extract_json(content)
    # Add other formats as needed (XML, YAML, etc.)
    
    return None
```

### JSON Extraction Patterns

This is the most robust way to extract JSON from LLM responses:

```python
import json
import re

def extract_json(text):
    """
    Extract JSON from text, handling various ways an LLM might format it:
    1. As a markdown code block
    2. As raw JSON
    3. With extra text before/after the JSON
    """
    try:
        # First try: Look for JSON in markdown code blocks
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        
        if matches:
            # Try each code block until one parses successfully
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
        
        # Second try: Look for JSON objects enclosed in curly braces
        json_pattern = r'(\{[\s\S]*\})'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            # Try each potential JSON object
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
                    
        # Third try: Try the entire response as JSON
        return json.loads(text.strip())
        
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

# Enhanced version with fallback for arrays
def extract_json_robust(text):
    """Even more robust JSON extraction with array support"""
    try:
        # Try the standard extraction first
        result = extract_json(text)
        if result:
            return result
            
        # Fallback: Look for JSON arrays
        array_pattern = r'(\[[\s\S]*\])'
        matches = re.findall(array_pattern, text, re.DOTALL)
        
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
                    
        return None
        
    except Exception as e:
        print(f"Error in robust JSON extraction: {e}")
        return None
```

## Advanced Techniques

### Conversation Context Management

Managing context for coherent conversations:

```python
class ConversationManager:
    def __init__(self, system_prompt="You are a helpful assistant.", max_history=10):
        self.context = [{"role": "system", "content": system_prompt}]
        self.max_history = max_history
        
    def add_user_message(self, message):
        self.context.append({"role": "user", "content": message})
        self._trim_history()
        
    def add_assistant_message(self, message):
        self.context.append({"role": "assistant", "content": message})
        self._trim_history()
        
    def get_context(self):
        return self.context.copy()
        
    def _trim_history(self):
        """Keep context size manageable by removing oldest messages"""
        if len(self.context) > self.max_history + 1:  # +1 for system prompt
            # Always keep the system prompt
            system_prompt = self.context[0]
            # Keep most recent messages
            self.context = [system_prompt] + self.context[-(self.max_history):]
            
    def clear_history(self, keep_system_prompt=True):
        if keep_system_prompt and self.context:
            self.context = [self.context[0]]
        else:
            self.context = []
```

### Prompt Engineering for Structured Data

To increase the likelihood of getting properly formatted JSON:

```python
def create_json_prompt(base_prompt, schema=None):
    """Create a prompt optimized for JSON responses"""
    prompt = f"{base_prompt}\n\n"
    prompt += "Return your response as a valid JSON object. "
    
    if schema:
        prompt += f"The JSON should follow this schema: {json.dumps(schema)}\n"
    
    prompt += "Wrap the JSON in triple backticks with the json identifier. For example:\n"
    prompt += "```json\n{\"key\": \"value\"}\n```"
    
    return prompt

# Example usage:
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "interests": {"type": "array", "items": {"type": "string"}}
    }
}

prompt = create_json_prompt("Generate a profile for a fictional character", schema)
```

### Error Recovery Strategies

When JSON parsing fails:

```python
def request_with_recovery(prompt, model="llama3", max_attempts=3):
    """Make request with recovery attempts for structured data"""
    for attempt in range(max_attempts):
        # First attempt with regular prompt
        if attempt == 0:
            current_prompt = prompt
        # Subsequent attempts with more explicit instructions
        else:
            current_prompt = f"""
            Your previous response couldn't be parsed as valid JSON.
            
            Please try again and ensure your response is ONLY valid JSON wrapped in 
            triple backticks with the json identifier.
            
            Original request: {prompt}
            """
        
        response = ollama.generate(model=model, prompt=current_prompt)
        json_data = extract_json(response['response'])
        
        if json_data:
            return json_data
            
    # If all attempts fail
    return None
```

## Complete Implementation Examples

### CLI Application

A simple CLI application using Ollama:

```python
import ollama
import argparse
import json
from threading import Thread, Event

def extract_json(text):
    """Extract JSON from text (implementation from above)"""
    # Implementation goes here
    pass

def progress_indicator(stop_event):
    """Show a simple spinner while waiting for response"""
    import sys
    import time
    spinner = ['-', '\\', '|', '/']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rThinking {spinner[i]} ")
        sys.stdout.flush()
        i = (i + 1) % len(spinner)
        time.sleep(0.1)
    sys.stdout.write("\r              \r")

def main():
    parser = argparse.ArgumentParser(description="Ollama CLI")
    parser.add_argument("prompt", help="The prompt to send to Ollama")
    parser.add_argument("--model", default="llama3", help="Model to use")
    parser.add_argument("--json", action="store_true", help="Extract JSON from response")
    args = parser.parse_args()
    
    # Start progress indicator
    stop_event = Event()
    spinner_thread = Thread(target=progress_indicator, args=(stop_event,))
    spinner_thread.start()
    
    try:
        response = ollama.generate(model=args.model, prompt=args.prompt)
        stop_event.set()  # Stop spinner
        spinner_thread.join()
        
        content = response['response']
        
        if args.json:
            json_data = extract_json(content)
            if json_data:
                print(json.dumps(json_data, indent=2))
            else:
                print("Could not extract valid JSON from response")
                print("\nRaw response:")
                print(content)
        else:
            print(content)
            
    except Exception as e:
        stop_event.set()  # Stop spinner
        spinner_thread.join()
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

### GUI Application (PyQt)

A minimal PyQt application with Ollama integration:

```python
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal

import ollama
import json
import re

class OllamaThread(QThread):
    response_ready = pyqtSignal(object)
    
    def __init__(self, model="llama3", prompt="", context=None):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.context = context or []
        
    def run(self):
        try:
            if self.context:
                response = ollama.chat(
                    model=self.model,
                    messages=self.context + [{"role": "user", "content": self.prompt}]
                )
            else:
                response = ollama.generate(model=self.model, prompt=self.prompt)
            self.response_ready.emit(response)
        except Exception as e:
            self.response_ready.emit({"error": str(e)})

def extract_json(text):
    """Extract JSON from text (implementation from above)"""
    # Implementation goes here
    pass

class OllamaUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.context = [{"role": "system", "content": "You are a helpful assistant."}]
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Ollama UI')
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["llama3", "mistral", "gemma"])
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Input area
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("Enter your prompt here...")
        layout.addWidget(self.input_box)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_prompt)
        
        self.json_btn = QPushButton("Extract JSON")
        self.json_btn.clicked.connect(self.extract_json_from_response)
        self.json_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_chat)
        
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.json_btn)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)
        
        # Output area
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        layout.addWidget(self.output_box)
        
        # Status bar
        self.statusBar().showMessage('Ready')
        
        self.setCentralWidget(main_widget)
        
    def send_prompt(self):
        prompt = self.input_box.toPlainText().strip()
        if not prompt:
            return
            
        model = self.model_combo.currentText()
        self.send_btn.setEnabled(False)
        self.statusBar().showMessage('Waiting for response...')
        
        # Add user message to output
        self.output_box.append(f"<b>You:</b> {prompt}")
        
        # Send request via thread
        self.ollama_thread = OllamaThread(model=model, prompt=prompt, context=self.context)
        self.ollama_thread.response_ready.connect(self.handle_response)
        self.ollama_thread.start()
        
    def handle_response(self, response):
        self.send_btn.setEnabled(True)
        
        if "error" in response:
            self.statusBar().showMessage(f"Error: {response['error']}")
            self.output_box.append(f"<b>Error:</b> {response['error']}")
            return
            
        # Extract content based on response type
        if "message" in response:
            content = response.get('message', {}).get('content', '')
            
            # Update context for chat
            self.context.append({"role": "user", "content": self.ollama_thread.prompt})
            self.context.append({"role": "assistant", "content": content})
            
            # Limit context size
            if len(self.context) > 11:  # Keep system + 10 messages
                system_msg = self.context[0]
                self.context = [system_msg] + self.context[-10:]
        else:
            content = response.get('response', '')
        
        # Update UI
        self.output_box.append(f"<b>Assistant:</b> {content}")
        self.statusBar().showMessage('Response received')
        self.json_btn.setEnabled(True)
        self.last_response = content
        
    def extract_json_from_response(self):
        json_data = extract_json(self.last_response)
        if json_data:
            self.output_box.append("<b>Extracted JSON:</b>")
            self.output_box.append(f"<pre>{json.dumps(json_data, indent=2)}</pre>")
        else:
            self.output_box.append("<b>Error:</b> Could not extract valid JSON from response")
            
    def clear_chat(self):
        self.output_box.clear()
        self.input_box.clear()
        # Keep only the system message in context
        self.context = [self.context[0]] if self.context else []
        self.json_btn.setEnabled(False)
        self.statusBar().showMessage('Chat cleared')

def main():
    app = QApplication(sys.argv)
    window = OllamaUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

### Web Application (FastAPI)

A simple FastAPI application using Ollama:

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import ollama
import json
import asyncio
import uuid

app = FastAPI(title="Ollama API Wrapper")

# Store for ongoing requests
requests = {}

def extract_json(text):
    """Extract JSON from text (implementation from above)"""
    # Implementation goes here
    pass

class PromptRequest(BaseModel):
    prompt: str
    model: str = "llama3"
    extract_json: bool = False
    context: Optional[List[Dict[str, str]]] = None

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "llama3"
    extract_json: bool = False

@app.post("/generate")
async def generate(request: PromptRequest, background_tasks: BackgroundTasks):
    """Generate a response using the Ollama generate API"""
    request_id = str(uuid.uuid4())
    requests[request_id] = {"status": "processing", "result": None}
    
    background_tasks.add_task(process_generate_request, request_id, request)
    
    return {"request_id": request_id, "status": "processing"}

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Generate a response using the Ollama chat API"""
    request_id = str(uuid.uuid4())
    requests[request_id] = {"status": "processing", "result": None}
    
    background_tasks.add_task(process_chat_request, request_id, request)
    
    return {"request_id": request_id, "status": "processing"}

@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Check the status of a request"""
    if request_id not in requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return requests[request_id]

async def process_generate_request(request_id: str, request_data: PromptRequest):
    """Process a generation request in the background"""
    try:
        response = ollama.generate(model=request_data.model, prompt=request_data.prompt)
        
        result = {"raw_response": response["response"]}
        
        if request_data.extract_json:
            json_data = extract_json(response["response"])
            if json_data:
                result["json"] = json_data
        
        requests[request_id] = {"status": "completed", "result": result}
        
    except Exception as e:
        requests[request_id] = {"status": "error", "error": str(e)}

async def process_chat_request(request_id: str, request_data: ChatRequest):
    """Process a chat request in the background"""
    try:
        response = ollama.chat(model=request_data.model, messages=request_data.messages)
        
        content = response["message"]["content"]
        result = {"raw_response": content}
        
        if request_data.extract_json:
            json_data = extract_json(content)
            if json_data:
                result["json"] = json_data
        
        requests[request_id] = {"status": "completed", "result": result}
        
    except Exception as e:
        requests[request_id] = {"status": "error", "error": str(e)}

# Clean up old requests every hour
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_requests())

async def cleanup_old_requests():
    while True:
        await asyncio.sleep(3600)  # 1 hour
        # Implementation of cleanup logic
```

## Best Practices

1. **Always Use Threading/Async**: 
   - Never make Ollama API calls directly in your main application thread
   - Use background threads, asyncio, or worker processes

2. **Prompt Engineering**:
   - Be explicit about response format requirements
   - Provide examples of desired output format
   - For JSON, specify the schema when possible

3. **Robust Parsing**:
   - Always use the JSON extraction patterns shown above
   - Have fallback strategies for when parsing fails
   - Consider implementing retry logic with clarified prompts

4. **Error Handling**:
   - Handle API errors gracefully
   - Implement timeouts for long-running requests
   - Have fallback content for when the AI fails

5. **Context Management**:
   - Keep context size reasonable (typically 5-10 message pairs)
   - Include a good system prompt for consistent behavior
   - Consider context window limitations of your model

6. **Security Considerations**:
   - Validate and sanitize user inputs before sending to Ollama
   - Consider implementing content filtering on responses
   - Be cautious with executing any code generated by the LLM

7. **Performance Optimization**:
   - Use smaller models for simple tasks
   - Implement caching for common queries
   - Consider batching multiple requests when appropriate
