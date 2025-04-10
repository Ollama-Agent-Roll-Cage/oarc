# Ollama Integration - Condensed Guide

This guide provides a concise implementation pattern for integrating Ollama into Python applications with non-blocking threading and robust JSON parsing.

## Basic Threading Implementation

```python
import json
import threading
import ollama

class OllamaThread(threading.Thread):
    """Thread for handling Ollama requests without blocking the main application"""
    
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
```

## Response Handling

```python
def handle_response(response):
    """Extract content from Ollama response"""
    if "error" in response:
        print(f"Error: {response['error']}")
        return None
        
    if "message" in response:
        # Chat API response format
        return response.get('message', {}).get('content', '')
    else:
        # Generate API response format
        return response.get('response', '')
```

## JSON Extraction

```python
def extract_json(text):
    """Extract JSON from LLM response text"""
    import json
    import re
    
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
```

## JSON-Optimized Prompting

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
```

## Complete Usage Example

```python
# Context example for chat
context = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about the solar system."},
    {"role": "assistant", "content": "The solar system consists of the Sun and objects that orbit it..."}
]

def process_json_response(response):
    """Process a response expecting JSON data"""
    content = handle_response(response)
    if not content:
        return
        
    json_data = extract_json(content)
    if json_data:
        print("Successfully extracted JSON:")
        print(json.dumps(json_data, indent=2))
    else:
        print("Failed to extract JSON from response")
        print("Raw content:", content)

# Example: Request with JSON response
schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
json_prompt = create_json_prompt("Generate a fictional character profile", schema)

thread = OllamaThread(
    model="llama3",
    prompt=json_prompt,
    callback=process_json_response
)
thread.start()

# Let the thread complete (in a real app you might wait differently)
thread.join()
```

## Context Management Pattern

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
```

## Best Practices

1. **Always use threading** to prevent blocking the main application
2. **Handle both API formats** (generate vs chat) in your response processing
3. **Use robust JSON extraction** with multiple fallback strategies
4. **Include clear instructions** in prompts when requesting structured data
5. **Implement error handling** for network issues and malformed responses
6. **Manage conversation context** to maintain coherent multi-turn dialogues
7. **Consider retries** when JSON parsing fails by sending a clarification prompt
