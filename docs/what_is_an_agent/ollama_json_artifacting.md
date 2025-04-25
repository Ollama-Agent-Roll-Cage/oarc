# Ollama JSON Markdown Artifact Formatting and Handling

This guide focuses on integrating Ollama with PyQt applications, specifically emphasizing how to use markdown formatting with proper JSON code blocks in system prompts to improve structured data responses.

## Core OllamaThread Class

```python
import json
import time
from threading import Thread
from PyQt6.QtCore import QThread, pyqtSignal, QObject

# Import Ollama for AI integration
import ollama

class OllamaThread(QThread):
    """Thread for handling Ollama requests without blocking the main UI"""
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
```

## System Prompts with JSON Markdown Formatting

The key to getting consistent, well-structured JSON responses is to use clear examples of markdown JSON formatting in your system prompts.

### Example System Prompt for JSON Responses

```python
def get_json_system_prompt(self):
    """Return a system prompt optimized for JSON responses"""
    return """
    # JSON Response Guidelines
    
    You are an AI assistant that returns structured data in JSON format.

    ## Format Requirements
    
    - ALWAYS format your responses using markdown code blocks with the json language specifier
    - Use the format shown in the example below:
    
    ```json
    {
      "key": "value",
      "number": 42,
      "boolean": true,
      "array": [1, 2, 3],
      "nested": {
        "inner_key": "inner_value"
      }
    }
    ```
    
    ## JSON Standards
    
    - Use double quotes for keys and string values
    - Do not use trailing commas
    - Use camelCase for property names
    - Ensure all JSON is valid and properly formatted
    
    ## Response Structure
    
    Your responses should always include:
    - A "status" field with value "success" or "error"
    - A "data" object containing the requested information
    - An optional "message" field for human-readable notes
    
    ## Example Response
    
    ```json
    {
      "status": "success",
      "data": {
        "requestedInfo": "value",
        "additionalDetails": ["item1", "item2"]
      },
      "message": "Information retrieved successfully"
    }
    ```
    """
```

## Handling and Extracting JSON from Responses

```python
def extract_json_from_markdown(self, content):
    """Extract and parse JSON from markdown-formatted content"""
    import re
    import json
    
    try:
        # Look specifically for ```json blocks
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        if matches:
            # Return the first valid JSON match
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
        
        # Fallback: Look for any code blocks
        code_block_pattern = r'```\s*([\s\S]*?)\s*```'
        matches = re.findall(code_block_pattern, content, re.DOTALL)
        
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
        
        # Fallback: Try the whole response as JSON
        return json.loads(content.strip())
        
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None
```

## Complete Example Application

```python
class OllamaJsonApp:
    def __init__(self):
        # Initialize context with JSON-optimized system prompt
        self.ai_context = [
            {"role": "system", "content": self.get_json_system_prompt()},
        ]
    
    def request_json_data(self, query, schema=None):
        """Request JSON data from Ollama"""
        # Format the prompt to encourage JSON response
        prompt = self.create_json_optimized_prompt(query, schema)
        
        # Create thread for async request
        self.ollama_thread = OllamaThread(
            model="llama3",
            prompt=prompt,
            context=self.ai_context
        )
        self.ollama_thread.response_ready.connect(self.handle_json_response)
        self.ollama_thread.start()
    
    def create_json_optimized_prompt(self, query, schema=None):
        """Create a prompt optimized for JSON response"""
        prompt = f"{query}\n\n"
        prompt += "Return your response as a valid JSON object. "
        
        if schema:
            prompt += f"The JSON should follow this schema:\n\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        
        prompt += "Make sure to format your entire response as a JSON object wrapped in markdown code blocks with the json language specifier."
        
        return prompt
    
    def handle_json_response(self, response):
        """Handle response and extract JSON"""
        if "error" in response:
            print(f"Error: {response['error']}")
            return None
        
        # Extract content based on response type
        if "message" in response:
            content = response.get('message', {}).get('content', '')
        else:
            content = response.get('response', '')
        
        # Extract JSON from the content
        json_data = self.extract_json_from_markdown(content)
        
        if json_data:
            print("Successfully extracted JSON:")
            print(json.dumps(json_data, indent=2))
            return json_data
        else:
            print("Failed to extract JSON from response")
            print("Raw content:", content)
            return None
```

## JSON Format for Different Queries

To ensure consistent JSON formatting, you can create specific schemas for different query types:

### Schema Examples

```python
# User profile schema
user_profile_schema = {
    "status": "success",
    "data": {
        "name": "string",
        "age": "number",
        "interests": ["string"],
        "contact": {
            "email": "string",
            "phone": "string"
        }
    }
}

# Product information schema
product_schema = {
    "status": "success",
    "data": {
        "name": "string",
        "price": "number",
        "description": "string",
        "specifications": {
            "dimensions": "string",
            "weight": "string",
            "features": ["string"]
        },
        "inStock": "boolean"
    }
}
```

## Example PyQt UI Integration

```python
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel
import sys
import json

class JsonQueryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama JSON Query Tool")
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # Query input
        self.query_label = QLabel("Enter your query:")
        layout.addWidget(self.query_label)
        
        self.query_input = QTextEdit()
        self.query_input.setMaximumHeight(100)
        self.query_input.setPlaceholderText("For example: Generate a profile for a fictional character")
        layout.addWidget(self.query_input)
        
        # Schema input
        self.schema_label = QLabel("JSON Schema (optional):")
        layout.addWidget(self.schema_label)
        
        self.schema_input = QTextEdit()
        self.schema_input.setMaximumHeight(150)
        self.schema_input.setPlaceholderText('{"name": "string", "age": "number", "interests": ["string"]}')
        layout.addWidget(self.schema_input)
        
        # Submit button
        self.submit_button = QPushButton("Get JSON Response")
        self.submit_button.clicked.connect(self.submit_query)
        layout.addWidget(self.submit_button)
        
        # Response display
        self.response_label = QLabel("JSON Response:")
        layout.addWidget(self.response_label)
        
        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        layout.addWidget(self.response_display)
        
        self.setCentralWidget(main_widget)
        
        # Initialize Ollama JSON app
        self.ollama_app = OllamaJsonApp()
        
    def submit_query(self):
        query = self.query_input.toPlainText()
        if not query:
            return
        
        # Parse schema if provided
        schema_text = self.schema_input.toPlainText()
        schema = None
        if schema_text:
            try:
                schema = json.loads(schema_text)
            except json.JSONDecodeError:
                self.response_display.setText("Error: Invalid JSON schema")
                return
        
        # Disable button during processing
        self.submit_button.setEnabled(False)
        self.response_display.setText("Processing request...")
        
        # Request JSON data
        self.ollama_app.request_json_data(query, schema)
        self.ollama_app.ollama_thread.response_ready.connect(self.handle_response)
    
    def handle_response(self, response):
        self.submit_button.setEnabled(True)
        
        if "error" in response:
            self.response_display.setText(f"Error: {response['error']}")
            return
        
        # Extract content and JSON
        if "message" in response:
            content = response.get('message', {}).get('content', '')
        else:
            content = response.get('response', '')
        
        json_data = self.ollama_app.extract_json_from_markdown(content)
        
        if json_data:
            # Display formatted JSON
            self.response_display.setText(json.dumps(json_data, indent=2))
        else:
            # Show raw response if JSON extraction failed
            self.response_display.setText(f"Failed to extract JSON. Raw response:\n\n{content}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JsonQueryWindow()
    window.show()
    sys.exit(app.exec())
```

## Best Practices for JSON Formatting

1. **Explicitly request markdown code blocks with the json identifier**:
   ```
   Return your response as a JSON object wrapped in triple backticks with the json language specifier.
   ```

2. **Provide clear examples** in your system prompt:
   ```
   ```json
   {
     "status": "success",
     "data": {
       "example": "value"
     }
   }
   ```
   ```

3. **Use schemas** to specify the expected structure:
   ```python
   schema = {
       "type": "object",
       "properties": {
           "key1": {"type": "string"},
           "key2": {"type": "number"}
       }
   }
   ```

4. **Implement robust parsing** with multiple fallback strategies:
   - First look for ```json blocks specifically
   - Then try any ``` code blocks
   - Finally attempt to parse the entire response

5. **Use consistent formatting** in your prompts:
   - Prefer camelCase or snake_case consistently
   - Always use a predictable response structure
   - Define required fields explicitly
