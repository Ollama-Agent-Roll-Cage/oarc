"""oarc_mcp.api.py
This file contains the API endpoints for the OARC MCP server.

"""
from fast_mcp_wrapper import FastMCPWrapper

# Initialize the wrapper
fast_mcp_wrapper = FastMCPWrapper()

# Define your tools, resources, and prompts as shown above

# Example endpoint for processing speech
@app.route('/process_speech', methods=['POST'])
async def process_speech_endpoint(request):
    audio_data = request.json.get('audio_data')
    result = await fast_mcp_wrapper.call_tool('process_speech', {'audio_data': audio_data})
    return jsonify(result)

# Example endpoint for interacting with Ollama
@app.route('/interact_with_ollama', methods=['POST'])
async def interact_with_ollama_endpoint(request):
    prompt_text = request.json.get('prompt_text')
    result = await fast_mcp_wrapper.call_tool('interact_with_ollama', {'prompt_text': prompt_text})
    return jsonify(result)

# Example endpoint for multimodal prompting
@app.route('/multimodal_prompt', methods=['POST'])
async def multimodal_prompt_endpoint(request):
    text = request.json.get('text')
    image_data = request.json.get('image_data')
    result = await fast_mcp_wrapper.call_tool('multimodal_prompt', {'text': text, 'image_data': image_data})
    return jsonify(result)

#TODO THESE WILL BE INTEGRATED AT THE SOURCE LEVEL SUCH AS speech_to_text.py, and the above are the direct 
# access functions

""" other_oarc_mcp_utils.py
This file contains utility functions for the OARC MCP server.

"""

# Speech Processing Tool
@fast_mcp_wrapper.tools()
def process_speech(audio_data):
    # Implement your speech processing logic here
    return {"transcription": "Processed transcription"}

# Ollama Interaction Tool
@fast_mcp_wrapper.tools()
def interact_with_ollama(prompt_text):
    # Implement interaction with Ollama API
    response = ollama_api_call(prompt_text)
    return {"response": response}

# Multimodal Prompting Tool
@fast_mcp_wrapper.tools()
def multimodal_prompt(text, image_data=None):
    # Implement logic for handling text and optional image data
    result = handle_multimodal_input(text, image_data)
    return {"result": result}

""" define resources
"""

# Example Resource for Templates
@fast_mcp_wrapper.resources("/templates")
def get_templates():
    return {"template1": "Hello, world!", "template2": "Goodbye!"}

""" defube prompts
"""

# Example Prompt for Generating Responses
@fast_mcp_wrapper.prompts()
def generate_response(prompt_text):
    # Implement logic to generate a response based on the prompt text
    return f"Response to: {prompt_text}"