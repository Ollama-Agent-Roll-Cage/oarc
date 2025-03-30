"""_api.py - ollamaAgentRollCage API for the ollamaChatbotWizard and ollamaAgentRollCage toolkit.

        ollamaChatbotWizard, is an opensource toolkit api for speech to text, text to speech 
    commands, multi-modal agent building with local LLM api's, including tools such as ollama, 
    transformers, keras, as well as closed source api endpoint integration such as openAI, 
    anthropic, groq, and more!
    
    ===============================================================================================
    
         has its own chatbot agent endpoint which you can find in the fastAPI at the bottom 
    of this file. This custom api is what empowers oarc to bundle/wrap AI models & other api endpoints 
    into one cohesive agent including the following models;
    
    Ollama -
        Llama: Text to Text 
        LLaVA: Text & Image to Text
        
    CoquiTTS -
        XTTSv2: Non-Emotive Transformer Text to Speech, With Custom Finetuned Voices
        Bark: Emotional Diffusion Text to Speech Model
        
    F5_TTS -
        Emotional TTS model, With Custom Finetuned Voices (coming soon) 
        
    YoloVX - 
        Object Recognition within image & video streams, providing bounding box location data.
        Supports YoloV6, YoloV7, YoloV8, and beyond! I would suggest YoloV8 seems to have the 
        highest accuracy. 
        
    Whisper -
        Speech to Text recognition, allowing the user to interface with any model directly
        using a local whisper model.
        
    Google python speech-recognition -
        A free alternative Speech to Text offered by google, powered by their api servers, this
        STT api is a good alternative especially if you need to offload the speech recognition 
        to the google servers due to your computers limitations.
        
    Musetalk -
        A local lypc sync, Avatar Image & Audio to Video model. Allowing the chatbot agent to
        generate in real time, the Avatar lypc sync for the current chatbot agent in OARC.
        
    ===============================================================================================
    
        This software was designed by Leo Borcherding with the intent of creating 
    an easy to use ai interface for anyone, through Speech to Text and Text to Speech.
        
        With ollamaChatbotWizard we can provide hands free access to LLM data. 
    This has a host of applications and I want to bring this software to users 
    suffering from blindness/vision loss, and children suffering from austism spectrum 
    disorder as way for learning and expanding communication and speech. 
    
        The C3PO ai is a great imaginary friend! I could envision myself 
    talking to him all day telling me stories about a land far far away! 
    This makes learning fun and accessible! Children would be directly 
    rewarded for better speech as the ai responds to subtle differences 
    in language ultimately educating them without them realizing it.

    Development for this software was started on: 4/20/2024 
    All users have the right to develop and distribute ollama agent roll cage,
    with proper citation of the developers and repositories. Be weary that some
    software may not be licensed for commerical use.

    By: Leo Borcherding, 4/20/2024
        on github @ 
            leoleojames1/ollama_agent_roll_cage
"""

#TODO CREATE API FOR SMOL AGENTS SPEECH, OLLAMA, AND VISION TOOLS
#TODO CREATE API FOR  WIZARD AGENT CONFIG LOADING FOR STORED AGENT CORE JSON TEMPLATE CONFIGS
#TODO BUILD  pip install such that from .speechtoSpeech import textToSpeech, or speechtoText etc.
#TODO so essentially all of the tools in the multimodal pip install  package can be written into scripts, 
# or you can access the entire  api for loading agent configs, HANDLE WITH GRACE, BUILD WITH CARE, TAKE IT SLOW THIS IS A MARATHON NOT A SPRINT.

import logging
import os
from pprint import pformat

from pydantic import BaseModel

from oarc.ollamaUtils.create_convert_manager import create_convert_manager
from oarc.ollamaUtils import model_write_class
from oarc.ollamaUtils.ollamaCommands import ollamaCommands

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrettyFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, (dict, list)):
            record.msg = f"\n{pformat(record.msg, indent=2, width=80)}"
        return super().format(record)
   
class ModelRequest(BaseModel):
    model_name: str
    agent_id: str
     
class AgentAccess(BaseModel):
    agent_id: str
    
class SpellLoader():
    """Class for loading spells for the ollamaAgentRollCage"""
    def __init__(self):
        self.initializeBasePaths()
        self.initializeSpells()
        
    def initializeBasePaths(self):
        """Initialize the base file path structure"""
        try:
            # Get base directories
            self.current_dir = os.getcwd()
            self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
            
            # Environment variables for model directories
            self.model_git_dir = os.getenv('OARC_MODEL_GIT')
            self.ollama_models_dir = os.getenv('OLLAMA_MODELS')
            self.hf_cache_dir = os.getenv('HF_HOME', os.path.join(self.model_git_dir, 'huggingface'))
            
            # Validate environment variables
            if not self.model_git_dir:
                raise EnvironmentError("OARC_MODEL_GIT environment variable not set")
            if not self.ollama_models_dir:
                raise EnvironmentError("OLLAMA_MODELS environment variable not set")
            
            # Initialize base path structure
            #TODO UPDATE TO STORE ALL NEW MODEL PATHS CORRECTLY and CREATE CENTRAL MODEL MANAGER FOR OLLAMA AND HUGGING FACE HUB
            
            self.pathLibrary = {
                # Main directories
                'current_dir': self.current_dir,
                'parent_dir': self.parent_dir,
                'model_git_dir': self.model_git_dir,
                'ollama_models_dir': self.ollama_models_dir,
                
                # Model directories 
                'huggingface_models': {
                    'base_dir': os.path.join(self.model_git_dir, 'huggingface'),
                    'whisper': os.path.join(self.model_git_dir, 'huggingface', 'whisper'),
                    'xtts': os.path.join(self.model_git_dir, 'huggingface', 'xtts'),
                    'yolo': os.path.join(self.model_git_dir, 'huggingface', 'yolo'),
                    'llm': os.path.join(self.model_git_dir, 'huggingface', 'llm')
                },
                
                # Agent directories
                'ignored_agents_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredAgents'),
                'agent_files_dir': os.path.join(self.model_git_dir, 'agentFiles', 'publicAgents'),
                'ignored_agentfiles': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredAgentfiles'),
                'public_agentfiles': os.path.join(self.model_git_dir, 'agentFiles', 'publicAgentfiles'),
                
                # Pipeline directories
                'ignored_pipeline_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline'),
                'llava_library_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline', 'llavaLibrary'),
                'conversation_library_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline', 'conversationLibrary'),
                
                # Data constructor directories
                'image_set_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline', 'dataConstructor', 'imageSet'),
                'video_set_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline', 'dataConstructor', 'videoSet'),
                
                # Speech directories
                'speech_library_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline', 'speechLibrary'),
                'recognize_speech_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'recognizeSpeech'),
                'generate_speech_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'generateSpeech'),
                'tts_voice_ref_wav_pack_dir': os.path.join(self.model_git_dir, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'publicVoiceReferencePack'),
            }
            logger.info("Base paths initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing base paths: {e}")
            raise 

    def initializeSpells(self):
        """Initialize all spell classes for the chatbot wizard"""
        try:
            # initialize ollama commands
            self.ollamaCommandInstance = ollamaCommands()
            # Write model files
            self.model_write_class_instance = model_write_class(self.pathLibrary)
            # Create model manager
            self.create_convert_manager_instance = create_convert_manager(self.pathLibrary)
            # TTS processor (initialize as None, will be created when needed)
            self.tts_processor_instance = None
            
            logger.info("Spells initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing spells: {e}")
            raise
