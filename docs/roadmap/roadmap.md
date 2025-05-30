
# OARC & Agent Chef: AI Model Development and Deployment Platform

## Table of Contents
1. [Introduction](#introduction)
2. [Key Components](#key-components)
   - [Agent Chef](#agent-chef)
   - [Wizard Training](#wizard-training)
   - [Ollama Agent Roll Cage (OARC)](#ollama-agent-roll-cage-oarc)
3. [Features](#features)
4. [Use Cases](#use-cases)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

OARC & Agent Chef is a comprehensive AI platform designed for continuous dataset building, model fine-tuning, and deploying advanced AI agents. This ecosystem enables the creation, training, and deployment of sophisticated AI models with multimodal capabilities, including speech-to-speech and vision-based interactions.

## Key Components

### Agent Chef

Agent Chef is a powerful tool for dataset refinement, structuring, and generation, crucial for fine-tuning AI models. It offers:

- Dataset scraping and processing
- Data construction and formatting
- Dataset augmentation
- Synthetic data generation for training

[GitHub Repository](https://github.com/Leoleojames1/Agent_Chef)

### Wizard Academy

Wizard Academy is our web-based platform for training and fine-tuning large language models, as well as other machine learning models. 

Key features include:
- Support for finetuning models up to 405B parameters
- Utilization of high-performance GPUs (H100s and 3090s)
- Implementation of LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Finetuning of 3B models and other sizes
- Training certain models from scratch
- Other model tuning options
  -> coqui
  -> cycleGAN
  -> stable diffusion & video
  -> agent chef -> multimodal dataset pipelines
  
### Ollama Agent Roll Cage (OARC)

OARC serves as the deployment and hosting platform for models created with Agent Chef and Wizard Training. It provides:

- An agentic action space for multimodal agent deployment
- Support for speech-to-speech and vision-based AI agents
- Avatar integration for visual representation of AI agents
- Local deployment options
- Web-based interface for GPU rental and hosting

## Features

- Continuous dataset building and augmentation
- Fine-tuning of large language models
- Multimodal agent creation (text, speech, vision)
- Real-time audio and text interactions
- Vision-based agent capabilities
- Avatar integration for AI agents
- LaTeX rendering for mathematical formulas
- Text-to-image, image-to-text, and image-to-video transformations
- Keyboard navigation and DuckDuckGo search integration

## Use Cases

1. **AI Model Development**
   - Continuous dataset creation and refinement
   - Fine-tuning of language models for specific tasks
   - Development of multimodal AI agents

2. **Speech-to-Speech AI Agents**
   - Virtual assistants with natural language processing
   - Real-time language translation
   - Voice-controlled systems

3. **Vision-Based AI Applications**
   - Object recognition and scene understanding
   - Visual question answering
   - Image and video analysis

4. **Educational AI**
   - Intelligent tutoring systems
   - Personalized learning assistants
   - Interactive educational simulations

## ==========================================================================================================
## ORIGINAL Development Cycle Roadmap FOR OARC v1, OARC v2 will follow in the same footsteps with new versioning
## ==========================================================================================================

## Updates 0.1.9 -> 0.3.0 - Development Cycle - New Commands, Features, & Optimizations:
```diff
- ***UPCOMING SOON***
```

### Update 0.1.9: Chatbot script, TTS processor class, Model /Swap
```diff
+ ***PUSHED TO GITHUB***
```
- /save - save current conversation to main history file
- /load - load the main conversation history file for long term intermodel conversation history keep seperate from /save as and /load as and only use to keep a long term history of your entire ollama agent base for specified history.
- /quit - break the main python loop and return to command line
- /swap - swap the current model with the specified model
  
### Update 0.2.1: Custom Agent /Create Automation 
```diff
+ ***PUSHED TO GITHUB***
```
- /create -> user input or voice -> "agent name" "SYM PROMPT" -> uses currently loaded model and the defined system prompt in speech or text to create a new agent with your own specific customizations

### Update 0.2.2: Speech Optimization
```diff
+ ***PUSHED TO GITHUB***
```
- "Smart Wait Length Timer": add method to manage the wait time for text to speech generation by controlling sd.wait() based on the token length of the next sentence. If tokens in next sentence are longer than current sentence, start processing next audio generation, if next sentence is not longer than current sentence, dont start text to speech generation otherwise there will be an overide
- "Wave File Storage Library": Found a solution to storing the audio wav files seperatley such that an overide of the current audio out is not possible: https://github.com/coqui-ai/TTS/discussions/2988
- SYM PROMPT: Template sentence structure such as periods and end marks like <> model response </> for intelligent output formats designs specifically with ollama_agent_roll_cage in mind
- filter unique strings such as `` , also manage bullet points for 1. 2. 3. 4., as these are not the end of sentence periods, maybe send the response to another llm for query boost and sentence filtering

### Update 0.2.3: Speech modes leap, listen, speech on/off 
```diff
+ ***PUSHED TO GITHUB***
```
- /speech on/off -> swap between Speech to Speech (STS) & Text to Text (TTT) interface
- /listen on/off -> turn off speech to text recognition, text to speech generation listen mode only
- /leap on/off -> turn off text to speech audio generation, speech to text recognition only, for speed interface
  
### Update 0.2.4: Agent voice swap & Conversation History Library 
```diff
+ ***PUSHED TO GITHUB***
```
- /voice swap {name} -> user input & voice? -> swap the current audio reference wav file to modify the agent's reference voice
- /save as -> user input & voice? -> "name" -> save the current conversation history with a name to the current model folder
- get model name, conversation name, and store in custom directory in conversation library for each model in ollama_list.cmd
- /load as -> user input & voice? -> "name" -> load selected conversation, spaces are replaces with underscores during voice command name save

### Update 0.2.5: Custom Xtts Model Training, Real Time Latex Rendering
```diff
@@ ***NEXT UPDATE*** @@
```
- coqui text to speech -> xtts model training with xtts-finetune-webui, train an xtts voice model with 2:00-10:00 minutes of audio data for a more accurate voice recording.
- custom xtts webui: https://github.com/aitrepreneur/xtts-finetune-webui
- borch/phi3_latex -> modified system prompt for smart latex document output for simpler regex parsing of the model response tokens.
  
- /latex on/off -> run latex real time render automation for current conversation when using a latex tuned model such as borch/phi3_latex or utilizing regex to splice out the latex and render the current formula document. This will be updated as the conversation continues and will contain the current prompts latex, where as the .tex file will contain the entire latex conversation history.
- /latex save -> save spliced and built latex file to .tex file
- /latex run -> run saved latex file with pdf generate command and open generated pdf
- add latex AI model citation section for citation automation, as well as website citation via duck duck go search api

### Update 0.2.5: Langchain, Function Caller, Ollama Chat Api with variable model delimiters
Optimized chat history, send prompt method, and model prompt template with the ollama python package:

https://pypi.org/project/ollama/

https://www.langchain.com/

RAG FROM SCRATCH: 

https://github.com/langchain-ai/rag-from-scratch

design custom server and api for OARC, then plug api into Open Web UI:

https://docs.openwebui.com/

train borch/phi3_latex model fine tune, with currated latex dataset for different math topic groups:
https://github.com/unslothai/unsloth

train sentiment detection for latex output to create custom math topic group analyizer model (calulus, complex analysis, vector analysis, etc):

Implement sebdg emotion classifier for routing functions:
https://huggingface.co/sebdg/emotions_classifier

### Update 0.2.6: DuckDuckGo API search & voice command function call model
- /search {request} -> send search request to DuckDuckGo free api (no key required) for context lookup
- search query boost automatic
- search query runs on serperate thread and returns the results to the current model.
- add search query digester and summarization model as a preprocessor before prompting the main model.
- /boost -> activate model prompt query boost utilizing secondary model to improve user input requests as an ingest preprocess before prompting the model, call secondary prompt method and run on seperate thread.

- /automatically assume all /{keyword} commands via a function call mediator model
  
### Update 0.2.7: voice clone record, playback wav, mp3, mp4, audiobook, music, movie
- /record -> user input & voice? -> "name" -> record wav file and save to agent or to wav library
- /record as -> user input & voice? -> "name" -> record wav file and save to agent or to wav library
- /clone voice -> call record, save and call /voice to swap voice instantly for instant voice clone transformation from library
  
- /playback -> playback any stored wav file in wav library
- /book audio -> load a book pdf or audiobook wav for playback
- /movie play "name" -> play back named movie mp4 file from library
- /music play "name" -> play back named music mp3 file from library

- RCV -> add audio to audio model for text to speech RVC voice audio tuning
  
### Update 0.2.8: PDF Document Access via RAG implementation
- /rag file on/off -> enable disable pdf & tex rag
- /rag model on/off -> enable disable rag access for model data
  
- PDF to data convert, pdf to latex, pdf to code, pdf image recognition? latex only?
- file & data conversion automations with read write symbol collector
  
- /PDF read -> user input & voice? -> "name" -> digest given pdf for context reference
- /PDF list -> list all pdfs stored in agent library

### Update 0.2.8: ComfyUI Automation with custom LORA &/or SORA
- comfyUI workflow library
- workflows for:
- text to img
- img to vid
- img to img
  
- SD & SD XL base model selection
- lora model library selection
  
- /generate image -> "prompt" -> generate image with custom LORA model
- /generate video -> "prompt" -> generate video with custom SORA model
- /story board -> generate an image for each prompt in the conversation to visualize the content of the conversation
- /generate movie -> generate an mp4 video for each prompt in the conversation to visualize the content of the conversation/story/game

- /generate agent portrait -> using trained video footage generate deepfake for text to speech audio as its being played with corresponding agent profile web camera.
- allow for combination of /generate movie & /generate agent portrait to generate movies with the deepfakes of the agent matching up to the audio generation.
- portrait casting
- lipsync deepfake generation
  
- /recognize video -> activate image recognition for video recording input for functional utility
- /recognize webcam -> activate image recognition for video web cam input for functional utility
  
- Sora directed agent profile deepfake animation
- https://github.com/Stability-AI/generative-models
- Sora directed game animation for games such as "Rick and Morty" portal journey explore endless worlds with video generation.
  
### Update 0.2.9: Smart Conversation, Listen and parse gaps from conversation, lookup data, moderate
- /smart listen 1 -> listens and responds after long pause, parses spaces from gapped chat history and recombines conversation history if for words said while the model is responding
- /smart listen 2 -> listen to the conversation between 2 people, record history, only trigger a response when the most likely human response would occur, i, e, talk short, give human like responses, yet still retain the knowledge of llama3. While 2 users converse, llama3 model learns the conversation flow, and know when stepping in for moderation, fact checking, search results, live in a heated debate where one would want to know the true nature of scientific data, historical data, language data, and all data in the moment of live conversation with agent roll cage
- /moderator -> make roll cage a conversation moderator for 2 different people having a conersation always listing and processing thoughts but never responding until "/yo llama what do you think about that" is asked after activating /moderator.
- /yo llama what do you think about that -> llama3 response for the /moderator chat history as a mediator between 2 people.
- /yo llama pull that up -> a copy of jamie from joe rogan using C3PO voice clone audio reference w/ google api search finds: youtube clips, wiki pedia google results, and explains the point, also screen shares macros with keyboard and/or google youtube wiki search browser. preferably with macro moves for opening complex task and managing operations. -> send to joe rogan and jamie? xD

### Update 0.3.0: On startup run default command setup, create automation job set with cmd automations and mouse/keyboard macros
- /preload command list -> command_list.txt, run desired default commands on "/preload command list" call
- /job set run {name} -> create macro job set with cmd automations and automated keyboard output for mouse and key to automate specific tasks
- /macro on - enabled keyboard macro mode, allowing the agent to exute jobs from voice commands or saved job lists, to automate tasks
- add program spacial recognition view to splice programs into desired spacial locations for the decision model to navigate.
- add agent decision automation for search, if search is relevant use search otherwise dont, then have /search on/off turn this on or off, so duck duck go doesnt return an error for people without internet connection.

### UPDATES 0.3.1 & BEYOND
  
### NOW REGEX PATTERN FACTORY: sentence parser - comprehensive filter
- SYM PROMPT: Template sentence structure such as periods and end marks like <> model response </> for intelligent output formats designs specifically with ollama_agent_roll_cage in mind
- filter unique strings such as `` , also manage bullet points for 1. 2. 3. 4., as these are not the end of sentence periods, maybe send the response to another llm for query boost and sentence filtering

### Beyond
- add ollama_agent_roll_cage_language variant for **German, Spanish, French, Mandarin, Russian, latin? xD, arabic, hebrew, italian, hindi, japanese, portugeuse,** which will include the translations down to the /voice commands and language models.
- /swap language
