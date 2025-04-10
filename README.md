<!-- markdownlint-disable MD033 MD041 MD045 -->
<p align="center">
  <img src="assets/OARC_LOGO_RMBG.png" alt="OARC LOGO" width="250"/>
</p>
<p align="center">
  <a href="https://discord.gg/vksT5csPbd"><img src="assets/Discord Button Ollama v3.png" height="48"></a>
  <a href="https://discord.gg/mNeQZzBHuW"><img src="assets/Discord Button Ollama v4.png" height="48"></a>
  <a href="https://ko-fi.com/theborch"><img src="assets/buy me a coffee button (2).png" height="48"></a>
</p>

# 👽🧙 OARC 🤬🤖

| **Feature**                | **Description**                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| **Multimodal AI**           | Integrates audio, text, vision, and automation workflows.                     |
| **Fine-tuned LLMs**         | Supports advanced language models for custom use cases.                       |
| **Speech & Vision**         | Includes TTS, STT, and YOLO-based object detection.                           |
| **Dataset Tools**           | Collection, generation, augmentation, and cleaning of datasets.               |
| **Search Integration**      | DuckDuckGo, GitHub repo cloning, and web scraping APIs.                       |
| **HuggingFace Support**     | Manage models and datasets seamlessly.                                        |
| **Extensible Design**       | Modular architecture for adding new features.                                 |
| **Developer Tools**         | Editable installation and build-from-source options.                          |
| **GPU Acceleration**        | Optimized for CUDA-enabled devices.                                           |
| **API Services**            | RESTful APIs for external integrations.                                       |
| **Open Source**             | Apache 2.0 license with active community support.                             |

## Installation

Install OARC by following the steps below. Ensure Python 3.10 or 3.11 is installed (Python 3.12+ is not yet supported due to TensorFlow compatibility). For GPU acceleration, verify CUDA drivers are set up. Dependencies, including language models, vision, and audio systems, will be configured automatically.
### Quick install

For fast usage oarc can be installed with pip:
```python
uv pip install oarc
```

### Direct Installation

For development purposes, oarc can be installed by cloning the repository and setting up the uv venv:
```bash
# Clone the repository
git clone https://github.com/Ollama-Agent-Roll-Cage/oarc.git
cd OARC

# Create & activate virtual environment with Python 3.10
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Install environment packages
python -m pip install --upgrade pip setuptools wheel appdirs uv

# Install the package with pip (editable mode)
uv pip install -e .

# Install additional dependencies
oarc setup
```

## Running OARC

```bash
# Activate environment where OARC is installed
oarc <command>
```

## Commands

- `oarc` - Run the main CLI tool
- `oarc setup` - Install all dependencies
- `oarc develop` - Setup developer mode
- `oarc build` - Build from source code
- `oarc publish` - Publish built code to pypi with twine

## Development

```bash
# Install development dependencies
uv pip install -e .[dev]

# Setup OARC development environment
oarc develop
```

### Building from source

```bash
# Build the oarc as a wheel from source
oarc build
```

## Architecture

OARC's modular architecture ensures extensibility and high performance. Core components like APIs, agent management, speech, vision, and dataset pipelines interact seamlessly. The dataset pipeline handles collection, generation, augmentation, and cleaning of AI training data, supporting both synchronous and asynchronous workflows with clear separation of concerns.

```mermaid
classDiagram
    %% Core System Components
    class API {
        +initialize_apis()
        +setup_routes()
    }
    
    class BaseToolAPI {
        +setup_routes()
        +router: APIRouter
    }
    
    class Server {
        +initialize()
        +start()
        +stop()
        +status()
    }
    
    class PandasDB {
        +setup_query_engine()
        +query_data()
        +storeAgent()
        +setAgent()
        +coreAgent()
        +initializeConversation()
    }
    
    %% Agent Components
    class AgentStorage {
        +initialize_agent_storage()
        +setup_default_agents()
        +initializeAgentFlags()
        +load_agent()
        +list_available_agents()
    }
    
    class MultiModalPrompting {
        +llava_prompt()
        +embedding_ollama_prompt()
        +shot_prompt()
        +design_prompt()
        +chainOfThought()
        +deepResearch()
        +swarmPrompt()
    }
    
    %% Speech Components
    class TextToSpeech {
        +process_tts_responses()
        +generate_audio()
        +initialize_tts_model()
    }
    
    class SpeechToText {
        +listen()
        +recognizer()
        +whisperSTT()
        +googleSTT()
    }
    
    class SpeechManager {
        +initialize_tts_model()
        +generate_speech()
    }
    
    %% Vision Components
    class YoloProcessor {
        +process_frame()
        +load_model()
        +capture_screen()
    }
    
    %% Command & Control
    class FlagManager {
        +speech()
        +llava_flow()
        +yolo_state()
        +get_agent_state()
    }
    
    class commandLibrary {
        +updateCommandLibrary()
    }
    
    %% Search & Dataset Components
    class DuckDuckGoSearch {
        +text_search()
        +image_search()
        +news_search()
        +store_results()
    }
    
    class GitHubRepoCloner {
        +clone_and_store_repo()
    }
    
    class Crawl4AISearchAPI {
        +scrape_url()
        +format_result()
        +store_result()
    }
    
    %% Dataset Agents (Your Focus Area)
    class DatasetAgents {
        +collect_dataset()
        +generate_dataset()
        +augment_dataset()
        +clean_dataset()
    }
    
    class DatasetCollector {
        +search_sources()
        +process_results()
        +store_dataset()
    }
    
    class DatasetGenerator {
        +ollama_generate()
        +process_generation()
        +save_generated_data()
    }
    
    class DatasetAugmenter {
        +augment_with_feedback()
        +deep_search_augmentation()
        +multimodal_augmentation()
    }
    
    class DatasetCleaner {
        +detect_errors()
        +regenerate_garbage_data()
        +validate_dataset()
    }
    
    %% HuggingFace Integration
    class HuggingFaceHub {
        +download_model()
        +upload_model()
        +upload_dataset()
    }
    
    %% Utilities
    class Paths {
        +get_model_dir()
        +ensure_paths()
    }
    
    %% Relationships
    API --|> BaseToolAPI
    TextToSpeech o-- SpeechManager
    SpeechToText o-- SpeechManager
    AgentStorage o-- PandasDB
    AgentStorage -- MultiModalPrompting
    
    FlagManager -- commandLibrary
    FlagManager -- YoloProcessor
    FlagManager -- TextToSpeech
    FlagManager -- SpeechToText
    
    DatasetAgents *-- DatasetCollector
    DatasetAgents *-- DatasetGenerator
    DatasetAgents *-- DatasetAugmenter
    DatasetAgents *-- DatasetCleaner
    
    DatasetCollector -- DuckDuckGoSearch
    DatasetCollector -- GitHubRepoCloner
    DatasetCollector -- Crawl4AISearchAPI
    
    DatasetGenerator -- MultiModalPrompting
    DatasetAugmenter -- MultiModalPrompting
    
    DatasetAgents -- HuggingFaceHub
    
    MultiModalPrompting -- PandasDB
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE)

## Citations

Please use the following BibTeX entry to cite this project:

```bibtex
@software{oarc,
  author = {Leo Borcherding, Kara Rawson},
  title = {OARC: Ollama Agent Roll Cage is a powerful multimodal toolkit for AI interactions, automation, and workflows.},
  date = {4-10-2025},
  howpublished = {\url{https://github.com/Ollama-Agent-Roll-Cage/oarc}}
}
```

## Contact

For questions or support, please contact us at:

- **Email**: <NotSetup@gmail.com>
- **Issues**: [GitHub Issues](https://github.com/Leoleojames1/OARC/issues)
