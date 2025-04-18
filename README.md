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

## Setup

OARC requires Python 3.10 or 3.11 (Python 3.12+ is not yet supported due to TensorFlow compatibility). For GPU acceleration, ensure CUDA capatible device. AMD support coming soon.

```bash
# Install UV package manager
pip install uv

# Create & activate virtual environment with UV
uv venv --python 3.11

# Install package from pypi
uv pip install oarc

# Run setup to fetch all dependencies
oarc setup
```

### Example

```bash
python .\tests\speech\tts_fast_tests.py
```

## Development

For development purposes, oarc can be installed by cloning the repository and setting up the uv venv:

```bash
# Clone the repository
git clone https://github.com/Ollama-Agent-Roll-Cage/oarc.git
cd oarc

# Install UV package manager
pip install uv

# Create & activate virtual environment with UV
uv venv --python 3.11

# Install the package and dependencies in one step
uv run pip install -e .[dev]

# Run the setup command directly
uv run oarc setup

# After setup, you can activate the virtual environment and run OARC directly
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
# source .venv/bin/activate
```

### OARC Commands

```bash
# Run a specific command, see Commands.
oarc <command>
```

- `oarc` - Run the main command-line tool
- `oarc setup` - Install all required dependencies
- `oarc build` - Build from source code (development)
- `oarc publish` - Publish code to pypi with twine (development)

### Running a test

```bash
uv run .\tests\run_all_tests.py
```

### Building from source

```bash
uv run oarc build
# Creates wheel distribution in dist/ directory
```

### Publish

```bash
# Publish to default PyPI repository
uv run oarc publish

# Publish to alternative repository
uv run oarc publish --repository testpypi

# Skip build step and publish existing files
uv run oarc publish --skip-build
```

## Architecture

OARC's modular architecture is designed for extensibility and optimized performance. Core components, including APIs, agent management, speech processing, vision systems, and dataset pipelines, work together seamlessly. The dataset pipeline is a standout feature, offering robust capabilities for data collection, generation, augmentation, and cleaning. It supports both synchronous and asynchronous workflows, ensuring efficient processing while maintaining a clear separation of concerns to enhance scalability and maintainability.

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
- **Issues**: [GitHub Issues](https://github.com/Ollama-Agent-Roll-Cage/oarc/issues)
