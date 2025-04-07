# TODO: Upgrade Discord Bot  

## Features to Add  
- Integrate **Text-to-Speech (TTS)** using Coqui models from `#codebase`.  
- Add **Speech-to-Text (STT)** functionality using Whisper models.  
- Implement **multi-modal prompts** for enhanced user interaction.  
- Enable **agent management** for dynamic bot behavior using `commandLibrary`.  

## Enhancements  
- Improve **error handling** in `initializeBasePaths` from `ollamaChatbotWizard_OLD.py`.  
- Add support for **async API calls** for better performance, inspired by `groq-magic.py`.  
- Extend **dataset collection** capabilities using `GitHubRepoCloner` and `DuckDuckGoSearch`.  

## Integration  
- Use `HuggingFaceHub` for model downloads and uploads.  
- Leverage `FlagManager` for managing bot states and configurations.  

## Testing  
- Write unit tests for new features in `/tests/discordBotTests`.  
- Validate dataset generation and augmentation pipelines.  

## Documentation  
- Update the bot's README to include new features and usage examples.  
- Add a section on **API compatibility** similar to `image-generator-readme.md`.  

## License  
- Ensure compliance with the **Apache 2.0 License** as outlined in `LICENSE`.  
