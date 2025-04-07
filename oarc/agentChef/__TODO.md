# datasetAgents.py: Dataset collection, generation, augmentation, and cleaning agents

# ==========================
# Dataset Collection Agents
# ==========================
# TODO: Implement dataset collection agents with searchAPISetup
# - searchAPISetup should be able to search for data from various sources:
#   - crawl4ai
#   - GitHub
#   - arXiv
#   - DuckDuckGo
#   - Hugging Face datasets
#   - Kaggle

# ==========================
# Dataset Generation Agents
# ==========================
# TODO: Implement dataset generation agents with Ollama multimodal prompting
# - Raw Ollama model generation without any prompt boosting

# ==========================
# Dataset Augmentation Agents
# ==========================
# TODO: Implement dataset augmentation agents using dataset collection and generation agents
# - Generate new data from existing data
# - Operate in a conversational manner, with user feedback on generated data
# - Generate new data based on feedback
# - Perform deep searches for relevant data
# - Utilize multimodal prompting
# - Support code generation

# ==========================
# Dataset Cleaning Agents
# ==========================
# TODO: Implement dataset cleaning agents with augmentation
# - Clean data and search for errors, either automatically or with user feedback
# - Generate new data based on feedback
# - Perform deep searches for garbage data
# - For all garbage data points, regenerate with Ollama until they are no longer garbage

# ==========================
# High-Level AgentChef Provider
# ==========================
# TODO: Create a high-level AgentChef provider class that interlocks with the Python API

# ==========================
# DTO Classes
# ==========================
# TODO: Create basic DTO (Data Transfer Object) classes for the AgentChef to use

# ==========================
# Server Wrappers
# ==========================
# TODO: Provide server wrappers for the AgentChef to live in