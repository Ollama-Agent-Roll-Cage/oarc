#datasetAgents.py: dataset collection, generation, augmentation, cleaning agents

#TODO implement dataset collection agents with searchAPISetup
# - searchAPISetup should be able to search for data from various sources
# - crawl4ai, github, arxiv, duckduckgo, huggingface datasets, kaggle.

#TODO implement dataset generation agents with ollama multimodal prompting
# - raw ollama model generation without any prompt boosting

#TODO implement dataset augmentation agents with dataset collection agents and dataset generation agents
# - dataset augmentation agents should be able to generate new data from existing data
# - conversational manner, with the user providing feedback on the generated data
# - the agent should be able to generate new data based on the feedback
# - deep search for relevant data
# - multimodal prompting
# - code generation

#TODO implement dataset cleaning agents with augmentation
# - dataset cleaning agents should be able to clean the data, search for errors, either automatically or with user feedback
# - the agent should be able to generate new data based on the feedback
# - deep search for garbage data, for all garbage datapoints, regenerate with ollama until the are no longer garbage

# TODO we need a high level AgentChef provider class which interlocks with the python API
# TODO we need to create some basic DTO classes for the agent chef to use
# TODO we should provide the server wrappers for the agent chef to live in