from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

model_registry = {
    "Google": {
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "env_key": "GOOGLE_API_KEY",
        "get_key_link": "https://aistudio.google.com/app/api-keys",
        "class": ChatGoogleGenerativeAI,
        "init_args": {"max_retries": 2}
    },
    "OpenAI": {
        "models": ["gpt-5-nano", "gpt-5-mini", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4.1-mini", "gpt-4.1", "gpt-5"],
        "env_key": "OPENAI_API_KEY",
        "get_key_link": "https://platform.openai.com/api-keys",
        "class": ChatOpenAI,
        "init_args": {"max_retries": 2}
    },
    "Ollama": {
        "models": [
            "gpt-oss:20b-cloud", "gpt-oss:120b-cloud",
            "deepseek-v3.1:671b-cloud",
            "kimi-k2:1t-cloud",
            "qwen3-coder:480b-cloud",
            "glm-4.6:cloud",
            "minimax-m2:cloud" # 230b
            ],
        "env_key": "OLLAMA_API_KEY",
        "get_key_link": "https://ollama.com/settings/keys",
        "class": ChatOllama,
        "init_args": {
            "max_retries": 2,
            "base_url": "https://ollama.com"
        }
    },
    "Groq": {
        "models": [
            "openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-safeguard-20b",
            "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct",
            # "groq/compound", "groq/compound-mini", # tool calling is not supported
            "moonshotai/kimi-k2-instruct", "moonshotai/kimi-k2-instruct-0905"
            ],
        "env_key": "GROQ_API_KEY",
        "get_key_link": "https://console.groq.com/keys",
        "class": ChatGroq,
        "init_args": {"max_retries": 2}
    },
    "Mistral": {
        "models": [
            # "ministral-3b-2410", "ministral-8b-2410", # small models
            "mistral-tiny-2407", "voxtral-mini-2507",
            "pixtral-12b",
            "mistral-small-2506", "voxtral-small-2507", "devstral-small-2507", "magistral-small-2509", 
            "codestral-2508",
            "mistral-medium", "devstral-medium-2507", "magistral-medium-2509",
            "pixtral-large-2411", "mistral-large-latest"
            ],
        "env_key": "MISTRAL_API_KEY",
        "get_key_link": "https://console.mistral.ai/build/playground?workspace_dialog=apiKeys&workspace_dialog_expanded=true",
        "class": ChatMistralAI,
        "init_args": {"max_retries": 2}
    },
}
