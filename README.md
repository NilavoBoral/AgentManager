# 🤖 AgentManager: Managing a unified ecosystem of LLM providers, agents, and MCP servers.

**AgentManager** is a Python package that provides a unified, high-level system for working with Large Language Models (LLMs). It simplifies managing provider credentials, selecting models, and constructing AI agents. It also supports integrating external tools via the Model Context Protocol (MCP) for more advanced workflows.

---

## ✨ Features

* **Universal Provider Support:** Seamlessly connect to various cloud LLM providers (e.g., **Google**, **OpenAI**, **Ollama**, **Mistral**, **Groq**) through a single line code.
* **Agent Construction:** Quickly set up basic agents or advanced tool-powered agents with minimal setup.
* **MCP Integration:** Easily incorporate tools from a MCP Server to enable complex, external actions.
* **Chat Support:** Supports both single-turn interactions and stateful, continuous conversations.
* **UI (Optional):** Interactive UI to explore the AgentManager.

---

### 📄 Requirements

* Python 3.11+

---

## 🛠️ Installation

- Install the core package without UI features:

  ```bash
  pip install agentmanager
  ```

- To use the interactive UI, install with the [ui] extras:

  ```bash
  pip install agentmanager[ui]
  ```

- If you already have an older version of AgentManager installed, upgrade to get the latest features:

  ```bash
  pip install --upgrade agentmanager[ui]
  ```

---

# Launching the UI
You can dive right in and explore `AgentManager’s` features through the interactive UI by running the following command in your terminal:

```bash
agentmanager-ui
```

This opens a intaractive interface in your browser, where you can:
- Manage LLM providers
- Configure and run agents
- Add and use multiple MCP tools

It’s a hands-on way to see everything in action.

[![AgentManager UI](https://github.com/NilavoBoral/AgentManager/blob/v0.1.1/Demo/AgentManager%20UI.jpeg)](https://github.com/NilavoBoral/AgentManager/blob/v0.1.1/Demo/AgentManager%20UI.mp4)

---

# ℹ️ Utility Methods

The `CloudAgentManager` provides helpful utility methods for discovering supported providers and models.

| Method                               | Description                                                          |
|--------------------------------------|----------------------------------------------------------------------|
| `cloud_agent_manager.get_providers()`            | Returns a list of all supported LLM provider names.                  |
| `cloud_agent_manager.get_models(provider_name)`  | Returns a list of all available model names for a given provider.   |
| `cloud_agent_manager.get_provider_key(provider_name)` | Returns the URL link where you can obtain your API key for the provider. |

## Example Utility Usage

```Python
from agentmanager import CloudAgentManager

cloud_agent_manager = CloudAgentManager()

# Get all supported providers
providers = cloud_agent_manager.get_providers()
print("Supported Providers:", providers)
# Output might be: ['OpenAI', 'Google', 'Ollama', ...]

# Get models for a specific provider
google_models = cloud_agent_manager.get_models("ollama")
print("Ollama Models:", ollama_models)
    
# Get API key page link
page_link_for_api_key = cloud_agent_manager.get_provider_key("mistral")
print("Mistral Key Link:", page_link_for_api_key)
```

---

# 🚀 Quick Start
The core functionality is encapsulated in the CloudAgentManager class. Here is a basic example for a single-turn chat.

## Single-Turn Chat Example

This script demonstrates initializing an LLM, creating a basic agent, and getting a single response.

```python
import asyncio
from agentmanager import CloudAgentManager

# The following constants would typically be read from environment variables or a secure vault
# Placeholder values are used for demonstration.
PROVIDER = "google"
API_KEY = "YOUR_API_KEY_HERE" # !!! REPLACE WITH YOUR ACTUAL API KEY !!!
MODEL_NAME = "gemini-2.5-flash"

async def single_chat():
    # 1️⃣ Initialize the CloudAgentManager
    cloud_agent_manager = CloudAgentManager()

    # 2️⃣ Prepare the LLM (handles API key and model validation)
    llm = cloud_agent_manager.prepare_llm(PROVIDER, API_KEY, MODEL_NAME)
        
    # 3️⃣ Prepare the agent (no MCP tools in this example)
    agent, tools = await cloud_agent_manager.prepare_agent(llm)
    
    # 4️⃣ Send message
    user_message = "what is the capital of India?"    
    response_messages = await cloud_agent_manager.get_agent_response(agent, user_message)
    
    for m in response_messages:
        print(m.content)


if __name__ == "__main__":
    asyncio.run(single_chat())
```

## 💬 Continuous Chat Loop

For an interactive, multi-turn conversation that maintains context using chat_history.

```python
import asyncio
from agentmanager import CloudAgentManager
from typing import List

# The following constants would typically be read from environment variables or a secure vault
# Placeholder values are used for demonstration.
PROVIDER = "google"
API_KEY = "YOUR_API_KEY_HERE" # !!! REPLACE WITH YOUR ACTUAL API KEY !!!
MODEL_NAME = "gemini-2.5-flash"

async def chat_loop():
    # 1️⃣ Initialize the CloudAgentManager
    cloud_agent_manager = CloudAgentManager()

    # 2️⃣ Prepare LLM
    llm = cloud_agent_manager.prepare_llm(PROVIDER, API_KEY, MODEL_NAME)
        
    # 3️⃣ Prepare agent (e.g., without MCP tools)
    agent, tools = await cloud_agent_manager.prepare_agent(llm)

    # 4️⃣ Initialize chat history
    # The agent will use this list to maintain context across turns.
    chat_history: List[Any] = []

    # 5️⃣ Terminal loop
    print("\n--- Start Chat ---")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        try:
            # The cloud_agent_manager updates chat_history in place
            new_messages = await cloud_agent_manager.get_agent_response(agent, user_input, chat_history)
            
            for m in new_messages:
                print(f"Agent: {m.content}")
                
        except Exception as e:
            print(f"❌ Agent failed to respond: {e}")

if __name__ == "__main__":
    asyncio.run(chat_loop())
```

## ⚙️ Advanced: Agent with MCP Tools

If your agent needs to interact with external tools via a Model Context Protocol (MCP), you can provide a list of MCP server configurations during agent preparation.

Each MCP entry can include:
- `name` (str): A unique name for the MCP server (required)
- `url` (str): Your MCP server's URL (required)
- `headers` (Optional[Dict[str, str]]): Dictionary of headers, e.g. 
  ```python
  {"Authorization": "Bearer XYZ", "X-Custom": "ABC123", ...}
  ```

```python
import asyncio
from agentmanager import CloudAgentManager

# The following constants would typically be read from environment variables or a secure vault
# Placeholder values are used for demonstration.
PROVIDER = "google" # !!! REPLACE WITH PROVIDER YOU WANT TO USE !!!
API_KEY = "YOUR_API_KEY_HERE" # !!! REPLACE WITH PROVIDER'S API KEY !!!
MODEL_NAME = "gemini-2.5-flash" # !!! REPLACE WITH MODEL YOU WANT TO USE !!!

# Define multiple MCP configurations (name, url, optional multiple headers)
mcps=[
    {
        "name": "MyFirstServer",
        "url": "MCP_URL_1" # !!! REPLACE WITH YOUR MCP URL !!!
    },
    {
        "name": "MySecondServer",
        "url": "MCP_URL_2", # !!! REPLACE WITH YOUR MCP URL !!!
        # Optional: Custom headers for authentication/routing
        "header": {
            # !!! REPLACE WITH YOUR ACTUAL HEADER NAMES & VALUES !!!
            "FIRST_HEADER_NAME": "FIRST_HEADER_VALUE",
            "SECOND_HEADER_NAME": "SECOND_HEADER_VALUE",
            # Add more headers as needed ...
        }
    },
    # Add more MCP servers as needed ...
]

async def mcp_agent_example():

    # 1️⃣ Initialize the CloudAgentManager
    cloud_agent_manager = CloudAgentManager()

    # 2️⃣ Prepare LLM
    llm = cloud_agent_manager.prepare_llm(PROVIDER, API_KEY, MODEL_NAME)
    
    # 3️⃣ Prepare agent with MCP configuration
    agent, tools = await cloud_agent_manager.prepare_agent(llm, mcps)

    # 4️⃣ Initialize chat history
    # The agent will use this list to maintain context across turns.
    chat_history: List[Any] = []

    # 5️⃣ Terminal loop
    print("\n--- Start Chat ---")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        try:
            # The cloud_agent_manager updates chat_history in place
            new_messages = await cloud_agent_manager.get_agent_response(agent, user_input, chat_history)
            
            for m in new_messages:
                print(f"Agent: {m.content}")
                
        except Exception as e:
            print(f"❌ Agent failed to respond: {e}")

if __name__ == "__main__":
    asyncio.run(mcp_agent_example())

```

---
