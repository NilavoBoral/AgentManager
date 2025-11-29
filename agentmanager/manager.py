import os
from typing import Any, Dict, Optional, Tuple, List
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from agentmanager.config import model_registry

class CloudAgentManager:
    """
    Manages Cloud LLM providers, API keys, available models, 
    and handles creation and invocation of agents, including optional MCP tools.

    Responsibilities:
    1. Normalize provider names and validate supported providers.
    2. Retrieve provider information, models, and API key links.
    3. Set up environment variables and provider-specific configurations.
    4. Initialize LLM instances and validate selected models.
    5. Create agents with or without MCP tools.
    6. Invoke agents asynchronously and manage chat history.

    Usage example:
        CloudAgentManager = CloudAgentManager()
        llm = CloudAgentManager.prepare_llm("OpenAI", api_key, "gpt-4o-mini")
        agent, tools = await CloudAgentManager.prepare_agent(llm, mcps=[{"url":..., "header":{...}}, ...])
        response = await CloudAgentManager.get_agent_response(agent, "Hello")
    """

    def __init__(self):
        self._config = model_registry

    # -------------------------------
    # Utils
    # -------------------------------

    def _normalize_provider(self, name: str) -> str:
        """
        Return the canonical provider name (case-insensitive).

        Args:
            name (str): Name of the provider (case-insensitive).

        Returns:
            str: Canonical provider name.

        Raises:
            ValueError: If provider is unsupported.
        """
        name = name.lower()
        for key in self._config.keys():
            if key.lower() == name:
                return key
        raise ValueError(
            f"Unsupported provider '{name}'. "
            f"Available providers: {self.get_providers()}"
        )


    def get_providers(self) -> List[str]:
        """
        Public: List all available provider names.

        Returns:
            List[str]: List of provider names.
        """
        return list(self._config.keys())
    

    def get_provider_key(self, provider: str) -> str:
        """
        Public: Get the API key page link for a given provider.

        Args:
            provider (str): Name of the provider.

        Returns:
            str: URL where users can get their API key.

        Raises:
            ValueError: If provider is unsupported or key link is not defined.
        """
        provider_handle = self._normalize_provider(provider)
        provider_info = self._config[provider_handle]

        api_page = provider_info.get("get_key_link")
        if not api_page:
            raise ValueError(f"No api key page link defined for provider: {provider}.")
        return api_page


    def get_models(self, provider: str) -> List[str]:
        """
        Public: List all available models for a given provider.

        Args:
            provider (str): Name of the provider (case-insensitive).

        Returns:
            List[str]: List of model names supported by the provider.

        Raises:
            ValueError: If the provider is unsupported.
        """
        provider = self._normalize_provider(provider)
        return self._config[provider]["models"]
    

    def _set_env(self, provider_handle: str, provider_info: dict, api_key: str):
        """
        Set environment variable for a provider and handle provider-specific special cases (like Ollama).

        Args:
            provider_handle (str): Canonical provider name.
            provider_info (dict): Configuration dictionary for the provider.
            api_key (str): User-provided API key.

        Raises:
            RuntimeError: If the provider does not have an environment key defined.
        """
        env_key = provider_info.get("env_key")
        if not env_key:
            raise RuntimeError(f"Provider {provider_handle} does not have an proper env setup defined.")
        os.environ[env_key] = api_key.strip()

        # Special case: Ollama headers
        if provider_handle.lower() == "ollama":
            init_args = provider_info.get("init_args", {})
            init_args["client_kwargs"] = {"headers": {"Authorization": f"Bearer {os.environ[env_key]}" }}
            provider_info["init_args"] = init_args

    # -------------------------------
    # LLM construction
    # -------------------------------

    def prepare_llm(
        self, provider: str, api_key: str, model_name: str
    ):
        """
        Fully prepare and initialize an LLM instance for a provider.

        Steps:
            1. Normalize provider name.
            2. Set API key (and handle provider-specific tweaks).
            3. Validate that the model exists.
            4. Initialize and return the LLM instance.

        Args:
            provider (str): Provider name (case-insensitive).
            api_key (str): API key for the provider.
            model_name (str): Name of the model to initialize.

        Returns:
            Any: Initialized LLM instance.

        Raises:
            ValueError: If the model is not available for the provider.
            RuntimeError: If LLM initialization fails.
        """
        # 1️⃣ Normalize provider
        provider_handle = self._normalize_provider(provider)
        provider_info = self._config[provider_handle]

        # 2️⃣ Set API key
        self._set_env(provider_handle, provider_info, api_key)

        # 3️⃣ Validate model
        if model_name not in provider_info["models"]:
            raise ValueError(
                f"Model '{model_name}' not available for provider '{provider_handle}'. "
                f"Available models: {provider_info['models']}"
            )

        # 4️⃣ Initialize LLM
        llm_class = provider_info["class"]
        init_args = provider_info.get("init_args", {})
        try:
            return llm_class(model=model_name, **init_args)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {e}") from e
        
    # -------------------------------
    # Agent creation
    # -------------------------------

    async def prepare_agent(
        self,
        llm,
        mcps: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Any, Optional[List[Any]]]:
        """
        Creates an agent with optional multiple MCP tools.

        Args:
            llm: Initialized LLM instance.
            mcps (Optional[List[Dict[str, Any]]]): List of MCP configurations. 
                Each dict can have:
                    - "url" (str): MCP service URL.
                    - "header" (Optional[Dict[str, str]]): Single header as {name: value}.

        Returns:
            Tuple[Any, Optional[List[Any]]]: Agent instance, and a combined list of tools if MCPs are used, else None.

        Raises:
            RuntimeError: If agent or MCP initialization fails.
        """

        # If MCP is not required → simple agent
        if not mcps:
            try:
                agent = create_agent(llm)
                return agent, None
            except Exception as e:
                raise RuntimeError(f"Failed to create agent: {e}") from e

        # MCP config
        mcp_config: Dict[str, Any] = {}

        for mcp in mcps:
            url = mcp.get("url", "").strip()
            if not url:
                continue

            mcp_config[f"name_{url}"] = {"transport": "streamable_http", "url": url}

            header = mcp.get("header")
            if header and isinstance(header, dict):
                mcp_config[f"name_{url}"]["headers"] = header

        # Initialize MCP
        try:
            client = MultiServerMCPClient(mcp_config)
            tools = await client.get_tools()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MCP: {e}") from e

        # Agent with tools
        try:
            agent = create_agent(llm, tools)
            return agent, tools
        except Exception as e:
            raise RuntimeError(f"Failed to create agent with MCP tools: {e}") from e

    # -------------------------------
    # Agent invocation
    # -------------------------------

    async def get_agent_response(self, agent, user_input: str, chat_history: Optional[list] = None):
        """
        Asynchronously invokes the agent with user input.

        Args:
            agent: Initialized agent instance.
            user_input (str): User message to send to the agent.
            chat_history (Optional[list]): List of previous messages (session state).

        Returns:
            list: List of new messages returned by the agent.

        Raises:
            RuntimeError: If agent execution fails or returns empty response.
        """
        if chat_history is None:
            chat_history = []

        chat_history.append(HumanMessage(content=user_input))

        try:
            response = await agent.ainvoke({"messages": chat_history})
            if not response:
                raise RuntimeError("Agent returned an empty response.")

            new_messages = response["messages"][len(chat_history):]
            chat_history.extend(new_messages)
            return new_messages

        except Exception as e:
            # rollback
            if chat_history and chat_history[-1].content == user_input:
                chat_history.pop()
            raise RuntimeError(f"Agent execution failed:\n{e}") from e
