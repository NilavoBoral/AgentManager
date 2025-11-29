# ============================================================
# Streamlit App (using your AgentManager package)
# ============================================================

import os
import json
import re
import asyncio
import streamlit as st

from agentmanager import CloudAgentManager
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import nest_asyncio
nest_asyncio.apply()


# ============================================================
# UI Helpers
# ============================================================

def parse_msg(message):

    # If already a dict → display as JSON
    if isinstance(message, dict):
        st.json(message)
        return

    # Try to parse as JSON string
    if isinstance(message, str):
        try:
            parsed = json.loads(message)
            if not message.isdigit():
                st.json(parsed)
            else:
                st.markdown(parsed)
            return
        except:
            pass

    # If it's a list → try extract text field
    if isinstance(message, list):
        try:
            st.markdown(message[0]["text"])
        except:
            st.markdown(message)
    else:
        st.markdown(message)


def render_msg(msg):
    """Display a message block elegantly"""
    content = msg.content

    if isinstance(msg, ToolMessage):
        tool_name = msg.name or "Tool"
        with st.expander(f"🛠️ {tool_name} Response", expanded=False):
            parse_msg(content)
    else:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                with st.expander(f"🔧 {call.get('name')} Call", expanded=False):
                    st.json(call.get("args"))
        else:
            parse_msg(content)


def show_chat(msg):
    """Render a chat bubble depending on role."""
    if isinstance(msg, HumanMessage):
        role = "user"
        avatar = None
    elif isinstance(msg, ToolMessage):
        role = "tool"
        avatar = "🛠️"
    else:
        role = "assistant"
        avatar = None

    with st.chat_message(role, avatar=avatar):
        render_msg(msg)


def remove_markdown_format(text):
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"#+\s*(.*)", r"\1", text)
    return text


def truncate(text, n=200):
    return text if len(text) <= n else text[:n] + "..."


@st.dialog("Tools", width="medium")
def show_tools(tools):
    with st.spinner("Loading tools..."):
        for tool in tools:
            st.markdown("---")
            st.write(f"### **Name:** {tool.name}\n**Description:** {remove_markdown_format(truncate(tool.description))}\n\n")


# ============================================================
# Streamlit Page Setup
# ============================================================

st.markdown("""
    <style>
    /* Active button style */
    div.stButton > button:first-child {
        background-color: #007bff;  /* Primary blue */
        color: white;
        border: none;
        transition: 0.3s ease;
        border-radius: 6px;
    }
    /* Hover effect for active button */
    div.stButton > button:first-child:hover {
        background-color: #0056b3;  /* Darker blue on hover */
        color: white;
    }
    /* Disabled (faded) button style */
    div.stButton > button:disabled {
        background-color: #aac4f6 !important;  /* Softer, faded blue */
        color: #333 !important;  /* Dark gray for better contrast */
        opacity: 0.9 !important;
        cursor: not-allowed !important;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AgentManager UI", page_icon="🤖", layout="wide")
st.title("🤖 AgentManager UI")
st.caption("A multi-provider chat agent, with optional MCP tools to explore the [AgentManager](https://pypi.org/project/AgentManager/).")

# ============================================================
# Session State Initialization
# ============================================================

if "cloud_agent_manager" not in st.session_state:
    st.session_state.cloud_agent_manager = CloudAgentManager()

if "agent" not in st.session_state:
    st.session_state.agent = None

if "tools" not in st.session_state:
    st.session_state.tools = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# Sidebar — Provider Selection
# ============================================================

cloud_agent_manager = st.session_state.cloud_agent_manager
providers = cloud_agent_manager.get_providers()

st.sidebar.header("⚙️ Agent Configuration")

col1, col2 = st.sidebar.columns(2)
with col1:
    provider = st.selectbox("Provider", providers, help="Select which model provider you want to use.")
with col2:
    # API key
    key_link = cloud_agent_manager.get_provider_key(provider)
    api_key = st.text_input(
        f"{provider} [API Key]({key_link})",
        type="password",
        help=f"Enter your {provider} API key."
    )

models = cloud_agent_manager.get_models(provider)
model_name = st.sidebar.selectbox("Model", models, help="Select a model you want to use.")

# Optional MCP Config
with st.sidebar.expander("🔌 MCP Config (optional)", expanded=True):

    # Initialize MCP list only ONCE
    if "mcps" not in st.session_state:
        st.session_state.mcps = []

    col1, col2 = st.columns(2)
    with col1:
        number_of_mcp_servers = st.number_input("Number of MCPs", min_value=0, value="min", step=1)
        
        # Resize list to match selected number
        # (Preserves existing URLs/headers instead of resetting everything)
        if len(st.session_state.mcps) < number_of_mcp_servers:
            # Add missing entries
            st.session_state.mcps.extend(
                [{"url": None}] * (number_of_mcp_servers - len(st.session_state.mcps))
            )
        elif len(st.session_state.mcps) > number_of_mcp_servers:
            # Remove extra entries
            st.session_state.mcps = st.session_state.mcps[:number_of_mcp_servers]

    with col2:
        temp_mcp_names = [f"MCP {i}" for i in range(1, number_of_mcp_servers + 1)]
        mcp_name = st.selectbox("MCP", options=temp_mcp_names)

    current_mcp_index = int(mcp_name[-1])-1 if mcp_name else -1  # -1 indicates no MCP server added
    if current_mcp_index >= 0:

        default_mcp_url = st.session_state.mcps[current_mcp_index]["url"]
        st.session_state.mcps[current_mcp_index]["url"] = st.text_input(f"{mcp_name} URL", value=default_mcp_url or None)

        col1, col2 = st.columns(2)
        header = st.session_state.mcps[current_mcp_index].get("header")
        # extract first key/value if header exists
        if header:
            first_key = list(header.keys())[0]
            first_value = header[first_key]
        else:
            first_key = "Authorization"
            first_value = None

        with col1:
            mcp_header_name = st.text_input(
                "Header Name (optional)",
                value=first_key,
                help="You can specify a custom request header name if your MCP server requires one."
            )
        with col2:
            mcp_header_value = st.text_input(
                "Header Value (optional)",
                value=first_value,
                type="password",
                help="You can specify a custom request header value if your MCP server requires one."
            )
        # --- Save back to session state ---
        # Always store {} -> {name: value}
        if mcp_header_name and mcp_header_value:
            st.session_state.mcps[current_mcp_index]["header"] = {
                mcp_header_name: mcp_header_value
            }


# ============================================================
# Prepare Agent Button
# ============================================================

disabled = not (provider and model_name and api_key)

if st.sidebar.button("Prepare Agent", disabled=disabled, use_container_width=True):

    has_none_url = any(item.get("url") is None for item in st.session_state.mcps)
    if has_none_url:
        st.sidebar.error("One or more MCP URLs are empty.")
    else:
        try:
            # 1️⃣ Prepare LLM
            st.session_state.llm = cloud_agent_manager.prepare_llm(provider, api_key, model_name)

            # 2️⃣ Prepare Agent (+ MCP)
            st.session_state.agent, st.session_state.tools = asyncio.run(
                cloud_agent_manager.prepare_agent(
                    llm=st.session_state.llm,
                    mcps=st.session_state.mcps
                )
            )
            st.success("Agent Ready!")
            st.rerun()

        except Exception as e:
            st.sidebar.error(str(e))


# ============================================================
# Chat Configuration (System Message)
# ============================================================

st.sidebar.markdown("---")

update_disabled = False

with st.sidebar.expander("💬 Chat Configuration (optional)", expanded=False):
    system_msg = st.text_area("System Message", placeholder="Enter system message...", height=40)

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        submitted = st.button("Update", use_container_width=True, disabled=update_disabled)
        if submitted and system_msg.strip():
            
            st.session_state.chat_history.append(SystemMessage(content=system_msg.strip()))
            st.success("System message added.")


# ============================================================
# Sidebar Chat Tools
# ============================================================

st.sidebar.markdown("---")
# Always show View MCP Tools button
tools_available = (
    "tools" in st.session_state
    and st.session_state.tools is not None
    and len(st.session_state.tools) > 0
)

col1, col2 = st.sidebar.columns(2)
with col1:
    view_tools_btn = st.button(
        "View MCP Tools",
        use_container_width=True,
        disabled=not tools_available
    )
    # If button clicked and tools exist → open the dialog
    if view_tools_btn and tools_available:
        show_tools(st.session_state.tools)

with col2:
    if st.button("New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.sidebar.markdown("""
<hr style="margin-top: 50px;"/>
<div style='text-align: center; color: gray; font-size: 0.8em'>
    👨‍💻 Made with ❤️ by <a href='https://www.linkedin.com/in/nilavo-boral-123bb5228/' target='_blank'>Nilavo Boral</a>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Main Chat UI
# ============================================================

# Show earlier messages
for msg in st.session_state.chat_history:
    show_chat(msg)

prompt = st.chat_input("Type your message...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.agent:
        st.warning("Prepare the agent first in the sidebar.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Prepare LLM
                st.session_state.llm = cloud_agent_manager.prepare_llm(provider, api_key, model_name)

                # Prepare Agent (+ MCP)
                st.session_state.agent, st.session_state.tools = asyncio.run(
                    cloud_agent_manager.prepare_agent(
                        llm=st.session_state.llm,
                        mcps=st.session_state.mcps
                    )
                )
                # Get model response
                new_msgs = asyncio.run(
                    cloud_agent_manager.get_agent_response(
                        st.session_state.agent,
                        prompt,
                        st.session_state.chat_history
                    )
                )
                for m in new_msgs:
                    show_chat(m)

            except Exception as e:
                st.error(str(e))
