from setuptools import setup, find_packages

setup(
    name='AgentManager',
    version='0.1.1',
    author='Nilavo Boral',
    author_email='nilavoboral@gmail.com',
    description='Managing a unified ecosystem of LLM providers, agents, and MCP servers.',
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    project_urls={
        "LinkedIn": "https://www.linkedin.com/in/nilavo-boral-123bb5228/"
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'langchain==1.0.5',
        'langchain-core==1.0.4',
        'langchain-mcp-adapters==0.1.12',
        'langchain-google-genai==3.0.1',
        'langchain-openai==1.0.2',
        'langchain-groq==1.0.0',
        'langchain-mistralai==1.0.1',
        'langchain-ollama==1.0.0',
        'nest_asyncio==1.6.0',
    ],
    extras_require={
        "ui": [
        'streamlit==1.49.1',
        ],
    },
    entry_points={
        'console_scripts': [
            'agentmanager-ui = agentmanager.ui_launcher:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.11',
)
