"""
setup.py

Packaging metadata and CLI entry point for the content-pipeline.

Version: 1.0.0 — First public release with complete pipeline, MCP server, REST API, and frozen schemas.
"""
from setuptools import setup, find_packages

setup(
    name="content-pipeline",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "yt_dlp",
        "click",
        "moviepy",
        "pydantic>=2.0",
        "ffmpeg-python",
        "openai-whisper",
        "boto3>=1.34.0",
        "openai>=1.0",
        "anthropic>=0.77.0",
        "tiktoken>=0.5.0",
        "pyyaml>=6.0",
        "jinja2>=3.0",
        "requests>=2.31.0",
        "mcp>=1.0",
        "fastapi>=0.100",
    ],
    entry_points={
        "console_scripts": [
            "content-pipeline=cli:cli",
        ],
    },
    python_requires=">=3.8",
)