"""
setup.py

Packaging metadata and CLI entry point for the content-pipeline.

Version: 0.8.5 — Technical debt fix release updating enrichment tests to provider terminology.
"""
from setuptools import setup, find_packages

setup(
    name="content-pipeline",
    version="0.8.5",
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
    ],
    entry_points={
        "console_scripts": [
            "content-pipeline=cli:cli",
        ],
    },
    python_requires=">=3.8",
)