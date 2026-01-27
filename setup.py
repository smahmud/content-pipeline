"""
setup.py

Packaging metadata and CLI entry point for the content-pipeline.

Version: 0.6.0 — Refactors CLI into modular subcommands with click groups,
improving maintainability and contributor onboarding. Establishes foundation
for future commands (enrich, format, etc.).
"""
from setuptools import setup, find_packages

setup(
    name="content-pipeline",
    version="0.6.0",
    packages=find_packages(),
    install_requires=[
        "yt_dlp",
        "click",
        "moviepy",
        "pydantic>=2.0",
        "ffmpeg-python",
        "openai-whisper",
    ],
    entry_points={
        "console_scripts": [
            "content-pipeline=cli:cli",
        ],
    },
    python_requires=">=3.8",
)