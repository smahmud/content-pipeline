"""
Transcription Providers

This module contains all transcription provider implementations following consistent
naming conventions: {deployment}_{service}.py pattern.

Available Providers:
    - LocalWhisperProvider: Local Whisper model (local_whisper.py)
    - CloudOpenAIWhisperProvider: OpenAI Whisper API (cloud_openai_whisper.py)
    - CloudAWSTranscribeProvider: AWS Transcribe (cloud_aws_transcribe.py)

All providers implement the TranscriberProvider protocol defined in base.py.
"""

from pipeline.transcription.providers.base import TranscriberProvider

__all__ = [
    "TranscriberProvider",
]
