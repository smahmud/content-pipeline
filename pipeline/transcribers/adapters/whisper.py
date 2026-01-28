"""
File: whisper.py

DEPRECATED: This module is deprecated in v0.6.5. Use local_whisper.py instead.

Implements the WhisperAdapter using OpenAI's Whisper model.
Conforms to the enhanced TranscriberAdapter protocol.
"""
import warnings
from pipeline.transcribers.adapters.local_whisper import LocalWhisperAdapter

# Deprecation warning
warnings.warn(
    "WhisperAdapter is deprecated in v0.6.5. Use LocalWhisperAdapter from local_whisper module instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility alias
WhisperAdapter = LocalWhisperAdapter
