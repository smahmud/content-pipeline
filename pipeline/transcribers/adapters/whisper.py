"""
File: whisper.py

DEPRECATED: This module is deprecated in v0.6.5. Use whisper_local.py instead.

Implements the WhisperAdapter using OpenAI's Whisper model.
Conforms to the enhanced TranscriberAdapter protocol.
"""
import warnings
from pipeline.transcribers.adapters.whisper_local import WhisperLocalAdapter

# Deprecation warning
warnings.warn(
    "WhisperAdapter is deprecated in v0.6.5. Use WhisperLocalAdapter from whisper_local module instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility alias
WhisperAdapter = WhisperLocalAdapter
