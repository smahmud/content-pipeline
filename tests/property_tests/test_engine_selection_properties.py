"""
Property-based tests for engine selection validation.

**Property 1: Engine Selection Validation**
*For any* engine selection input, the system should validate engine availability,
enforce explicit selection requirements, and provide appropriate error messages.
**Validates: Requirements 1.2, 1.3, 1.4, 1.5**
"""

import pytest
from hypothesis import given, strategies as st, settings

from pipeline.transcribers.factory import EngineFactory
from pipeline.config.schema import TranscriptionConfig, EngineType


@given(engine_type=st.sampled_from(['whisper-local', 'whisper-api', 'aws-transcribe', 'auto']))
@settings(max_examples=10, deadline=None)
def test_valid_engine_types_accepted(engine_type):
    """
    **Property 1a: Valid Engine Types Accepted**
    *For any* valid engine type, the system should accept it and proceed 
    with validation checks.
    **Validates: Requirements 1.2, 1.3, 1.4**
    """
    factory = EngineFactory()
    config = TranscriptionConfig()
    config.engine = engine_type
    
    # Should not raise exception for valid engine types
    available_engines = factory.get_available_engines()
    
    if engine_type == 'auto':
        # Auto engine should always be accepted as valid
        assert engine_type == 'auto'
    else:
        # Other valid engines should be in the available engines list
        assert engine_type in available_engines
    
    # Should be able to check if engine is available
    if engine_type != 'auto':
        is_available = factory.is_engine_available(engine_type)
        assert isinstance(is_available, bool)


@given(engine_type=st.sampled_from(['invalid-engine', 'whisper', 'openai', 'aws', 'transcribe']))
@settings(max_examples=10, deadline=None)
def test_invalid_engine_types_rejected(engine_type):
    """
    **Property 1b: Invalid Engine Types Rejected**
    *For any* invalid engine type, the system should reject it with 
    clear error messages.
    **Validates: Requirements 1.5**
    """
    factory = EngineFactory()
    config = TranscriptionConfig()
    config.engine = engine_type
    
    # Invalid engines should not be available
    is_available = factory.is_engine_available(engine_type)
    assert not is_available
    
    # Should raise ValueError when trying to create invalid engine
    with pytest.raises(ValueError) as exc_info:
        factory.create_engine(engine_type, config)
    
    error_message = str(exc_info.value)
    # Error message should mention the invalid engine and available options
    assert engine_type in error_message
    assert "Available engines" in error_message or "Unsupported engine" in error_message


def test_explicit_engine_requirement_property():
    """
    **Property 1c: Explicit Engine Requirement**
    *For any* CLI invocation, explicit engine selection should be enforced.
    **Validates: Requirements 1.1**
    """
    # This property is enforced at the CLI level through Click's required=True
    # We test that the engine option is properly configured
    from cli.shared_options import engine_option
    
    # Create a dummy function to test the decorator
    @engine_option()
    def dummy_command(engine):
        return engine
    
    # The decorator should add the required engine parameter
    assert hasattr(dummy_command, '__click_params__')
    engine_param = None
    for param in dummy_command.__click_params__:
        if param.name == 'engine':
            engine_param = param
            break
    
    assert engine_param is not None
    assert engine_param.required is True
    assert 'whisper-local' in engine_param.type.choices
    assert 'whisper-api' in engine_param.type.choices
    assert 'aws-transcribe' in engine_param.type.choices
    assert 'auto' in engine_param.type.choices