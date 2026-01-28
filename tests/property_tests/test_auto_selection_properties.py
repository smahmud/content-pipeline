"""
Property-based tests for auto-selection priority logic.

**Property 5: Auto-Selection Priority Logic**
*For any* configuration of available engines and priority preferences, the auto-selector 
should consistently select engines according to the defined priority order and availability.
**Validates: Requirements 4.1, 4.2, 4.3, 4.5**
"""

import pytest
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, assume, settings

from pipeline.transcribers.auto_selector import AutoSelector
from pipeline.transcribers.factory import EngineFactory
from pipeline.config.schema import (
    TranscriptionConfig, AutoSelectionConfig, EngineType,
    WhisperLocalConfig, WhisperAPIConfig, AWSTranscribeConfig
)


# Strategy for generating valid engine types
engine_types = st.sampled_from([
    EngineType.WHISPER_LOCAL.value,
    EngineType.WHISPER_API.value,
    EngineType.AWS_TRANSCRIBE.value
])

# Strategy for generating priority orders (permutations of available engines)
def priority_order_strategy():
    """Generate valid priority orders."""
    all_engines = [
        EngineType.WHISPER_LOCAL.value,
        EngineType.WHISPER_API.value,
        EngineType.AWS_TRANSCRIBE.value
    ]
    return st.lists(
        st.sampled_from(all_engines),
        min_size=1,
        max_size=3,
        unique=True
    )

# Strategy for engine availability scenarios
def availability_scenario_strategy():
    """Generate engine availability scenarios."""
    return st.dictionaries(
        keys=engine_types,
        values=st.booleans(),
        min_size=1,
        max_size=3
    )


class TestAutoSelectionPriorityLogic:
    """Test auto-selection priority logic properties."""

    @given(
        priority_order=priority_order_strategy(),
        availability=availability_scenario_strategy()
    )
    @settings(max_examples=50)
    def test_priority_order_consistency(self, priority_order, availability):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* priority order and availability scenario, the auto-selector should 
        consistently select the highest priority available engine.
        """
        # Skip if no engines are available
        assume(any(availability.values()))
        
        # Create mock factory
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = list(availability.keys())
        
        # Mock engine availability based on our scenario
        def mock_is_available(engine_type):
            return availability.get(engine_type, False)
        
        def mock_validate_requirements(engine_type, config):
            # Return empty list (no errors) if available, errors if not
            return [] if availability.get(engine_type, False) else ['Not available']
        
        factory.is_engine_available.side_effect = mock_is_available
        factory.validate_engine_requirements.side_effect = mock_validate_requirements
        
        # Create configuration with custom priority order
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(priority_order=priority_order)
        
        selector = AutoSelector(factory, config)
        
        # Get the actual priority order (which may include engines not in our priority_order)
        actual_priority = selector.get_selection_priority()
        
        # Find expected engine (first available in actual priority order)
        expected_engine = None
        for engine in actual_priority:
            if availability.get(engine, False):
                expected_engine = engine
                break
        
        if expected_engine:
            selected_engine, reason = selector.select_engine()
            
            # Should select the highest priority available engine
            assert selected_engine == expected_engine
            assert isinstance(reason, str)
            assert len(reason) > 0
        else:
            # No engines available, should raise RuntimeError
            with pytest.raises(RuntimeError, match="No transcription engines are available"):
                selector.select_engine()

    @given(available_engines=st.lists(engine_types, min_size=1, max_size=3, unique=True))
    @settings(max_examples=30)
    def test_default_priority_order_respected(self, available_engines):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* set of available engines, when using default priority order,
        the selector should prefer local > AWS > API.
        """
        # Create mock factory
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = available_engines
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []  # All available
        
        # Use default configuration (default priority order)
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        selected_engine, reason = selector.select_engine()
        
        # Determine expected engine based on default priority
        default_priority = [
            EngineType.WHISPER_LOCAL.value,
            EngineType.AWS_TRANSCRIBE.value,
            EngineType.WHISPER_API.value
        ]
        
        expected_engine = None
        for engine in default_priority:
            if engine in available_engines:
                expected_engine = engine
                break
        
        assert selected_engine == expected_engine
        assert isinstance(reason, str)

    def test_priority_position_reflected_in_reason(self):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* selected engine, the selection reason should correctly reflect 
        its position in the priority order.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        factory.is_engine_available.return_value = True
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        # Test first priority (highest)
        factory.validate_engine_requirements.return_value = []  # All available
        selected_engine, reason = selector.select_engine()
        
        assert selected_engine == 'local-whisper'  # First in default priority
        assert 'highest priority' in reason
        
        # Test second priority
        def mock_validate_second_priority(engine_type, config):
            if engine_type == 'local-whisper':
                return ['Not available']  # First not available
            return []  # Others available
        
        factory.validate_engine_requirements.side_effect = mock_validate_second_priority
        selected_engine, reason = selector.select_engine()
        
        assert selected_engine == 'aws-transcribe'  # Second in default priority
        assert 'priority 2' in reason
        
        # Test third priority
        def mock_validate_third_priority(engine_type, config):
            if engine_type in ['local-whisper', 'aws-transcribe']:
                return ['Not available']  # First two not available
            return []  # Others available
        
        factory.validate_engine_requirements.side_effect = mock_validate_third_priority
        selected_engine, reason = selector.select_engine()
        
        assert selected_engine == 'openai-whisper'  # Third in default priority
        assert 'priority 3' in reason

    @given(
        priority_order=priority_order_strategy(),
        unavailable_engines=st.lists(engine_types, min_size=0, max_size=2, unique=True)
    )
    @settings(max_examples=40)
    def test_unavailable_engines_skipped(self, priority_order, unavailable_engines):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* priority order with some unavailable engines, the selector should 
        skip unavailable engines and select the next available one.
        """
        # Ensure at least one engine is available
        available_engines = [e for e in priority_order if e not in unavailable_engines]
        assume(len(available_engines) > 0)
        
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = priority_order
        factory.is_engine_available.return_value = True
        
        def mock_validate_requirements(engine_type, config):
            return ['Not available'] if engine_type in unavailable_engines else []
        
        factory.validate_engine_requirements.side_effect = mock_validate_requirements
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(priority_order=priority_order)
        
        selector = AutoSelector(factory, config)
        selected_engine, reason = selector.select_engine()
        
        # Should select first available engine in priority order
        expected_engine = available_engines[0]  # First available
        assert selected_engine == expected_engine
        assert selected_engine not in unavailable_engines

    def test_selection_reason_contains_engine_specific_information(self):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* selected engine, the reason should contain engine-specific information.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        # Test local Whisper reason
        selected_engine, reason = selector.select_engine()
        assert selected_engine == 'local-whisper'
        assert 'local processing preferred for privacy' in reason
        
        # Test API reason (make local unavailable)
        def mock_validate_api_only(engine_type, config):
            return ['Not available'] if engine_type == 'local-whisper' else []
        
        factory.validate_engine_requirements.side_effect = mock_validate_api_only
        selected_engine, reason = selector.select_engine()
        assert selected_engine == 'aws-transcribe'
        assert 'AWS credits available' in reason
        
        # Test OpenAI API reason (make local and AWS unavailable)
        def mock_validate_openai_only(engine_type, config):
            return ['Not available'] if engine_type in ['local-whisper', 'aws-transcribe'] else []
        
        factory.validate_engine_requirements.side_effect = mock_validate_openai_only
        selected_engine, reason = selector.select_engine()
        assert selected_engine == 'openai-whisper'
        assert 'OpenAI API key configured' in reason

    def test_empty_priority_order_uses_default(self):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* configuration with empty priority order, the selector should 
        fall back to default priority order.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []
        
        # Empty priority order
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(priority_order=[])
        
        selector = AutoSelector(factory, config)
        priority = selector.get_selection_priority()
        
        # Should use default order
        expected_default = ['local-whisper', 'aws-transcribe', 'openai-whisper']
        assert priority == expected_default

    @given(
        registered_engines=st.lists(engine_types, min_size=1, max_size=3, unique=True),
        priority_engines=st.lists(engine_types, min_size=1, max_size=3, unique=True)
    )
    @settings(max_examples=30)
    def test_priority_filtered_by_registered_engines(self, registered_engines, priority_engines):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* priority order, only engines registered in the factory should be considered.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = registered_engines
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(priority_order=priority_engines)
        
        selector = AutoSelector(factory, config)
        priority = selector.get_selection_priority()
        
        # All engines in priority should be registered
        for engine in priority:
            assert engine in registered_engines
        
        # Should maintain relative order from priority_engines for registered engines
        filtered_priority = [e for e in priority_engines if e in registered_engines]
        assert priority[:len(filtered_priority)] == filtered_priority

    def test_auto_engine_excluded_from_priority(self):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* priority order, the 'auto' engine should never appear in the priority list.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'auto']
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(
            priority_order=['auto', 'local-whisper', 'openai-whisper']
        )
        
        selector = AutoSelector(factory, config)
        priority = selector.get_selection_priority()
        
        # 'auto' should not appear in priority order
        assert 'auto' not in priority
        assert 'local-whisper' in priority
        assert 'openai-whisper' in priority

    @given(
        all_engines_available=st.booleans(),
        some_engines_available=st.booleans()
    )
    @settings(max_examples=20)
    def test_availability_checking_methods_consistency(self, all_engines_available, some_engines_available):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* availability scenario, individual availability checking methods should 
        be consistent with the overall selection logic.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        factory.is_engine_available.return_value = True
        
        # Set up availability scenario
        if all_engines_available:
            factory.validate_engine_requirements.return_value = []
        elif some_engines_available:
            def mock_validate_some(engine_type, config):
                return [] if engine_type == 'local-whisper' else ['Not available']
            factory.validate_engine_requirements.side_effect = mock_validate_some
        else:
            factory.validate_engine_requirements.return_value = ['Not available']
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        # Check individual methods
        local_available = selector.check_local_whisper_availability()
        api_available = selector.check_api_key_availability()
        aws_available = selector.check_aws_credentials_availability()
        
        if all_engines_available:
            assert local_available is True
            assert api_available is True
            assert aws_available is True
            
            # Should be able to select an engine
            selected_engine, reason = selector.select_engine()
            assert selected_engine is not None
            
        elif some_engines_available:
            assert local_available is True
            assert api_available is False
            assert aws_available is False
            
            # Should select the available engine
            selected_engine, reason = selector.select_engine()
            assert selected_engine == 'local-whisper'
            
        else:
            assert local_available is False
            assert api_available is False
            assert aws_available is False
            
            # Should raise error
            with pytest.raises(RuntimeError):
                selector.select_engine()

    def test_selection_deterministic_for_same_inputs(self):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* identical configuration and availability scenario, the selector 
        should always return the same result.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        factory.is_engine_available.return_value = True
        
        # Make only whisper-api available
        def mock_validate_api_only(engine_type, config):
            return [] if engine_type == 'openai-whisper' else ['Not available']
        
        factory.validate_engine_requirements.side_effect = mock_validate_api_only
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(
            priority_order=['local-whisper', 'openai-whisper', 'aws-transcribe']
        )
        
        selector = AutoSelector(factory, config)
        
        # Multiple selections should return the same result
        results = []
        for _ in range(5):
            selected_engine, reason = selector.select_engine()
            results.append((selected_engine, reason))
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result
        
        assert first_result[0] == 'openai-whisper'

    def test_priority_order_validation_properties(self):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* priority order configuration, validation should catch invalid engines
        and provide helpful error messages.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper']
        
        # Test with invalid engine in priority order
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(
            priority_order=['local-whisper', 'invalid-engine', 'openai-whisper']
        )
        
        selector = AutoSelector(factory, config)
        errors = selector.validate_selection_preferences()
        
        assert len(errors) > 0
        assert any('invalid-engine' in error for error in errors)
        assert any('Invalid engine in priority order' in error for error in errors)

    @given(
        priority_order=priority_order_strategy(),
        factory_engines=st.lists(engine_types, min_size=1, max_size=3, unique=True)
    )
    @settings(max_examples=30)
    def test_extensibility_with_additional_engines(self, priority_order, factory_engines):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* priority order and factory engines, engines not in priority order 
        should be appended for extensibility.
        """
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = factory_engines
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(priority_order=priority_order)
        
        selector = AutoSelector(factory, config)
        final_priority = selector.get_selection_priority()
        
        # All factory engines (except 'auto') should be in final priority
        for engine in factory_engines:
            if engine != 'auto':
                assert engine in final_priority
        
        # Priority order should be maintained for engines that were specified
        specified_engines = [e for e in priority_order if e in factory_engines]
        for i, engine in enumerate(specified_engines):
            assert final_priority.index(engine) == i


class TestAutoSelectionIntegrationProperties:
    """Integration property tests for auto-selection with real components."""

    def test_real_factory_integration_properties(self):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* real factory configuration, auto-selection should work with 
        actual registered engines.
        """
        from pipeline.transcribers.factory import EngineFactory
        
        factory = EngineFactory()
        config = TranscriptionConfig()
        
        selector = AutoSelector(factory, config)
        
        # Should get a valid priority order
        priority = selector.get_selection_priority()
        assert isinstance(priority, list)
        assert len(priority) > 0
        
        # All engines in priority should be registered in factory
        available_engines = factory.get_available_engines()
        for engine in priority:
            assert engine in available_engines
        
        # Should be able to get capabilities for all engines
        for engine in priority:
            capabilities = selector.get_engine_capabilities(engine)
            assert isinstance(capabilities, dict)
            assert 'available' in capabilities

    def test_selection_summary_properties(self):
        """
        **Property 5: Auto-Selection Priority Logic**
        *For any* configuration, the selection summary should provide complete 
        information about all engines and selection logic.
        """
        from pipeline.transcribers.factory import EngineFactory
        
        factory = EngineFactory()
        config = TranscriptionConfig()
        
        selector = AutoSelector(factory, config)
        summary = selector.get_selection_summary()
        
        # Summary should have required structure
        assert isinstance(summary, dict)
        assert 'priority_order' in summary
        assert 'engines' in summary
        assert isinstance(summary['priority_order'], list)
        assert isinstance(summary['engines'], dict)
        
        # Should have information for all engines in priority order
        for engine in summary['priority_order']:
            assert engine in summary['engines']
            engine_info = summary['engines'][engine]
            assert isinstance(engine_info, dict)
            assert 'available' in engine_info