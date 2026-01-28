"""
Unit tests for AutoSelector intelligent engine selection.

Tests cover:
- Priority-based engine selection logic
- Availability checking and requirement validation
- Configuration-driven preferences
- Error handling and messaging
- Selection reasoning and logging
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pipeline.transcribers.auto_selector import AutoSelector
from pipeline.transcribers.factory import EngineFactory
from pipeline.config.schema import (
    TranscriptionConfig, AutoSelectionConfig, 
    WhisperLocalConfig, WhisperAPIConfig, AWSTranscribeConfig,
    EngineType
)


class TestAutoSelector:
    """Test the AutoSelector class."""

    def test_initialization(self):
        """Test AutoSelector initialization."""
        factory = Mock(spec=EngineFactory)
        config = TranscriptionConfig()
        
        selector = AutoSelector(factory, config)
        
        assert selector.factory is factory
        assert selector.config is config
        assert hasattr(selector, 'logger')

    def test_get_selection_priority_default_order(self):
        """Test default priority order when none configured."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig()  # Default priority order
        
        selector = AutoSelector(factory, config)
        priority = selector.get_selection_priority()
        
        # Should use default order and filter by available engines
        expected = ['local-whisper', 'aws-transcribe', 'openai-whisper']
        assert priority == expected

    def test_get_selection_priority_custom_order(self):
        """Test custom priority order from configuration."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(
            priority_order=['openai-whisper', 'local-whisper', 'aws-transcribe']
        )
        
        selector = AutoSelector(factory, config)
        priority = selector.get_selection_priority()
        
        # Should use custom order
        expected = ['openai-whisper', 'local-whisper', 'aws-transcribe']
        assert priority == expected

    def test_get_selection_priority_filters_unavailable_engines(self):
        """Test that priority order filters out unavailable engines."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper']  # No AWS
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(
            priority_order=['local-whisper', 'aws-transcribe', 'openai-whisper']
        )
        
        selector = AutoSelector(factory, config)
        priority = selector.get_selection_priority()
        
        # Should filter out aws-transcribe since it's not available
        expected = ['local-whisper', 'openai-whisper']
        assert priority == expected

    def test_select_engine_first_available(self):
        """Test selecting the first available engine in priority order."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper']
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []  # No errors
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        engine_type, reason = selector.select_engine()
        
        assert engine_type == 'local-whisper'
        assert 'highest priority' in reason
        assert 'local processing preferred' in reason

    def test_select_engine_fallback_to_second_choice(self):
        """Test falling back to second choice when first is unavailable."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper']
        
        # Mock first engine unavailable, second available
        def mock_validate_requirements(engine_type, config):
            if engine_type == 'local-whisper':
                return ['Whisper not installed']  # Has errors
            return []  # No errors for other engines
        
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.side_effect = mock_validate_requirements
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        engine_type, reason = selector.select_engine()
        
        assert engine_type == 'openai-whisper'
        assert 'priority 2' in reason
        assert 'OpenAI API key configured' in reason

    def test_select_engine_no_engines_available(self):
        """Test error when no engines are available."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper']
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = ['Requirements not met']  # All have errors
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        with pytest.raises(RuntimeError, match="No transcription engines are available"):
            selector.select_engine()

    def test_check_local_whisper_availability_true(self):
        """Test checking local Whisper availability when available."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []  # No errors
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        assert selector.check_local_whisper_availability() is True
        factory.is_engine_available.assert_called_with('local-whisper')

    def test_check_local_whisper_availability_false(self):
        """Test checking local Whisper availability when unavailable."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = ['Whisper not installed']  # Has errors
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        assert selector.check_local_whisper_availability() is False

    def test_check_api_key_availability_true(self):
        """Test checking API key availability when available."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []  # No errors
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        assert selector.check_api_key_availability() is True
        factory.is_engine_available.assert_called_with('openai-whisper')

    def test_check_api_key_availability_false(self):
        """Test checking API key availability when unavailable."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = ['API key not found']  # Has errors
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        assert selector.check_api_key_availability() is False

    def test_check_aws_credentials_availability_true(self):
        """Test checking AWS credentials availability when available."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []  # No errors
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        assert selector.check_aws_credentials_availability() is True
        factory.is_engine_available.assert_called_with('aws-transcribe')

    def test_check_aws_credentials_availability_false(self):
        """Test checking AWS credentials availability when unavailable."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = ['AWS credentials not found']  # Has errors
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        assert selector.check_aws_credentials_availability() is False

    def test_get_engine_capabilities_available_engine(self):
        """Test getting capabilities for an available engine."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        
        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.get_supported_formats.return_value = ['mp3', 'wav']
        mock_adapter.estimate_cost.return_value = 0.006
        mock_adapter.get_engine_info.return_value = ('openai-whisper', 'whisper-1')
        
        factory.create_engine.return_value = mock_adapter
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        capabilities = selector.get_engine_capabilities('openai-whisper')
        
        assert capabilities['available'] is True
        assert capabilities['supported_formats'] == ['mp3', 'wav']
        assert capabilities['cost_per_minute'] == 0.006
        assert capabilities['engine_info'] == ('openai-whisper', 'whisper-1')
        assert capabilities['is_free'] is False

    def test_get_engine_capabilities_free_engine(self):
        """Test getting capabilities for a free engine."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        
        # Mock adapter for free engine
        mock_adapter = Mock()
        mock_adapter.get_supported_formats.return_value = ['mp3', 'wav']
        mock_adapter.estimate_cost.return_value = None  # Free engine
        mock_adapter.get_engine_info.return_value = ('local-whisper', 'base')
        
        factory.create_engine.return_value = mock_adapter
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        capabilities = selector.get_engine_capabilities('local-whisper')
        
        assert capabilities['available'] is True
        assert capabilities['is_free'] is True

    def test_get_engine_capabilities_unavailable_engine(self):
        """Test getting capabilities for an unavailable engine."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = False
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        capabilities = selector.get_engine_capabilities('nonexistent-engine')
        
        assert capabilities['available'] is False
        assert 'Engine not registered' in capabilities['error']

    def test_get_engine_capabilities_engine_creation_fails(self):
        """Test getting capabilities when engine creation fails."""
        factory = Mock(spec=EngineFactory)
        factory.is_engine_available.return_value = True
        factory.create_engine.side_effect = RuntimeError("Engine requirements not met")
        factory.validate_engine_requirements.return_value = ['Missing dependency']
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        capabilities = selector.get_engine_capabilities('local-whisper')
        
        assert capabilities['available'] is False
        assert 'Engine requirements not met' in capabilities['error']
        assert capabilities['requirements_errors'] == ['Missing dependency']

    def test_get_selection_summary(self):
        """Test getting a complete selection summary."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper']
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []
        
        # Mock adapter capabilities
        mock_adapter = Mock()
        mock_adapter.get_supported_formats.return_value = ['mp3', 'wav']
        mock_adapter.estimate_cost.return_value = None
        mock_adapter.get_engine_info.return_value = ('local-whisper', 'base')
        factory.create_engine.return_value = mock_adapter
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        summary = selector.get_selection_summary()
        
        assert 'priority_order' in summary
        assert 'engines' in summary
        assert 'recommended_engine' in summary
        assert 'selection_reason' in summary
        
        assert summary['recommended_engine'] == 'local-whisper'
        assert len(summary['engines']) == 2

    def test_get_selection_summary_no_engines_available(self):
        """Test selection summary when no engines are available."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper']
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = ['Requirements not met']
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        summary = selector.get_selection_summary()
        
        assert 'selection_error' in summary
        assert 'No transcription engines are available' in summary['selection_error']

    def test_validate_selection_preferences_valid(self):
        """Test validating valid selection preferences."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(
            priority_order=['local-whisper', 'openai-whisper']
        )
        
        selector = AutoSelector(factory, config)
        errors = selector.validate_selection_preferences()
        
        assert errors == []

    def test_validate_selection_preferences_invalid_engine(self):
        """Test validating preferences with invalid engine."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper']
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(
            priority_order=['local-whisper', 'nonexistent-engine']
        )
        
        selector = AutoSelector(factory, config)
        errors = selector.validate_selection_preferences()
        
        assert len(errors) == 1
        assert 'Invalid engine in priority order' in errors[0]
        assert 'nonexistent-engine' in errors[0]

    def test_validate_selection_preferences_no_available_engines(self):
        """Test validating preferences when no engines in priority are available."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper']
        
        config = TranscriptionConfig()
        config.auto_selection = AutoSelectionConfig(
            priority_order=['nonexistent-engine']
        )
        
        selector = AutoSelector(factory, config)
        errors = selector.validate_selection_preferences()
        
        assert len(errors) == 2  # Invalid engine + no available engines
        assert any('Invalid engine' in error for error in errors)
        assert any('No engines in priority order are registered' in error for error in errors)

    @patch('pipeline.transcribers.auto_selector.logger')
    def test_logging_during_selection(self, mock_logger):
        """Test that appropriate logging occurs during selection."""
        factory = Mock(spec=EngineFactory)
        factory.get_available_engines.return_value = ['local-whisper', 'openai-whisper']
        factory.is_engine_available.return_value = True
        factory.validate_engine_requirements.return_value = []
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        # Mock the logger on the selector instance
        selector.logger = Mock()
        
        engine_type, reason = selector.select_engine()
        
        # Should log debug and info messages
        selector.logger.debug.assert_called()
        selector.logger.info.assert_called()
        
        # Check that info log contains selection details
        info_calls = selector.logger.info.call_args_list
        assert any('Auto-selected engine' in str(call) for call in info_calls)

    def test_generate_no_engines_error_comprehensive(self):
        """Test comprehensive error message generation."""
        factory = Mock(spec=EngineFactory)
        factory.validate_engine_requirements.side_effect = lambda engine, config: {
            'local-whisper': ['Whisper package not installed'],
            'openai-whisper': ['OpenAI API key not found'],
            'aws-transcribe': ['AWS credentials not configured']
        }.get(engine, [])
        
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        attempted_engines = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        error_msg = selector._generate_no_engines_error(attempted_engines)
        
        assert 'No transcription engines are available' in error_msg
        assert 'Setup instructions:' in error_msg
        assert 'For local Whisper:' in error_msg
        assert 'Whisper package not installed' in error_msg
        assert 'For OpenAI API:' in error_msg
        assert 'OpenAI API key not found' in error_msg
        assert 'For AWS Transcribe:' in error_msg
        assert 'AWS credentials not configured' in error_msg

    def test_selection_reason_generation(self):
        """Test selection reason generation for different engines."""
        factory = Mock(spec=EngineFactory)
        config = TranscriptionConfig()
        selector = AutoSelector(factory, config)
        
        priority_order = ['local-whisper', 'openai-whisper', 'aws-transcribe']
        
        # Test first priority
        reason = selector._get_selection_reason('local-whisper', priority_order)
        assert 'highest priority' in reason
        assert 'local processing preferred' in reason
        
        # Test second priority
        reason = selector._get_selection_reason('openai-whisper', priority_order)
        assert 'priority 2/3' in reason
        assert 'OpenAI API key configured' in reason
        
        # Test third priority
        reason = selector._get_selection_reason('aws-transcribe', priority_order)
        assert 'priority 3/3' in reason
        assert 'AWS credits available' in reason


class TestAutoSelectorIntegration:
    """Integration tests for AutoSelector with real factory."""

    def test_integration_with_real_factory(self):
        """Test AutoSelector integration with real EngineFactory."""
        from pipeline.transcribers.factory import EngineFactory
        
        factory = EngineFactory()
        config = TranscriptionConfig()
        
        selector = AutoSelector(factory, config)
        
        # Should be able to get priority order
        priority = selector.get_selection_priority()
        assert isinstance(priority, list)
        assert len(priority) > 0
        
        # Should be able to get capabilities for registered engines
        available_engines = factory.get_available_engines()
        for engine in available_engines:
            if engine != 'auto':  # Skip auto engine
                capabilities = selector.get_engine_capabilities(engine)
                assert isinstance(capabilities, dict)
                assert 'available' in capabilities

    def test_integration_selection_with_mocked_requirements(self):
        """Test integration selection with mocked requirements validation."""
        from pipeline.transcribers.factory import EngineFactory
        
        factory = EngineFactory()
        config = TranscriptionConfig()
        
        selector = AutoSelector(factory, config)
        
        # Mock all engines as having unmet requirements
        with patch.object(factory, 'validate_engine_requirements', return_value=['Mocked error']):
            with pytest.raises(RuntimeError, match="No transcription engines are available"):
                selector.select_engine()
        
        # Mock first engine as available
        def mock_validate(engine_type, config):
            if engine_type == 'local-whisper':
                return []  # No errors
            return ['Mocked error']
        
        with patch.object(factory, 'validate_engine_requirements', side_effect=mock_validate):
            engine_type, reason = selector.select_engine()
            assert engine_type == 'local-whisper'
            assert isinstance(reason, str)