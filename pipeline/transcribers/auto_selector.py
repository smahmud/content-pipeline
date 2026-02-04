"""
File: auto_selector.py

Intelligent engine selection based on availability and configuration.

Enhanced in v0.6.5 to support:
- Priority-based engine selection (local > API > error)
- Configuration-driven preferences
- Availability checking with requirement validation
- Selection reasoning and logging

DEPRECATED: This module is deprecated. Use pipeline.transcription.TranscriptionProviderFactory instead.
"""
import logging
from typing import Tuple, List, Optional
from pipeline.transcribers.factory import EngineFactory
from pipeline.config.schema import TranscriptionConfig, EngineType


logger = logging.getLogger(__name__)


class AutoSelector:
    """
    Intelligent engine selection based on availability and configuration.
    
    Evaluates available engines in priority order and selects the best option
    based on user preferences, engine availability, and system capabilities.
    
    DEPRECATED: Use pipeline.transcription.TranscriptionProviderFactory instead.
    """
    
    def __init__(self, factory: EngineFactory, config: TranscriptionConfig):
        """
        Initialize the AutoSelector.
        
        Args:
            factory: EngineFactory instance for creating and validating engines
            config: Complete transcription configuration
        """
        self.factory = factory
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def select_engine(self) -> Tuple[str, str]:
        """
        Select the best available engine and return selection details.
        
        Returns:
            Tuple of (engine_type, selection_reason)
            
        Raises:
            RuntimeError: If no engines are available
        """
        priority_order = self.get_selection_priority()
        
        self.logger.debug(f"Auto-selection priority order: {priority_order}")
        
        for engine_type in priority_order:
            if self._is_engine_available(engine_type):
                reason = self._get_selection_reason(engine_type, priority_order)
                self.logger.info(f"Auto-selected engine: {engine_type} ({reason})")
                return engine_type, reason
        
        # No engines available
        error_msg = self._generate_no_engines_error(priority_order)
        self.logger.error(f"Auto-selection failed: {error_msg}")
        raise RuntimeError(error_msg)
    
    def get_selection_priority(self) -> List[str]:
        """
        Return engine selection priority order based on configuration.
        
        Returns:
            List of engine types in priority order
        """
        if self.config.auto_selection.priority_order:
            # Use configured priority order
            priority = self.config.auto_selection.priority_order.copy()
        else:
            # Use default priority order
            priority = [
                EngineType.WHISPER_LOCAL.value,
                EngineType.AWS_TRANSCRIBE.value,
                EngineType.WHISPER_API.value
            ]
        
        # Filter out engines that aren't registered in the factory and exclude 'auto'
        available_engines = self.factory.get_available_engines()
        filtered_priority = [
            engine for engine in priority 
            if engine in available_engines and engine != EngineType.AUTO.value
        ]
        
        # Add any registered engines not in the priority list (for extensibility)
        for engine in available_engines:
            if engine not in filtered_priority and engine != EngineType.AUTO.value:
                filtered_priority.append(engine)
        
        return filtered_priority
    
    def check_local_whisper_availability(self) -> bool:
        """
        Check if local Whisper is available and functional.
        
        Returns:
            True if local Whisper can be used, False otherwise
        """
        return self._is_engine_available(EngineType.WHISPER_LOCAL.value)
    
    def check_api_key_availability(self) -> bool:
        """
        Check if OpenAI API key is available and valid.
        
        Returns:
            True if API key is configured and valid, False otherwise
        """
        return self._is_engine_available(EngineType.WHISPER_API.value)
    
    def check_aws_credentials_availability(self) -> bool:
        """
        Check if AWS credentials are available and valid.
        
        Returns:
            True if AWS credentials are configured, False otherwise
        """
        return self._is_engine_available(EngineType.AWS_TRANSCRIBE.value)
    
    def _is_engine_available(self, engine_type: str) -> bool:
        """
        Check if an engine is available and meets all requirements.
        
        Args:
            engine_type: Type of engine to check
            
        Returns:
            True if engine is available and functional, False otherwise
        """
        try:
            # Check if engine is registered in factory
            if not self.factory.is_engine_available(engine_type):
                self.logger.debug(f"Engine {engine_type} not registered in factory")
                return False
            
            # Validate engine requirements
            errors = self.factory.validate_engine_requirements(engine_type, self.config)
            if errors:
                self.logger.debug(f"Engine {engine_type} requirements not met: {errors}")
                return False
            
            self.logger.debug(f"Engine {engine_type} is available and functional")
            return True
            
        except Exception as e:
            self.logger.debug(f"Error checking engine {engine_type} availability: {e}")
            return False
    
    def _get_selection_reason(self, selected_engine: str, priority_order: List[str]) -> str:
        """
        Generate a human-readable reason for engine selection.
        
        Args:
            selected_engine: The engine that was selected
            priority_order: The priority order used for selection
            
        Returns:
            Human-readable selection reason
        """
        position = priority_order.index(selected_engine) + 1
        total = len(priority_order)
        
        reasons = {
            EngineType.WHISPER_LOCAL.value: "local processing preferred for privacy",
            EngineType.AWS_TRANSCRIBE.value: "AWS credits available",
            EngineType.WHISPER_API.value: "OpenAI API key configured"
        }
        
        base_reason = reasons.get(selected_engine, "engine available")
        
        if position == 1:
            return f"highest priority engine, {base_reason}"
        else:
            return f"priority {position}/{total}, {base_reason}"
    
    def _generate_no_engines_error(self, attempted_engines: List[str]) -> str:
        """
        Generate a comprehensive error message when no engines are available.
        
        Args:
            attempted_engines: List of engines that were attempted
            
        Returns:
            Detailed error message with setup instructions
        """
        error_parts = ["No transcription engines are available."]
        
        setup_instructions = []
        
        if EngineType.WHISPER_LOCAL.value in attempted_engines:
            local_errors = self.factory.validate_engine_requirements(
                EngineType.WHISPER_LOCAL.value, self.config
            )
            if local_errors:
                setup_instructions.append(
                    f"For local Whisper:\n  " + "\n  ".join(local_errors)
                )
        
        if EngineType.WHISPER_API.value in attempted_engines:
            api_errors = self.factory.validate_engine_requirements(
                EngineType.WHISPER_API.value, self.config
            )
            if api_errors:
                setup_instructions.append(
                    f"For OpenAI API:\n  " + "\n  ".join(api_errors)
                )
        
        if EngineType.AWS_TRANSCRIBE.value in attempted_engines:
            aws_errors = self.factory.validate_engine_requirements(
                EngineType.AWS_TRANSCRIBE.value, self.config
            )
            if aws_errors:
                setup_instructions.append(
                    f"For AWS Transcribe:\n  " + "\n  ".join(aws_errors)
                )
        
        if setup_instructions:
            error_parts.append("\nSetup instructions:")
            error_parts.extend(setup_instructions)
        else:
            error_parts.append("\nEnsure at least one transcription engine is properly configured.")
        
        return "\n".join(error_parts)
    
    def get_engine_capabilities(self, engine_type: str) -> dict:
        """
        Get capabilities and limitations of a specific engine.
        
        Args:
            engine_type: Type of engine to query
            
        Returns:
            Dictionary with engine capabilities
        """
        if not self.factory.is_engine_available(engine_type):
            return {"available": False, "error": "Engine not registered"}
        
        try:
            # Create a temporary adapter to query capabilities
            adapter = self.factory.create_engine(engine_type, self.config)
            
            return {
                "available": True,
                "supported_formats": adapter.get_supported_formats(),
                "cost_per_minute": adapter.estimate_cost(60.0),  # Cost for 1 minute
                "engine_info": adapter.get_engine_info(),
                "is_free": adapter.estimate_cost(60.0) is None or adapter.estimate_cost(60.0) == 0.0
            }
            
        except Exception as e:
            errors = self.factory.validate_engine_requirements(engine_type, self.config)
            return {
                "available": False,
                "error": str(e),
                "requirements_errors": errors
            }
    
    def get_selection_summary(self) -> dict:
        """
        Get a summary of all engines and their availability status.
        
        Returns:
            Dictionary with engine availability summary
        """
        priority_order = self.get_selection_priority()
        summary = {
            "priority_order": priority_order,
            "engines": {},
            "recommended_engine": None,
            "selection_reason": None
        }
        
        for engine_type in priority_order:
            summary["engines"][engine_type] = self.get_engine_capabilities(engine_type)
        
        # Try to get recommended engine
        try:
            recommended, reason = self.select_engine()
            summary["recommended_engine"] = recommended
            summary["selection_reason"] = reason
        except RuntimeError as e:
            summary["selection_error"] = str(e)
        
        return summary
    
    def validate_selection_preferences(self) -> List[str]:
        """
        Validate auto-selection configuration preferences.
        
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Validate priority order contains valid engines
        available_engines = self.factory.get_available_engines()
        for engine in self.config.auto_selection.priority_order:
            if engine not in available_engines and engine != EngineType.AUTO.value:
                errors.append(f"Invalid engine in priority order: '{engine}'. Available: {available_engines}")
        
        # Check if at least one engine in priority order is potentially available
        has_potentially_available = False
        for engine in self.config.auto_selection.priority_order:
            if engine in available_engines:
                has_potentially_available = True
                break
        
        if not has_potentially_available:
            errors.append("No engines in priority order are registered in the factory")
        
        return errors