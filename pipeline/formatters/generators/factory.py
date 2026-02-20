"""
Generator factory for creating output type formatters.

The factory manages registration and instantiation of generators,
providing a central point for accessing all output type formatters.
"""

from typing import Optional, Type

from pipeline.formatters.base import BaseFormatter, OutputType
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.template_engine import TemplateEngine
from pipeline.formatters.validator import PlatformValidator


class GeneratorFactoryError(Exception):
    """Exception raised for generator factory errors."""
    
    pass


class GeneratorFactory:
    """Factory for creating and managing output type generators.
    
    The factory maintains a registry of generator classes and creates
    instances on demand with shared configuration.
    
    Usage:
        factory = GeneratorFactory()
        blog_gen = factory.get_generator("blog")
        result = blog_gen.format(request)
    """
    
    # Registry of generator classes by output type
    _registry: dict[str, Type[BaseGenerator]] = {}
    
    def __init__(
        self,
        template_engine: Optional[TemplateEngine] = None,
        platform_validator: Optional[PlatformValidator] = None,
        auto_truncate: bool = True,
    ):
        """Initialize the factory.
        
        Args:
            template_engine: TemplateEngine instance (created if not provided)
            platform_validator: PlatformValidator instance (created if not provided)
            auto_truncate: Whether generators should auto-truncate content
        """
        self._template_engine = template_engine or TemplateEngine()
        self._platform_validator = platform_validator or PlatformValidator()
        self._auto_truncate = auto_truncate
        
        # Cache of instantiated generators
        self._instances: dict[str, BaseGenerator] = {}
    
    @classmethod
    def register(cls, output_type: str):
        """Decorator to register a generator class.
        
        Usage:
            @GeneratorFactory.register("blog")
            class BlogGenerator(BaseGenerator):
                ...
        
        Args:
            output_type: The output type this generator handles
            
        Returns:
            Decorator function
        """
        def decorator(generator_class: Type[BaseGenerator]) -> Type[BaseGenerator]:
            cls._registry[output_type] = generator_class
            return generator_class
        return decorator
    
    @classmethod
    def register_generator(
        cls,
        output_type: str,
        generator_class: Type[BaseGenerator],
    ) -> None:
        """Register a generator class for an output type.
        
        Args:
            output_type: The output type this generator handles
            generator_class: The generator class to register
        """
        cls._registry[output_type] = generator_class
    
    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get list of registered output types.
        
        Returns:
            List of output type strings
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, output_type: str) -> bool:
        """Check if an output type has a registered generator.
        
        Args:
            output_type: The output type to check
            
        Returns:
            True if generator is registered
        """
        return output_type in cls._registry
    
    def get_generator(self, output_type: str) -> BaseGenerator:
        """Get a generator instance for the specified output type.
        
        Generators are cached after first creation.
        
        Args:
            output_type: The output type to get generator for
            
        Returns:
            Generator instance
            
        Raises:
            GeneratorFactoryError: If no generator registered for output type
        """
        # Normalize output type
        output_type = output_type.lower()
        
        # Check cache first
        if output_type in self._instances:
            return self._instances[output_type]
        
        # Check registry
        if output_type not in self._registry:
            raise GeneratorFactoryError(
                f"No generator registered for output type: {output_type}. "
                f"Available types: {list(self._registry.keys())}"
            )
        
        # Create config
        config = GeneratorConfig(
            template_engine=self._template_engine,
            platform_validator=self._platform_validator,
            auto_truncate=self._auto_truncate,
        )
        
        # Instantiate and cache
        generator_class = self._registry[output_type]
        generator = generator_class(config)
        self._instances[output_type] = generator
        
        return generator
    
    def get_all_generators(self) -> dict[str, BaseGenerator]:
        """Get all registered generators.
        
        Returns:
            Dictionary mapping output type to generator instance
        """
        generators = {}
        for output_type in self._registry:
            generators[output_type] = self.get_generator(output_type)
        return generators
    
    def clear_cache(self) -> None:
        """Clear the generator instance cache.
        
        Useful for testing or when configuration changes.
        """
        self._instances.clear()


def register_all_generators() -> None:
    """Register all built-in generators.
    
    This function imports all generator modules to trigger their
    @GeneratorFactory.register decorators.
    """
    # Import all generator modules to trigger registration
    # These imports have side effects (registration)
    from pipeline.formatters.generators import (
        blog,
        tweet,
        youtube,
        seo,
        linkedin,
        newsletter,
        chapters,
        transcript_clean,
        podcast_notes,
        meeting_minutes,
        slides,
        notion,
        obsidian,
        quote_cards,
        video_script,
        tiktok_script,
        ai_video_script,
    )
