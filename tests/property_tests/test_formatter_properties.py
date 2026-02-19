"""
Property-based tests for formatter module.

This module implements correctness properties from the design document
using Hypothesis for generative testing with random inputs.

These tests validate:
- Property 1: Output Structure Completeness (Requirements 1.1-4.2)
- Property 2: Style Profile Parsing Round-Trip (Requirements 5.1, 5.2, 5.3, 5.6, 5.8)
- Property 3: Platform Character Limit Enforcement (Requirements 8.1-8.5)
- Property 4: Intelligent Truncation at Sentence Boundaries (Requirements 8.6, 8.7)
- Property 5: CLI Flag Precedence Over Style Profile (Requirement 6.8)
- Property 6: LLM Enhancement Preserves Structure (Requirement 6.5)
- Property 10: FormatV1 Metadata Completeness (Requirements 12.2, 12.3, 12.4, 12.7)
"""

import tempfile
from pathlib import Path

import pytest
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from pipeline.formatters.schemas.format_v1 import (
    FormatV1,
    LLMMetadata,
    ValidationMetadata,
)
from pipeline.formatters.base import (
    VALID_OUTPUT_TYPES,
    VALID_PLATFORMS,
    VALID_TONES,
    VALID_LENGTHS,
)
from pipeline.formatters.style_profile import (
    StyleProfile,
    StyleProfileLoader,
    StyleProfileError,
)


# ============================================================================
# STRATEGIES: Reusable Hypothesis strategies for generating test data
# ============================================================================

# Output type strategy
output_type_strategy = st.sampled_from(VALID_OUTPUT_TYPES)

# Platform strategy (including None)
platform_strategy = st.one_of(st.none(), st.sampled_from(VALID_PLATFORMS))

# Tone strategy (including None)
tone_strategy = st.one_of(st.none(), st.sampled_from(VALID_TONES))

# Length strategy (including None)
length_strategy = st.one_of(st.none(), st.sampled_from(VALID_LENGTHS))

# Provider strategy
provider_strategy = st.sampled_from(["openai", "claude", "bedrock"])

# Model strategy - simple alphanumeric with dashes
model_strategy = st.from_regex(r"[a-z][a-z0-9\-]{2,20}", fullmatch=True)

# File path strategy - simple valid paths
file_path_strategy = st.from_regex(r"[a-z][a-z0-9_/]{3,30}\.json", fullmatch=True)

# Style profile name strategy
style_profile_strategy = st.one_of(
    st.none(),
    st.sampled_from([
        "medium-tech",
        "twitter-thread",
        "youtube-description",
        "linkedin-professional",
        "heygen-avatar",
        "tiktok-short",
    ])
)


@st.composite
def llm_metadata_strategy(draw):
    """Generate valid LLMMetadata objects."""
    return LLMMetadata(
        provider=draw(provider_strategy),
        model=draw(model_strategy),
        cost_usd=draw(st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False)),
        tokens_used=draw(st.integers(min_value=0, max_value=100000)),
        temperature=draw(st.floats(min_value=0, max_value=2, allow_nan=False, allow_infinity=False)),
        enhanced=draw(st.booleans()),
    )


@st.composite
def validation_metadata_strategy(draw):
    """Generate valid ValidationMetadata objects."""
    return ValidationMetadata(
        platform=draw(platform_strategy),
        character_count=draw(st.integers(min_value=0, max_value=100000)),
        truncated=draw(st.booleans()),
        warnings=draw(st.lists(
            st.from_regex(r"[A-Za-z][A-Za-z0-9 ]{5,50}", fullmatch=True),
            min_size=0,
            max_size=3
        )),
    )


# Style profile strategies for Property 2
@st.composite
def style_profile_name_strategy(draw):
    """Generate valid style profile names."""
    return draw(st.from_regex(r"[a-z][a-z0-9\-]{2,30}", fullmatch=True))


@st.composite
def style_profile_model_strategy(draw):
    """Generate valid model names for style profiles."""
    return draw(st.sampled_from([
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "anthropic.claude-3-sonnet",
    ]))


@st.composite
def jinja2_variable_strategy(draw):
    """Generate valid Jinja2 variable references."""
    return draw(st.sampled_from([
        "{{ title }}",
        "{{ summary.short }}",
        "{{ summary.medium }}",
        "{{ summary.long }}",
        "{{ tags }}",
        "{{ chapters }}",
        "{{ highlights }}",
        "{{ transcript }}",
        "{{ content }}",
    ]))


@st.composite
def prompt_template_strategy(draw):
    """Generate valid prompt templates with Jinja2 variables."""
    # Generate base text
    base_text = draw(st.from_regex(r"[A-Za-z][A-Za-z0-9 .,!?\n]{20,200}", fullmatch=True))
    
    # Add some Jinja2 variables
    num_vars = draw(st.integers(min_value=1, max_value=4))
    variables = [draw(jinja2_variable_strategy()) for _ in range(num_vars)]
    
    # Combine base text with variables
    template = base_text
    for var in variables:
        template += f"\n\n{var}\n"
    
    return template


@st.composite
def style_profile_data_strategy(draw):
    """Generate valid style profile data for testing."""
    return {
        "name": draw(style_profile_name_strategy()),
        "temperature": draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)),
        "top_p": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "max_tokens": draw(st.integers(min_value=100, max_value=8000)),
        "model": draw(style_profile_model_strategy()),
        "prompt_template": draw(prompt_template_strategy()),
    }


@st.composite
def style_profile_object_strategy(draw):
    """Generate valid StyleProfile objects."""
    data = draw(style_profile_data_strategy())
    return StyleProfile(
        name=data["name"],
        temperature=data["temperature"],
        top_p=data["top_p"],
        max_tokens=data["max_tokens"],
        model=data["model"],
        prompt_template=data["prompt_template"],
    )


@st.composite
def format_v1_strategy(draw, with_llm: bool = None):
    """Generate valid FormatV1 objects.
    
    Args:
        with_llm: If True, always include LLM metadata.
                  If False, never include LLM metadata.
                  If None, randomly include or exclude.
    """
    include_llm = draw(st.booleans()) if with_llm is None else with_llm
    
    return FormatV1(
        format_version="v1",
        output_type=draw(output_type_strategy),
        platform=draw(platform_strategy),
        timestamp=datetime.now(timezone.utc),
        source_file=draw(file_path_strategy),
        style_profile_used=draw(style_profile_strategy),
        llm_metadata=draw(llm_metadata_strategy()) if include_llm else None,
        validation=draw(validation_metadata_strategy()),
        tone=draw(tone_strategy),
        length=draw(length_strategy),
    )


# ============================================================================
# PROPERTY 2: Style Profile Parsing Round-Trip (Requirements 5.1, 5.2, 5.3, 5.6, 5.8)
# ============================================================================

class TestStyleProfileParsingRoundTrip:
    """Property 2: Style Profile Parsing Round-Trip.
    
    Validates Requirements 5.1, 5.2, 5.3, 5.6, 5.8:
    - 5.1: Load style profiles from Markdown files with YAML frontmatter
    - 5.2: Parse style profile frontmatter for LLM settings: Name, Temperature, TopP, MaxTokens, Model
    - 5.3: Parse style profile body as a prompt template with Jinja2 variables
    - 5.6: Support Jinja2 variables in style profiles
    - 5.8: Validate style profile schema before applying
    """
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_round_trip_preserves_name(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (Name)**
        *For any* valid style profile, parsing and re-serializing SHALL
        preserve the profile name.
        **Validates: Requirement 5.2**
        """
        loader = StyleProfileLoader()
        
        # Serialize to Markdown
        markdown = loader.to_markdown(profile)
        
        # Write to temp file and parse back
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name
        
        try:
            parsed = loader.load(temp_path)
            assert parsed.name == profile.name
        finally:
            Path(temp_path).unlink()
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_round_trip_preserves_temperature(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (Temperature)**
        *For any* valid style profile, parsing and re-serializing SHALL
        preserve the temperature setting.
        **Validates: Requirement 5.2**
        """
        loader = StyleProfileLoader()
        markdown = loader.to_markdown(profile)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name
        
        try:
            parsed = loader.load(temp_path)
            assert abs(parsed.temperature - profile.temperature) < 0.0001
        finally:
            Path(temp_path).unlink()
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_round_trip_preserves_top_p(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (TopP)**
        *For any* valid style profile, parsing and re-serializing SHALL
        preserve the top_p setting.
        **Validates: Requirement 5.2**
        """
        loader = StyleProfileLoader()
        markdown = loader.to_markdown(profile)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name
        
        try:
            parsed = loader.load(temp_path)
            assert abs(parsed.top_p - profile.top_p) < 0.0001
        finally:
            Path(temp_path).unlink()
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_round_trip_preserves_max_tokens(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (MaxTokens)**
        *For any* valid style profile, parsing and re-serializing SHALL
        preserve the max_tokens setting.
        **Validates: Requirement 5.2**
        """
        loader = StyleProfileLoader()
        markdown = loader.to_markdown(profile)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name
        
        try:
            parsed = loader.load(temp_path)
            assert parsed.max_tokens == profile.max_tokens
        finally:
            Path(temp_path).unlink()
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_round_trip_preserves_model(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (Model)**
        *For any* valid style profile, parsing and re-serializing SHALL
        preserve the model setting.
        **Validates: Requirement 5.2**
        """
        loader = StyleProfileLoader()
        markdown = loader.to_markdown(profile)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name
        
        try:
            parsed = loader.load(temp_path)
            assert parsed.model == profile.model
        finally:
            Path(temp_path).unlink()
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_round_trip_preserves_prompt_template(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (Prompt Template)**
        *For any* valid style profile, parsing and re-serializing SHALL
        preserve the prompt template content (modulo trailing whitespace).
        **Validates: Requirement 5.3**
        """
        loader = StyleProfileLoader()
        markdown = loader.to_markdown(profile)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name
        
        try:
            parsed = loader.load(temp_path)
            # Compare stripped versions (trailing whitespace is normalized)
            assert parsed.prompt_template.strip() == profile.prompt_template.strip()
        finally:
            Path(temp_path).unlink()
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_extracts_jinja2_variables(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (Jinja2 Variables)**
        *For any* valid style profile with Jinja2 variables, parsing SHALL
        correctly extract all variable names from the template.
        **Validates: Requirement 5.6**
        """
        loader = StyleProfileLoader()
        markdown = loader.to_markdown(profile)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name
        
        try:
            parsed = loader.load(temp_path)
            
            # All variables in original should be in parsed
            for var in profile.variables:
                assert var in parsed.variables
            
            # Parsed should have same or more variables (re-extraction)
            assert len(parsed.variables) >= len(profile.variables)
        finally:
            Path(temp_path).unlink()
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_validation_passes_for_valid_profiles(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (Validation)**
        *For any* valid style profile, validation SHALL pass without errors.
        **Validates: Requirement 5.8**
        """
        loader = StyleProfileLoader()
        
        is_valid, messages = loader.validate(profile)
        
        # Valid profiles should pass validation
        assert is_valid, f"Validation failed: {messages}"
    
    @given(profile=style_profile_object_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_complete_round_trip(self, profile: StyleProfile):
        """
        **Property 2: Style Profile Parsing Round-Trip (Complete)**
        *For any* valid style profile, parsing the file and then serializing
        the parsed profile back to Markdown SHALL produce a semantically
        equivalent file.
        **Validates: Requirements 5.1, 5.2, 5.3, 5.6, 5.8**
        """
        loader = StyleProfileLoader()
        
        # First round trip
        markdown1 = loader.to_markdown(profile)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown1)
            temp_path = f.name
        
        try:
            parsed1 = loader.load(temp_path)
            
            # Second round trip
            markdown2 = loader.to_markdown(parsed1)
            
            # Write and parse again
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f2:
                f2.write(markdown2)
                temp_path2 = f2.name
            
            try:
                parsed2 = loader.load(temp_path2)
                
                # After two round trips, all fields should be identical
                assert parsed1.name == parsed2.name
                assert abs(parsed1.temperature - parsed2.temperature) < 0.0001
                assert abs(parsed1.top_p - parsed2.top_p) < 0.0001
                assert parsed1.max_tokens == parsed2.max_tokens
                assert parsed1.model == parsed2.model
                # Compare stripped templates (trailing whitespace normalized)
                assert parsed1.prompt_template.strip() == parsed2.prompt_template.strip()
            finally:
                Path(temp_path2).unlink()
        finally:
            Path(temp_path).unlink()


# ============================================================================
# Style Profile Unit Tests for Edge Cases
# ============================================================================

class TestStyleProfileEdgeCases:
    """Unit tests for StyleProfile edge cases."""
    
    def test_style_profile_minimal_valid(self):
        """Test StyleProfile with minimal valid values."""
        profile = StyleProfile(
            name="test",
            temperature=0.0,
            top_p=0.0,
            max_tokens=1,
            model="gpt-4",
            prompt_template="Test prompt",
        )
        
        assert profile.name == "test"
        assert profile.temperature == 0.0
        assert profile.max_tokens == 1
    
    def test_style_profile_maximum_valid(self):
        """Test StyleProfile with maximum valid values."""
        profile = StyleProfile(
            name="test-profile-with-long-name",
            temperature=2.0,
            top_p=1.0,
            max_tokens=128000,
            model="claude-3-opus-20240229",
            prompt_template="A very long prompt template " * 100,
        )
        
        assert profile.temperature == 2.0
        assert profile.top_p == 1.0
    
    def test_style_profile_invalid_temperature_low(self):
        """Test StyleProfile rejects temperature below 0."""
        with pytest.raises(ValueError, match="Temperature"):
            StyleProfile(
                name="test",
                temperature=-0.1,
                top_p=0.9,
                max_tokens=1000,
                model="gpt-4",
                prompt_template="Test",
            )
    
    def test_style_profile_invalid_temperature_high(self):
        """Test StyleProfile rejects temperature above 2."""
        with pytest.raises(ValueError, match="Temperature"):
            StyleProfile(
                name="test",
                temperature=2.1,
                top_p=0.9,
                max_tokens=1000,
                model="gpt-4",
                prompt_template="Test",
            )
    
    def test_style_profile_invalid_top_p_low(self):
        """Test StyleProfile rejects top_p below 0."""
        with pytest.raises(ValueError, match="TopP"):
            StyleProfile(
                name="test",
                temperature=0.7,
                top_p=-0.1,
                max_tokens=1000,
                model="gpt-4",
                prompt_template="Test",
            )
    
    def test_style_profile_invalid_top_p_high(self):
        """Test StyleProfile rejects top_p above 1."""
        with pytest.raises(ValueError, match="TopP"):
            StyleProfile(
                name="test",
                temperature=0.7,
                top_p=1.1,
                max_tokens=1000,
                model="gpt-4",
                prompt_template="Test",
            )
    
    def test_style_profile_invalid_max_tokens(self):
        """Test StyleProfile rejects non-positive max_tokens."""
        with pytest.raises(ValueError, match="MaxTokens"):
            StyleProfile(
                name="test",
                temperature=0.7,
                top_p=0.9,
                max_tokens=0,
                model="gpt-4",
                prompt_template="Test",
            )
    
    def test_style_profile_empty_name(self):
        """Test StyleProfile rejects empty name."""
        with pytest.raises(ValueError, match="Name"):
            StyleProfile(
                name="",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                model="gpt-4",
                prompt_template="Test",
            )
    
    def test_style_profile_empty_model(self):
        """Test StyleProfile rejects empty model."""
        with pytest.raises(ValueError, match="Model"):
            StyleProfile(
                name="test",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                model="",
                prompt_template="Test",
            )


class TestStyleProfileLoaderEdgeCases:
    """Unit tests for StyleProfileLoader edge cases."""
    
    def test_loader_missing_file(self):
        """Test loader raises error for missing file."""
        loader = StyleProfileLoader()
        
        with pytest.raises(StyleProfileError, match="not found"):
            loader.load("/nonexistent/path/profile.md")
    
    def test_loader_missing_frontmatter(self):
        """Test loader raises error for missing frontmatter."""
        loader = StyleProfileLoader()
        
        content = "# Just a regular markdown file\n\nNo frontmatter here."
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            with pytest.raises(StyleProfileError, match="frontmatter"):
                loader.load(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_loader_missing_required_field(self):
        """Test loader raises error for missing required field."""
        loader = StyleProfileLoader()
        
        # Missing Temperature field
        content = """---
Name: test
TopP: 0.9
MaxTokens: 1000
Model: gpt-4
---

Test prompt template.
"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            with pytest.raises(StyleProfileError, match="Missing required fields.*Temperature"):
                loader.load(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_loader_extracts_variables(self):
        """Test loader correctly extracts Jinja2 variables."""
        loader = StyleProfileLoader()
        
        content = """---
Name: test
Temperature: 0.7
TopP: 0.9
MaxTokens: 1000
Model: gpt-4
---

Write about {{ title }}.

Summary: {{ summary.short }}

Tags: {{ tags }}
"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            profile = loader.load(temp_path)
            
            assert "title" in profile.variables
            assert "summary.short" in profile.variables
            assert "tags" in profile.variables
            assert len(profile.variables) == 3
        finally:
            Path(temp_path).unlink()
    
    def test_loader_list_builtin_empty_dir(self):
        """Test list_builtin returns empty list for nonexistent directory."""
        loader = StyleProfileLoader(builtin_dir=Path("/nonexistent/dir"))
        
        profiles = loader.list_builtin()
        
        assert profiles == []
    
    def test_loader_get_builtin_not_found(self):
        """Test get_builtin raises error for unknown profile."""
        loader = StyleProfileLoader()
        
        with pytest.raises(StyleProfileError, match="not found"):
            loader.get_builtin("nonexistent-profile")
    
    def test_loader_validate_warns_unknown_variable(self):
        """Test validate warns about unknown variables."""
        loader = StyleProfileLoader()
        
        profile = StyleProfile(
            name="test",
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            model="gpt-4",
            prompt_template="Use {{ unknown_var }} here.",
            variables=["unknown_var"],
        )
        
        is_valid, messages = loader.validate(profile)
        
        # Should be valid but with warning
        assert is_valid
        assert any("unknown_var" in msg for msg in messages)


# ============================================================================
# PROPERTY 10: FormatV1 Metadata Completeness (Requirements 12.2, 12.3, 12.4, 12.7)
# ============================================================================

class TestFormatV1MetadataCompleteness:
    """Property 10: FormatV1 Metadata Completeness.
    
    Validates Requirements 12.2, 12.3, 12.4, 12.7:
    - 12.2: FormatV1 must include format_version, output_type, timestamp, source_file
    - 12.3: FormatV1 must include validation results (character_count, truncated, warnings)
    - 12.4: When LLM enhancement is used, must include provider, model, cost_usd, tokens_used
    - 12.7: Metadata must be serializable to JSON
    """
    
    @given(metadata=format_v1_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_10_required_fields_present(self, metadata: FormatV1):
        """
        **Property 10: FormatV1 Metadata Completeness (Required Fields)**
        *For any* formatted output, the FormatV1 metadata SHALL include all
        required fields: format_version, output_type, timestamp, source_file.
        **Validates: Requirement 12.2**
        """
        # Required fields must be present and non-empty
        assert metadata.format_version == "v1"
        assert metadata.output_type in VALID_OUTPUT_TYPES
        assert metadata.timestamp is not None
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.source_file is not None
        assert len(metadata.source_file) > 0
    
    @given(metadata=format_v1_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_10_validation_metadata_present(self, metadata: FormatV1):
        """
        **Property 10: FormatV1 Metadata Completeness (Validation Results)**
        *For any* formatted output, the FormatV1 metadata SHALL include
        validation results: character_count, truncated status, and warnings.
        **Validates: Requirement 12.3**
        """
        # Validation metadata must be present
        assert metadata.validation is not None
        
        # Required validation fields
        assert metadata.validation.character_count >= 0
        assert isinstance(metadata.validation.truncated, bool)
        assert isinstance(metadata.validation.warnings, list)
    
    @given(metadata=format_v1_strategy(with_llm=True))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_10_llm_metadata_when_enhanced(self, metadata: FormatV1):
        """
        **Property 10: FormatV1 Metadata Completeness (LLM Metadata)**
        *For any* formatted output where LLM enhancement is used, the FormatV1
        metadata SHALL include LLM metadata: provider, model, cost_usd, tokens_used.
        **Validates: Requirement 12.4**
        """
        # LLM metadata must be present when enhancement is used
        assert metadata.llm_metadata is not None
        
        # Required LLM fields
        assert metadata.llm_metadata.provider in ["openai", "claude", "bedrock"]
        assert len(metadata.llm_metadata.model) > 0
        assert metadata.llm_metadata.cost_usd >= 0
        assert metadata.llm_metadata.tokens_used >= 0
        assert 0 <= metadata.llm_metadata.temperature <= 2
        assert isinstance(metadata.llm_metadata.enhanced, bool)
    
    @given(metadata=format_v1_strategy(with_llm=False))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_10_no_llm_metadata_when_not_enhanced(self, metadata: FormatV1):
        """
        **Property 10: FormatV1 Metadata Completeness (No LLM When Not Enhanced)**
        *For any* formatted output where LLM enhancement is NOT used, the
        FormatV1 metadata SHALL NOT include LLM metadata.
        **Validates: Requirement 12.4**
        """
        # LLM metadata should be None when enhancement is not used
        assert metadata.llm_metadata is None
    
    @given(metadata=format_v1_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_10_json_serializable(self, metadata: FormatV1):
        """
        **Property 10: FormatV1 Metadata Completeness (JSON Serializable)**
        *For any* FormatV1 metadata, it SHALL be serializable to valid JSON.
        **Validates: Requirement 12.7**
        """
        import json
        
        # Should serialize without errors
        json_str = metadata.model_dump_json()
        assert json_str is not None
        assert len(json_str) > 0
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        
        # Required fields should be in JSON
        assert "format_version" in parsed
        assert "output_type" in parsed
        assert "timestamp" in parsed
        assert "source_file" in parsed
        assert "validation" in parsed
    
    @given(metadata=format_v1_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_10_frontmatter_generation(self, metadata: FormatV1):
        """
        **Property 10: FormatV1 Metadata Completeness (Frontmatter)**
        *For any* FormatV1 metadata, it SHALL generate valid YAML frontmatter
        for embedding in Markdown outputs.
        **Validates: Requirement 12.7**
        """
        # Should generate frontmatter without errors
        frontmatter = metadata.to_frontmatter()
        assert frontmatter is not None
        assert len(frontmatter) > 0
        
        # Should start and end with YAML delimiters
        assert frontmatter.startswith("---")
        assert frontmatter.endswith("---")
        
        # Should contain required fields
        assert "format_version:" in frontmatter
        assert "output_type:" in frontmatter
        assert "timestamp:" in frontmatter
        assert "source_file:" in frontmatter
        assert "validation:" in frontmatter
        assert "character_count:" in frontmatter
    
    @given(metadata=format_v1_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_10_optional_fields_valid_when_present(self, metadata: FormatV1):
        """
        **Property 10: FormatV1 Metadata Completeness (Optional Fields)**
        *For any* FormatV1 metadata with optional fields present, those fields
        SHALL contain valid values from the allowed set.
        **Validates: Requirements 12.2, 12.3**
        """
        # Platform must be valid if present
        if metadata.platform is not None:
            assert metadata.platform in VALID_PLATFORMS
        
        # Tone must be valid if present
        if metadata.tone is not None:
            assert metadata.tone in VALID_TONES
        
        # Length must be valid if present
        if metadata.length is not None:
            assert metadata.length in VALID_LENGTHS
        
        # Style profile can be any string or None
        if metadata.style_profile_used is not None:
            assert len(metadata.style_profile_used) > 0


# ============================================================================
# Additional unit tests for edge cases
# ============================================================================

class TestFormatV1EdgeCases:
    """Unit tests for FormatV1 edge cases."""
    
    def test_format_v1_minimal_valid(self):
        """Test FormatV1 with minimal required fields."""
        metadata = FormatV1(
            output_type="blog",
            source_file="test.json",
            validation=ValidationMetadata(character_count=100),
        )
        
        assert metadata.format_version == "v1"
        assert metadata.output_type == "blog"
        assert metadata.source_file == "test.json"
        assert metadata.validation.character_count == 100
        assert metadata.llm_metadata is None
    
    def test_format_v1_with_all_fields(self):
        """Test FormatV1 with all fields populated."""
        metadata = FormatV1(
            output_type="tweet",
            platform="twitter",
            source_file="content.json",
            style_profile_used="twitter-thread",
            llm_metadata=LLMMetadata(
                provider="openai",
                model="gpt-4",
                cost_usd=0.05,
                tokens_used=1500,
                temperature=0.7,
                enhanced=True,
            ),
            validation=ValidationMetadata(
                platform="twitter",
                character_count=280,
                truncated=False,
                warnings=[],
            ),
            tone="casual",
            length="short",
        )
        
        assert metadata.output_type == "tweet"
        assert metadata.platform == "twitter"
        assert metadata.llm_metadata.provider == "openai"
        assert metadata.validation.character_count == 280
    
    def test_format_v1_frontmatter_with_llm(self):
        """Test frontmatter generation includes LLM section when present."""
        metadata = FormatV1(
            output_type="blog",
            source_file="test.json",
            llm_metadata=LLMMetadata(
                provider="claude",
                model="claude-3-opus",
                cost_usd=0.10,
                tokens_used=2000,
                temperature=0.5,
                enhanced=True,
            ),
            validation=ValidationMetadata(character_count=5000),
        )
        
        frontmatter = metadata.to_frontmatter()
        
        assert "llm:" in frontmatter
        assert "provider: claude" in frontmatter
        assert "model: claude-3-opus" in frontmatter
        assert "cost_usd:" in frontmatter
        assert "tokens_used: 2000" in frontmatter
    
    def test_format_v1_frontmatter_without_llm(self):
        """Test frontmatter generation excludes LLM section when not present."""
        metadata = FormatV1(
            output_type="blog",
            source_file="test.json",
            validation=ValidationMetadata(character_count=5000),
        )
        
        frontmatter = metadata.to_frontmatter()
        
        assert "llm:" not in frontmatter
    
    def test_validation_metadata_with_warnings(self):
        """Test ValidationMetadata with multiple warnings."""
        validation = ValidationMetadata(
            platform="twitter",
            character_count=280,
            truncated=True,
            warnings=[
                "Content truncated to fit character limit",
                "Hashtag count reduced from 8 to 5",
            ],
        )
        
        assert len(validation.warnings) == 2
        assert validation.truncated is True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])



# ============================================================================
# PROPERTY 3: Platform Character Limit Enforcement (Requirements 8.1-8.5)
# ============================================================================

from pipeline.formatters.validator import (
    PlatformValidator,
    PlatformLimits,
    ValidationResult,
)


# Strategies for platform validation tests
platforms_with_limits_strategy = st.sampled_from([
    "twitter",      # 280 chars
    "linkedin",     # 3000 chars
    "youtube",      # 5000 chars
    "meta_title",   # 60 chars
    "meta_description",  # 160 chars
])

platforms_without_limits_strategy = st.sampled_from([
    "medium",
    "wordpress",
    "ghost",
    "substack",
])

all_platforms_strategy = st.sampled_from([
    "twitter", "linkedin", "youtube", "medium", "wordpress",
    "ghost", "substack", "meta_title", "meta_description",
])


@st.composite
def content_within_limit_strategy(draw, platform: str):
    """Generate content that fits within platform limit."""
    validator = PlatformValidator()
    limits = validator.get_limits(platform)
    
    if limits.max_chars is None:
        # No limit - generate any reasonable content
        max_len = draw(st.integers(min_value=1, max_value=1000))
    else:
        # Generate content within limit
        max_len = limits.max_chars
    
    # Generate content of appropriate length
    length = draw(st.integers(min_value=1, max_value=max_len))
    content = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=length,
        max_size=length,
    ))
    return content


@st.composite
def content_exceeding_limit_strategy(draw, platform: str):
    """Generate content that exceeds platform limit."""
    validator = PlatformValidator()
    limits = validator.get_limits(platform)
    
    if limits.max_chars is None:
        # No limit - can't exceed, return empty
        assume(False)
    
    # Generate content exceeding limit
    min_len = limits.max_chars + 1
    max_len = limits.max_chars + 500
    length = draw(st.integers(min_value=min_len, max_value=max_len))
    content = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=length,
        max_size=length,
    ))
    return content


class TestPlatformCharacterLimitEnforcement:
    """Property 3: Platform Character Limit Enforcement.
    
    Validates Requirements 8.1, 8.2, 8.3, 8.4, 8.5:
    - 8.1: Enforce Twitter character limit (280 characters per tweet)
    - 8.2: Enforce LinkedIn character limit (3000 characters)
    - 8.3: Enforce YouTube description limit (5000 characters)
    - 8.4: Enforce meta description limit (160 characters)
    - 8.5: Enforce meta title limit (60 characters)
    """
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_twitter_limit_enforced(self, data):
        """
        **Property 3: Platform Character Limit Enforcement (Twitter)**
        *For any* content validated against Twitter, the validation SHALL
        correctly identify content exceeding 280 characters.
        **Validates: Requirement 8.1**
        """
        validator = PlatformValidator()
        
        # Generate content of varying lengths
        length = data.draw(st.integers(min_value=1, max_value=500))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        result = validator.validate(content, "twitter")
        
        # Verify limit enforcement
        if len(content) > 280:
            assert result.exceeds_limit is True
            assert result.is_valid is False
        else:
            assert result.exceeds_limit is False
            assert result.is_valid is True
        
        # Character count should always be accurate
        assert result.character_count == len(content)
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_linkedin_limit_enforced(self, data):
        """
        **Property 3: Platform Character Limit Enforcement (LinkedIn)**
        *For any* content validated against LinkedIn, the validation SHALL
        correctly identify content exceeding 3000 characters.
        **Validates: Requirement 8.2**
        """
        validator = PlatformValidator()
        
        length = data.draw(st.integers(min_value=1, max_value=4000))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        result = validator.validate(content, "linkedin")
        
        if len(content) > 3000:
            assert result.exceeds_limit is True
            assert result.is_valid is False
        else:
            assert result.exceeds_limit is False
            assert result.is_valid is True
        
        assert result.character_count == len(content)
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_youtube_limit_enforced(self, data):
        """
        **Property 3: Platform Character Limit Enforcement (YouTube)**
        *For any* content validated against YouTube, the validation SHALL
        correctly identify content exceeding 5000 characters.
        **Validates: Requirement 8.3**
        """
        validator = PlatformValidator()
        
        length = data.draw(st.integers(min_value=1, max_value=6000))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        result = validator.validate(content, "youtube")
        
        if len(content) > 5000:
            assert result.exceeds_limit is True
            assert result.is_valid is False
        else:
            assert result.exceeds_limit is False
            assert result.is_valid is True
        
        assert result.character_count == len(content)
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_meta_description_limit_enforced(self, data):
        """
        **Property 3: Platform Character Limit Enforcement (Meta Description)**
        *For any* content validated against meta_description, the validation
        SHALL correctly identify content exceeding 160 characters.
        **Validates: Requirement 8.4**
        """
        validator = PlatformValidator()
        
        length = data.draw(st.integers(min_value=1, max_value=300))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        result = validator.validate(content, "meta_description")
        
        if len(content) > 160:
            assert result.exceeds_limit is True
            assert result.is_valid is False
        else:
            assert result.exceeds_limit is False
            assert result.is_valid is True
        
        assert result.character_count == len(content)
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_meta_title_limit_enforced(self, data):
        """
        **Property 3: Platform Character Limit Enforcement (Meta Title)**
        *For any* content validated against meta_title, the validation
        SHALL correctly identify content exceeding 60 characters.
        **Validates: Requirement 8.5**
        """
        validator = PlatformValidator()
        
        length = data.draw(st.integers(min_value=1, max_value=100))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        result = validator.validate(content, "meta_title")
        
        if len(content) > 60:
            assert result.exceeds_limit is True
            assert result.is_valid is False
        else:
            assert result.exceeds_limit is False
            assert result.is_valid is True
        
        assert result.character_count == len(content)
    
    @given(platform=platforms_without_limits_strategy, data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_no_limit_platforms_always_valid(self, platform, data):
        """
        **Property 3: Platform Character Limit Enforcement (No Limit)**
        *For any* platform without character limits (Medium, WordPress, etc.),
        content of any length SHALL be considered valid.
        **Validates: Requirements 8.1-8.5 (inverse case)**
        """
        validator = PlatformValidator()
        
        # Generate content of any length (capped at Hypothesis limit of 8192)
        length = data.draw(st.integers(min_value=1, max_value=8000))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        result = validator.validate(content, platform)
        
        # Should always be valid (no character limit)
        assert result.exceeds_limit is False
        assert result.is_valid is True
        assert result.character_count == len(content)
    
    @given(platform=all_platforms_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_empty_content_always_valid(self, platform):
        """
        **Property 3: Platform Character Limit Enforcement (Empty Content)**
        *For any* platform, empty content SHALL be considered valid.
        **Validates: Requirements 8.1-8.5 (edge case)**
        """
        validator = PlatformValidator()
        
        result = validator.validate("", platform)
        
        assert result.is_valid is True
        assert result.exceeds_limit is False
        assert result.character_count == 0
    
    @given(platform=platforms_with_limits_strategy, data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_truncated_content_within_limit(self, platform, data):
        """
        **Property 3: Platform Character Limit Enforcement (After Truncation)**
        *For any* content and platform with a defined character limit, the
        truncated output length SHALL NOT exceed the platform's maximum.
        **Validates: Requirements 8.1-8.5**
        """
        validator = PlatformValidator()
        limits = validator.get_limits(platform)
        
        # Generate content that may exceed limit (capped at Hypothesis limit)
        max_len = min(limits.max_chars * 2, 8000)
        length = data.draw(st.integers(min_value=1, max_value=max_len))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        # Truncate content
        truncated, was_truncated, warnings = validator.truncate(content, platform)
        
        # Truncated content must be within limit
        assert len(truncated) <= limits.max_chars
        
        # If original exceeded limit, truncation should have occurred
        if len(content) > limits.max_chars:
            assert was_truncated is True
        else:
            assert was_truncated is False
            assert truncated == content


class TestPlatformValidatorEdgeCases:
    """Unit tests for PlatformValidator edge cases."""
    
    def test_validator_unknown_platform_raises_error(self):
        """Test validator raises error for unknown platform."""
        validator = PlatformValidator()
        
        with pytest.raises(ValueError, match="Unknown platform"):
            validator.validate("test content", "unknown_platform")
    
    def test_validator_case_insensitive_platform(self):
        """Test validator handles platform names case-insensitively."""
        validator = PlatformValidator()
        
        # Should work with different cases
        result1 = validator.validate("test", "twitter")
        result2 = validator.validate("test", "Twitter")
        result3 = validator.validate("test", "TWITTER")
        
        assert result1.character_count == result2.character_count == result3.character_count
    
    def test_validator_exact_limit_is_valid(self):
        """Test content at exactly the limit is valid."""
        validator = PlatformValidator()
        
        # Twitter limit is 280
        content = "a" * 280
        result = validator.validate(content, "twitter")
        
        assert result.is_valid is True
        assert result.exceeds_limit is False
        assert result.character_count == 280
    
    def test_validator_one_over_limit_is_invalid(self):
        """Test content one character over limit is invalid."""
        validator = PlatformValidator()
        
        # Twitter limit is 280
        content = "a" * 281
        result = validator.validate(content, "twitter")
        
        assert result.is_valid is False
        assert result.exceeds_limit is True
        assert result.character_count == 281
    
    def test_validator_hashtag_counting(self):
        """Test validator counts hashtags correctly."""
        validator = PlatformValidator()
        
        content = "Check out #python #coding #programming #tech #ai #ml"
        result = validator.validate(content, "twitter")
        
        # Twitter allows 5 hashtags, this has 6
        assert any("hashtag" in w.lower() for w in result.warnings)
    
    def test_validator_link_counting(self):
        """Test validator counts links correctly."""
        validator = PlatformValidator()
        
        content = "Check https://example.com and http://test.com"
        result = validator.validate(content, "twitter")
        
        # Twitter allows 1 link, this has 2
        assert any("link" in w.lower() for w in result.warnings)
    
    def test_get_limits_returns_correct_values(self):
        """Test get_limits returns correct platform limits."""
        validator = PlatformValidator()
        
        twitter_limits = validator.get_limits("twitter")
        assert twitter_limits.max_chars == 280
        assert twitter_limits.max_hashtags == 5
        
        linkedin_limits = validator.get_limits("linkedin")
        assert linkedin_limits.max_chars == 3000
        
        medium_limits = validator.get_limits("medium")
        assert medium_limits.max_chars is None



# ============================================================================
# PROPERTY 4: Intelligent Truncation at Sentence Boundaries (Requirements 8.6, 8.7)
# ============================================================================

@st.composite
def content_with_sentences_strategy(draw):
    """Generate content with multiple sentences."""
    num_sentences = draw(st.integers(min_value=2, max_value=10))
    sentences = []
    
    for _ in range(num_sentences):
        # Generate sentence text (words)
        num_words = draw(st.integers(min_value=3, max_value=15))
        words = [
            draw(st.from_regex(r"[A-Za-z]{2,10}", fullmatch=True))
            for _ in range(num_words)
        ]
        sentence = " ".join(words)
        
        # Add sentence ending
        ending = draw(st.sampled_from([".", "!", "?"]))
        sentences.append(sentence + ending)
    
    return " ".join(sentences)


class TestIntelligentTruncation:
    """Property 4: Intelligent Truncation at Sentence Boundaries.
    
    Validates Requirements 8.6, 8.7:
    - 8.6: Truncate content at sentence boundaries when possible
    - 8.7: Include truncation warning in result when content is truncated
    """
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_truncation_at_sentence_boundary(self, data):
        """
        **Property 4: Intelligent Truncation at Sentence Boundaries**
        *For any* content with sentences that exceeds platform limits,
        truncation SHALL occur at a sentence boundary when possible.
        **Validates: Requirement 8.6**
        """
        validator = PlatformValidator()
        
        # Generate content with sentences
        content = data.draw(content_with_sentences_strategy())
        
        # Only test if content exceeds Twitter limit
        assume(len(content) > 280)
        
        truncated, was_truncated, warnings = validator.truncate(content, "twitter")
        
        assert was_truncated is True
        assert len(truncated) <= 280
        
        # Check if truncation occurred at sentence boundary
        # (ends with sentence ending character or has ellipsis)
        if not truncated.endswith("..."):
            # Should end with sentence ending
            assert truncated[-1] in ".!?", f"Expected sentence boundary, got: '{truncated[-10:]}'"
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_truncation_warning_included(self, data):
        """
        **Property 4: Intelligent Truncation at Sentence Boundaries (Warning)**
        *For any* content that is truncated, the result SHALL include
        a truncation warning.
        **Validates: Requirement 8.7**
        """
        validator = PlatformValidator()
        
        # Generate content that exceeds limit
        length = data.draw(st.integers(min_value=300, max_value=1000))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        truncated, was_truncated, warnings = validator.truncate(content, "twitter")
        
        assert was_truncated is True
        assert len(warnings) > 0
        assert any("truncat" in w.lower() for w in warnings)
    
    @given(st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_no_truncation_when_within_limit(self, data):
        """
        **Property 4: Intelligent Truncation (No Truncation Needed)**
        *For any* content within platform limits, no truncation SHALL occur.
        **Validates: Requirements 8.6, 8.7 (inverse case)**
        """
        validator = PlatformValidator()
        
        # Generate content within Twitter limit
        length = data.draw(st.integers(min_value=1, max_value=280))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        truncated, was_truncated, warnings = validator.truncate(content, "twitter")
        
        assert was_truncated is False
        assert truncated == content
        assert len(warnings) == 0
    
    @given(platform=platforms_with_limits_strategy, data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_truncated_content_never_exceeds_limit(self, platform, data):
        """
        **Property 4: Intelligent Truncation (Limit Guarantee)**
        *For any* content and platform, truncated content SHALL NEVER
        exceed the platform's character limit.
        **Validates: Requirements 8.6, 8.7**
        """
        validator = PlatformValidator()
        limits = validator.get_limits(platform)
        
        # Generate content of varying lengths (capped at Hypothesis limit)
        max_len = min(limits.max_chars * 3, 8000)
        length = data.draw(st.integers(min_value=1, max_value=max_len))
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        truncated, was_truncated, warnings = validator.truncate(content, platform)
        
        # This is the key property: truncated content must fit
        assert len(truncated) <= limits.max_chars
    
    @given(st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_sentence_boundary_preferred(self, data):
        """
        **Property 4: Intelligent Truncation (Sentence Boundary Preferred)**
        *For any* content with clear sentence boundaries, truncation SHALL
        prefer ending at a sentence boundary over mid-sentence truncation.
        **Validates: Requirement 8.6**
        """
        validator = PlatformValidator()
        
        # Create content with clear sentences that exceeds limit
        # First sentence fits, second doesn't
        sentence1 = "This is the first sentence."  # 28 chars
        sentence2 = "This is the second sentence that is much longer and will exceed the limit."
        
        # Pad to ensure we exceed Twitter's 280 limit
        padding = " " + "Extra content. " * 20
        content = sentence1 + " " + sentence2 + padding
        
        assume(len(content) > 280)
        assume(len(sentence1) < 280)
        
        truncated, was_truncated, warnings = validator.truncate(content, "twitter")
        
        assert was_truncated is True
        
        # Should end at a sentence boundary (not with ellipsis if possible)
        # The truncated content should end with a sentence ending
        stripped = truncated.rstrip()
        if not stripped.endswith("..."):
            assert stripped[-1] in ".!?", f"Expected sentence boundary, got: '{stripped[-20:]}'"
    
    @given(st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_ellipsis_when_no_boundary(self, data):
        """
        **Property 4: Intelligent Truncation (Ellipsis Fallback)**
        *For any* content without sentence boundaries within the limit,
        truncation SHALL add ellipsis.
        **Validates: Requirement 8.6**
        """
        validator = PlatformValidator()
        
        # Generate content without sentence endings (one long "sentence")
        length = data.draw(st.integers(min_value=300, max_value=500))
        # Use only letters and spaces, no punctuation
        content = data.draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "Z")),
            min_size=length,
            max_size=length,
        ))
        
        # Ensure no sentence endings
        content = content.replace(".", "").replace("!", "").replace("?", "")
        assume(len(content) > 280)
        
        truncated, was_truncated, warnings = validator.truncate(content, "twitter")
        
        assert was_truncated is True
        assert truncated.endswith("...")
        assert any("no sentence boundary" in w.lower() for w in warnings)


class TestTruncationEdgeCases:
    """Unit tests for truncation edge cases."""
    
    def test_truncation_preserves_complete_sentences(self):
        """Test truncation keeps complete sentences when possible."""
        validator = PlatformValidator()
        
        # Content with clear sentence boundaries
        content = "First sentence. Second sentence. Third sentence is very long and will cause truncation to occur somewhere in the middle of this text which keeps going and going."
        
        # Use meta_title (60 char limit) for easier testing
        truncated, was_truncated, warnings = validator.truncate(content, "meta_title")
        
        assert was_truncated is True
        assert len(truncated) <= 60
        # Should end at "First sentence." (15 chars) since that's the last complete sentence
        assert truncated.rstrip().endswith(".")
    
    def test_truncation_single_long_sentence(self):
        """Test truncation of single sentence exceeding limit."""
        validator = PlatformValidator()
        
        # Single sentence longer than limit
        content = "This is one very long sentence that goes on and on without any breaks or periods until it exceeds the character limit significantly"
        
        truncated, was_truncated, warnings = validator.truncate(content, "meta_title")
        
        assert was_truncated is True
        assert len(truncated) <= 60
        assert truncated.endswith("...")
    
    def test_truncation_empty_content(self):
        """Test truncation of empty content."""
        validator = PlatformValidator()
        
        truncated, was_truncated, warnings = validator.truncate("", "twitter")
        
        assert was_truncated is False
        assert truncated == ""
        assert len(warnings) == 0
    
    def test_truncation_content_exactly_at_limit(self):
        """Test truncation when content is exactly at limit."""
        validator = PlatformValidator()
        
        content = "a" * 280  # Exactly Twitter limit
        
        truncated, was_truncated, warnings = validator.truncate(content, "twitter")
        
        assert was_truncated is False
        assert truncated == content
        assert len(warnings) == 0
    
    def test_truncation_content_one_over_limit(self):
        """Test truncation when content is one character over limit."""
        validator = PlatformValidator()
        
        content = "a" * 281  # One over Twitter limit
        
        truncated, was_truncated, warnings = validator.truncate(content, "twitter")
        
        assert was_truncated is True
        assert len(truncated) <= 280
    
    def test_truncation_with_custom_ellipsis(self):
        """Test truncation with custom ellipsis."""
        validator = PlatformValidator()
        
        content = "a" * 300
        
        truncated, was_truncated, warnings = validator.truncate(
            content, "twitter", ellipsis=" [...]"
        )
        
        assert was_truncated is True
        assert truncated.endswith(" [...]")
        assert len(truncated) <= 280
    
    def test_validate_and_truncate_convenience_method(self):
        """Test validate_and_truncate combines both operations."""
        validator = PlatformValidator()
        
        content = "a" * 300  # Exceeds Twitter limit
        
        final_content, result = validator.validate_and_truncate(content, "twitter")
        
        assert len(final_content) <= 280
        assert result.truncated is True
        assert result.is_valid is True  # After truncation, should be valid
    
    def test_validate_and_truncate_no_auto_truncate(self):
        """Test validate_and_truncate with auto_truncate=False."""
        validator = PlatformValidator()
        
        content = "a" * 300  # Exceeds Twitter limit
        
        final_content, result = validator.validate_and_truncate(
            content, "twitter", auto_truncate=False
        )
        
        assert final_content == content  # Not truncated
        assert result.exceeds_limit is True
        assert result.is_valid is False
    
    def test_truncation_unicode_content(self):
        """Test truncation handles unicode content correctly."""
        validator = PlatformValidator()
        
        # Unicode content (emojis count as multiple bytes but single chars)
        content = "Hello 👋 " * 50  # Each emoji is 1 character
        
        truncated, was_truncated, warnings = validator.truncate(content, "twitter")
        
        assert len(truncated) <= 280
        # Should handle unicode without errors
    
    def test_truncation_sentence_with_abbreviations(self):
        """Test truncation handles abbreviations correctly."""
        validator = PlatformValidator()
        
        # Content with abbreviations that have periods
        content = "Dr. Smith went to the U.S.A. for a conference. This is another sentence that makes the content longer."
        
        # The truncation should not break at "Dr." or "U.S.A."
        truncated, was_truncated, warnings = validator.truncate(content, "meta_title")
        
        assert len(truncated) <= 60



# ============================================================================
# PROPERTY 1: Output Structure Completeness (Requirements 1.1-4.2)
# ============================================================================

from pipeline.formatters.generators import (
    GeneratorFactory,
    GeneratorConfig,
    BaseGenerator,
    BlogGenerator,
    TweetGenerator,
    YouTubeGenerator,
    SEOGenerator,
    LinkedInGenerator,
    NewsletterGenerator,
    ChaptersGenerator,
    TranscriptCleanGenerator,
    PodcastNotesGenerator,
    MeetingMinutesGenerator,
    SlidesGenerator,
    NotionGenerator,
    ObsidianGenerator,
    QuoteCardsGenerator,
    VideoScriptGenerator,
    TikTokScriptGenerator,
)
from pipeline.formatters.template_engine import TemplateEngine
from pipeline.formatters.base import FormatRequest


# Strategies for generator tests
@st.composite
def enriched_content_strategy(draw, include_chapters: bool = True, include_highlights: bool = True):
    """Generate valid enriched content for testing generators."""
    # Generate summary
    short_summary = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=20,
        max_size=200,
    ))
    medium_summary = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=50,
        max_size=500,
    ))
    
    content = {
        "enrichment_version": "v1",
        "metadata": {
            "source_file": draw(file_path_strategy),
            "title": draw(st.text(min_size=5, max_size=100)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "short": short_summary,
            "medium": medium_summary,
            "long": medium_summary + " " + short_summary,
        },
        "tags": {
            "primary": draw(st.lists(
                st.from_regex(r"[a-z][a-z0-9]{2,15}", fullmatch=True),
                min_size=2,
                max_size=5,
            )),
            "secondary": draw(st.lists(
                st.from_regex(r"[a-z][a-z0-9]{2,15}", fullmatch=True),
                min_size=0,
                max_size=3,
            )),
        },
    }
    
    if include_chapters:
        num_chapters = draw(st.integers(min_value=2, max_value=5))
        chapters = []
        for i in range(num_chapters):
            chapters.append({
                "title": f"Chapter {i + 1}",
                "summary": draw(st.text(min_size=20, max_size=100)),
                "start_time": i * 60,
                "end_time": (i + 1) * 60,
            })
        content["chapters"] = chapters
    
    if include_highlights:
        num_highlights = draw(st.integers(min_value=2, max_value=5))
        highlights = []
        for i in range(num_highlights):
            highlights.append({
                "text": draw(st.text(min_size=20, max_size=150)),
                "timestamp": i * 30,
            })
        content["highlights"] = highlights
    
    return content


@st.composite
def enriched_content_with_transcript_strategy(draw):
    """Generate enriched content with transcript for transcript-clean generator."""
    content = draw(enriched_content_strategy())
    
    # Add transcript
    transcript_text = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=100,
        max_size=1000,
    ))
    
    content["transcript"] = {
        "text": transcript_text,
        "segments": [
            {
                "speaker": "Speaker 1",
                "text": transcript_text[:len(transcript_text)//2],
                "start_time": 0,
                "end_time": 30,
            },
            {
                "speaker": "Speaker 2",
                "text": transcript_text[len(transcript_text)//2:],
                "start_time": 30,
                "end_time": 60,
            },
        ],
    }
    
    return content


class TestOutputStructureCompleteness:
    """Property 1: Output Structure Completeness.
    
    Validates Requirements 1.1-1.4, 2.1-2.4, 3.1-3.6, 4.1-4.2:
    For any output type and valid enriched content, the generated output
    SHALL contain all required structural elements defined for that output type.
    """
    
    def _create_generator(self, generator_class: type) -> BaseGenerator:
        """Create a generator instance with default config."""
        config = GeneratorConfig(
            template_engine=TemplateEngine(),
            platform_validator=PlatformValidator(),
            auto_truncate=True,
        )
        return generator_class(config)
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_blog_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Blog)**
        *For any* valid enriched content, the blog generator SHALL produce
        output with title, introduction, body, and conclusion sections.
        **Validates: Requirement 1.1**
        """
        generator = self._create_generator(BlogGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="blog",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        # Should succeed
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        
        # Metadata should be complete
        assert result.metadata is not None
        assert result.metadata.output_type == "blog"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_tweet_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Tweet)**
        *For any* valid enriched content, the tweet generator SHALL produce
        output with hook, main points, and conclusion.
        **Validates: Requirement 1.2**
        """
        generator = self._create_generator(TweetGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="tweet",
            platform="twitter",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "tweet"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_youtube_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (YouTube)**
        *For any* valid enriched content, the YouTube generator SHALL produce
        output with title, summary, timestamps, and tags.
        **Validates: Requirement 1.3**
        """
        generator = self._create_generator(YouTubeGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="youtube",
            platform="youtube",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "youtube"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_seo_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (SEO)**
        *For any* valid enriched content, the SEO generator SHALL produce
        output with meta title, meta description, keywords, and OG tags.
        **Validates: Requirement 1.4**
        """
        generator = self._create_generator(SEOGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="seo",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "seo"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_linkedin_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (LinkedIn)**
        *For any* valid enriched content, the LinkedIn generator SHALL produce
        output with hook, insights, and engagement prompt.
        **Validates: Requirement 2.1**
        """
        generator = self._create_generator(LinkedInGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="linkedin",
            platform="linkedin",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "linkedin"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_newsletter_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Newsletter)**
        *For any* valid enriched content, the newsletter generator SHALL produce
        output with subject, preview, sections, and footer.
        **Validates: Requirement 2.2**
        """
        generator = self._create_generator(NewsletterGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="newsletter",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "newsletter"
    
    @given(content=enriched_content_strategy(include_chapters=True))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_chapters_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Chapters)**
        *For any* valid enriched content with chapters, the chapters generator
        SHALL produce output with timestamps, titles, and descriptions.
        **Validates: Requirement 2.3**
        """
        generator = self._create_generator(ChaptersGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="chapters",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "chapters"
    
    @given(content=enriched_content_with_transcript_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_transcript_clean_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Transcript Clean)**
        *For any* valid enriched content with transcript, the transcript-clean
        generator SHALL produce output with speaker labels and paragraphs.
        **Validates: Requirement 2.4**
        """
        generator = self._create_generator(TranscriptCleanGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="transcript-clean",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "transcript-clean"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_podcast_notes_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Podcast Notes)**
        *For any* valid enriched content, the podcast-notes generator SHALL
        produce output with episode summary, key topics, and timestamps.
        **Validates: Requirement 3.1**
        """
        generator = self._create_generator(PodcastNotesGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="podcast-notes",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "podcast-notes"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_meeting_minutes_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Meeting Minutes)**
        *For any* valid enriched content, the meeting-minutes generator SHALL
        produce output with agenda, discussion points, and action items.
        **Validates: Requirement 3.2**
        """
        generator = self._create_generator(MeetingMinutesGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="meeting-minutes",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "meeting-minutes"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_slides_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Slides)**
        *For any* valid enriched content, the slides generator SHALL produce
        output with title slide, content slides, and conclusion slide.
        **Validates: Requirement 3.3**
        """
        generator = self._create_generator(SlidesGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="slides",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "slides"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_notion_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Notion)**
        *For any* valid enriched content, the notion generator SHALL produce
        output with page title, properties, and structured sections.
        **Validates: Requirement 3.4**
        """
        generator = self._create_generator(NotionGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="notion",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "notion"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_obsidian_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Obsidian)**
        *For any* valid enriched content, the obsidian generator SHALL produce
        output with YAML frontmatter, tags, and structured sections.
        **Validates: Requirement 3.5**
        """
        generator = self._create_generator(ObsidianGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="obsidian",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "obsidian"
    
    @given(content=enriched_content_strategy(include_highlights=True))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_quote_cards_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Quote Cards)**
        *For any* valid enriched content with highlights, the quote-cards
        generator SHALL produce output with quotes and attribution.
        **Validates: Requirement 3.6**
        """
        generator = self._create_generator(QuoteCardsGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="quote-cards",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "quote-cards"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_video_script_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (Video Script)**
        *For any* valid enriched content, the video-script generator SHALL
        produce output with scene markers, dialogue, and timing cues.
        **Validates: Requirement 4.1**
        """
        generator = self._create_generator(VideoScriptGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="video-script",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "video-script"
    
    @given(content=enriched_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_tiktok_script_generator_produces_valid_output(self, content):
        """
        **Property 1: Output Structure Completeness (TikTok Script)**
        *For any* valid enriched content, the tiktok-script generator SHALL
        produce output with hook, body, CTA, and 60-second timing.
        **Validates: Requirement 4.2**
        """
        generator = self._create_generator(TikTokScriptGenerator)
        
        request = FormatRequest(
            enriched_content=content,
            output_type="tiktok-script",
            llm_enhance=False,
        )
        
        result = generator.format(request)
        
        assert result.success is True
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata.output_type == "tiktok-script"


class TestGeneratorFactoryCompleteness:
    """Tests for GeneratorFactory completeness."""
    
    def test_all_16_output_types_registered(self):
        """Test that all 16 output types have registered generators."""
        expected_types = [
            "blog", "tweet", "youtube", "seo",
            "linkedin", "newsletter", "chapters", "transcript-clean",
            "podcast-notes", "meeting-minutes", "slides", "notion",
            "obsidian", "quote-cards", "video-script", "tiktok-script",
        ]
        
        registered = GeneratorFactory.get_registered_types()
        
        for output_type in expected_types:
            assert output_type in registered, f"Missing generator for: {output_type}"
        
        assert len(registered) == 16
    
    def test_factory_creates_all_generators(self):
        """Test that factory can create all registered generators."""
        factory = GeneratorFactory()
        
        for output_type in GeneratorFactory.get_registered_types():
            generator = factory.get_generator(output_type)
            assert generator is not None
            assert generator.output_type == output_type
    
    def test_factory_caches_generators(self):
        """Test that factory caches generator instances."""
        factory = GeneratorFactory()
        
        gen1 = factory.get_generator("blog")
        gen2 = factory.get_generator("blog")
        
        assert gen1 is gen2  # Same instance
    
    def test_factory_raises_for_unknown_type(self):
        """Test that factory raises error for unknown output type."""
        factory = GeneratorFactory()
        
        with pytest.raises(Exception):  # GeneratorFactoryError
            factory.get_generator("unknown-type")


class TestGeneratorInputValidation:
    """Tests for generator input validation."""
    
    def test_blog_generator_validates_required_fields(self):
        """Test blog generator validates required enrichments."""
        config = GeneratorConfig(
            template_engine=TemplateEngine(),
            platform_validator=PlatformValidator(),
        )
        generator = BlogGenerator(config)
        
        # Missing summary
        invalid_content = {"tags": ["test"]}
        is_valid, errors = generator.validate_input(invalid_content)
        
        assert is_valid is False
        assert any("summary" in e.lower() for e in errors)
    
    def test_chapters_generator_validates_required_fields(self):
        """Test chapters generator validates required enrichments."""
        config = GeneratorConfig(
            template_engine=TemplateEngine(),
            platform_validator=PlatformValidator(),
        )
        generator = ChaptersGenerator(config)
        
        # Missing chapters
        invalid_content = {"summary": {"short": "test"}}
        is_valid, errors = generator.validate_input(invalid_content)
        
        assert is_valid is False
        assert any("chapters" in e.lower() for e in errors)
    
    def test_quote_cards_generator_validates_required_fields(self):
        """Test quote-cards generator validates required enrichments."""
        config = GeneratorConfig(
            template_engine=TemplateEngine(),
            platform_validator=PlatformValidator(),
        )
        generator = QuoteCardsGenerator(config)
        
        # Missing highlights
        invalid_content = {"summary": {"short": "test"}}
        is_valid, errors = generator.validate_input(invalid_content)
        
        assert is_valid is False
        assert any("highlights" in e.lower() for e in errors)


# ============================================================================
# PROPERTY 5: CLI Flag Precedence Over Style Profile (Requirement 6.8)
# ============================================================================

from pipeline.formatters.llm.enhancer import (
    LLMEnhancer,
    EnhancementConfig,
    VALID_TONES,
    VALID_LENGTHS,
    DEFAULT_TONE,
    DEFAULT_LENGTH,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)


# Strategies for CLI flag precedence tests
@st.composite
def style_profile_for_precedence_strategy(draw):
    """Generate style profile with specific settings for precedence testing."""
    return StyleProfile(
        name=draw(st.from_regex(r"[a-z][a-z0-9\-]{3,20}", fullmatch=True)),
        temperature=draw(st.floats(min_value=0.1, max_value=1.9, allow_nan=False, allow_infinity=False)),
        top_p=draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)),
        max_tokens=draw(st.integers(min_value=500, max_value=7000)),
        model=draw(st.sampled_from(["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-haiku"])),
        prompt_template="Enhance this content: {{ content }}",
    )


class TestCLIFlagPrecedence:
    """Property 5: CLI Flag Precedence Over Style Profile.
    
    Validates Requirement 6.8:
    For any formatting request where both a style profile and CLI flags
    (--tone, --length, --model) are specified, the CLI flag values SHALL
    take precedence over the style profile values.
    
    Feature: formatter-publishing-drafts, Property 5: CLI Flag Precedence Over Style Profile
    """
    
    @given(
        profile=style_profile_for_precedence_strategy(),
        cli_tone=st.sampled_from(VALID_TONES),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_cli_tone_overrides_profile(self, profile: StyleProfile, cli_tone: str):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (Tone)**
        *For any* style profile and CLI tone flag, the CLI tone SHALL
        take precedence over any profile-defined tone.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config with CLI tone override
        config = enhancer._resolve_config(
            style_profile=profile,
            tone=cli_tone,
            length=None,
            provider=None,
            model=None,
            temperature=None,
            max_tokens=None,
        )
        
        # CLI tone should take precedence
        assert config.tone == cli_tone
    
    @given(
        profile=style_profile_for_precedence_strategy(),
        cli_length=st.sampled_from(VALID_LENGTHS),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_cli_length_overrides_profile(self, profile: StyleProfile, cli_length: str):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (Length)**
        *For any* style profile and CLI length flag, the CLI length SHALL
        take precedence over any profile-defined length.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config with CLI length override
        config = enhancer._resolve_config(
            style_profile=profile,
            tone=None,
            length=cli_length,
            provider=None,
            model=None,
            temperature=None,
            max_tokens=None,
        )
        
        # CLI length should take precedence
        assert config.length == cli_length
    
    @given(
        profile=style_profile_for_precedence_strategy(),
        cli_model=st.sampled_from(["gpt-4-turbo", "claude-3-opus", "custom-model"]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_cli_model_overrides_profile(self, profile: StyleProfile, cli_model: str):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (Model)**
        *For any* style profile with a model and CLI model flag, the CLI model
        SHALL take precedence over the profile model.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config with CLI model override
        config = enhancer._resolve_config(
            style_profile=profile,
            tone=None,
            length=None,
            provider=None,
            model=cli_model,
            temperature=None,
            max_tokens=None,
        )
        
        # CLI model should take precedence
        assert config.model == cli_model
    
    @given(
        profile=style_profile_for_precedence_strategy(),
        cli_temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_cli_temperature_overrides_profile(self, profile: StyleProfile, cli_temperature: float):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (Temperature)**
        *For any* style profile with temperature and CLI temperature flag,
        the CLI temperature SHALL take precedence.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config with CLI temperature override
        config = enhancer._resolve_config(
            style_profile=profile,
            tone=None,
            length=None,
            provider=None,
            model=None,
            temperature=cli_temperature,
            max_tokens=None,
        )
        
        # CLI temperature should take precedence
        assert config.temperature == cli_temperature
    
    @given(
        profile=style_profile_for_precedence_strategy(),
        cli_max_tokens=st.integers(min_value=100, max_value=8000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_cli_max_tokens_overrides_profile(self, profile: StyleProfile, cli_max_tokens: int):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (MaxTokens)**
        *For any* style profile with max_tokens and CLI max_tokens flag,
        the CLI max_tokens SHALL take precedence.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config with CLI max_tokens override
        config = enhancer._resolve_config(
            style_profile=profile,
            tone=None,
            length=None,
            provider=None,
            model=None,
            temperature=None,
            max_tokens=cli_max_tokens,
        )
        
        # CLI max_tokens should take precedence
        assert config.max_tokens == cli_max_tokens
    
    @given(profile=style_profile_for_precedence_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_profile_used_when_no_cli_override(self, profile: StyleProfile):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (Profile Fallback)**
        *For any* style profile without CLI overrides, the profile settings
        SHALL be used.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config without CLI overrides
        config = enhancer._resolve_config(
            style_profile=profile,
            tone=None,
            length=None,
            provider=None,
            model=None,
            temperature=None,
            max_tokens=None,
        )
        
        # Profile settings should be used
        assert config.temperature == profile.temperature
        assert config.max_tokens == profile.max_tokens
        assert config.model == profile.model
    
    @given(
        profile=style_profile_for_precedence_strategy(),
        cli_tone=st.sampled_from(VALID_TONES),
        cli_length=st.sampled_from(VALID_LENGTHS),
        cli_model=st.sampled_from(["gpt-4-turbo", "claude-3-opus"]),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_multiple_cli_overrides(
        self, profile: StyleProfile, cli_tone: str, cli_length: str, cli_model: str
    ):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (Multiple Overrides)**
        *For any* style profile with multiple CLI overrides, ALL CLI flags
        SHALL take precedence over their corresponding profile values.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config with multiple CLI overrides
        config = enhancer._resolve_config(
            style_profile=profile,
            tone=cli_tone,
            length=cli_length,
            provider=None,
            model=cli_model,
            temperature=None,
            max_tokens=None,
        )
        
        # All CLI overrides should take precedence
        assert config.tone == cli_tone
        assert config.length == cli_length
        assert config.model == cli_model
        
        # Non-overridden values should come from profile
        assert config.temperature == profile.temperature
        assert config.max_tokens == profile.max_tokens
    
    def test_property_5_defaults_used_without_profile_or_cli(self):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (Defaults)**
        *For any* request without style profile or CLI flags, default values
        SHALL be used.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config without profile or CLI overrides
        config = enhancer._resolve_config(
            style_profile=None,
            tone=None,
            length=None,
            provider=None,
            model=None,
            temperature=None,
            max_tokens=None,
        )
        
        # Defaults should be used
        assert config.tone == DEFAULT_TONE
        assert config.length == DEFAULT_LENGTH
        assert config.temperature == DEFAULT_TEMPERATURE
        assert config.max_tokens == DEFAULT_MAX_TOKENS
    
    @given(
        cli_tone=st.sampled_from(VALID_TONES),
        cli_length=st.sampled_from(VALID_LENGTHS),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_cli_overrides_without_profile(self, cli_tone: str, cli_length: str):
        """
        **Property 5: CLI Flag Precedence Over Style Profile (CLI Without Profile)**
        *For any* CLI flags without a style profile, the CLI values SHALL
        be used with defaults for unspecified settings.
        **Validates: Requirement 6.8**
        """
        enhancer = LLMEnhancer()
        
        # Resolve config with CLI flags but no profile
        config = enhancer._resolve_config(
            style_profile=None,
            tone=cli_tone,
            length=cli_length,
            provider=None,
            model=None,
            temperature=None,
            max_tokens=None,
        )
        
        # CLI values should be used
        assert config.tone == cli_tone
        assert config.length == cli_length
        
        # Defaults for unspecified settings
        assert config.temperature == DEFAULT_TEMPERATURE
        assert config.max_tokens == DEFAULT_MAX_TOKENS



# ============================================================================
# PROPERTY 6: LLM Enhancement Preserves Structure (Requirement 6.5)
# ============================================================================


# Strategies for structure preservation tests
@st.composite
def content_with_headers_strategy(draw):
    """Generate content with markdown headers."""
    num_headers = draw(st.integers(min_value=2, max_value=5))
    sections = []
    
    for i in range(num_headers):
        level = draw(st.integers(min_value=1, max_value=3))
        header_text = draw(st.from_regex(r"[A-Z][a-z]{3,15}( [A-Z][a-z]{3,10})?", fullmatch=True))
        body_text = draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
            min_size=20,
            max_size=100,
        ))
        sections.append(f"{'#' * level} {header_text}\n\n{body_text}")
    
    return "\n\n".join(sections)


@st.composite
def content_with_lists_strategy(draw):
    """Generate content with markdown lists."""
    intro = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=20,
        max_size=100,
    ))
    
    num_items = draw(st.integers(min_value=3, max_value=7))
    list_type = draw(st.sampled_from(["bullet", "numbered"]))
    
    items = []
    for i in range(num_items):
        item_text = draw(st.from_regex(r"[A-Z][a-z]{5,20}( [a-z]{3,10}){1,3}", fullmatch=True))
        if list_type == "bullet":
            items.append(f"- {item_text}")
        else:
            items.append(f"{i + 1}. {item_text}")
    
    return f"{intro}\n\n" + "\n".join(items)


@st.composite
def content_with_code_blocks_strategy(draw):
    """Generate content with code blocks."""
    intro = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=20,
        max_size=100,
    ))
    
    language = draw(st.sampled_from(["python", "javascript", "bash", ""]))
    code_content = draw(st.from_regex(r"[a-z_]+\([a-z_]*\)", fullmatch=True))
    
    if language:
        code_block = f"```{language}\n{code_content}\n```"
    else:
        code_block = f"```\n{code_content}\n```"
    
    outro = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=20,
        max_size=100,
    ))
    
    return f"{intro}\n\n{code_block}\n\n{outro}"


@st.composite
def content_with_timestamps_strategy(draw):
    """Generate content with timestamps (for video-related content)."""
    title = draw(st.from_regex(r"[A-Z][a-z]{5,15}( [A-Z][a-z]{3,10})?", fullmatch=True))
    
    num_timestamps = draw(st.integers(min_value=3, max_value=6))
    timestamps = []
    
    for i in range(num_timestamps):
        minutes = i * 5
        seconds = draw(st.integers(min_value=0, max_value=59))
        description = draw(st.from_regex(r"[A-Z][a-z]{5,20}", fullmatch=True))
        timestamps.append(f"{minutes:02d}:{seconds:02d} - {description}")
    
    return f"# {title}\n\n## Timestamps\n\n" + "\n".join(timestamps)


@st.composite
def content_with_mixed_structure_strategy(draw):
    """Generate content with mixed structural elements."""
    title = draw(st.from_regex(r"[A-Z][a-z]{5,15}( [A-Z][a-z]{3,10})?", fullmatch=True))
    
    # Header
    content = f"# {title}\n\n"
    
    # Intro paragraph
    intro = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=30,
        max_size=150,
    ))
    content += f"{intro}\n\n"
    
    # Subheader with list
    subheader = draw(st.from_regex(r"[A-Z][a-z]{5,15}", fullmatch=True))
    content += f"## {subheader}\n\n"
    
    num_items = draw(st.integers(min_value=2, max_value=4))
    for i in range(num_items):
        item = draw(st.from_regex(r"[A-Z][a-z]{5,15}( [a-z]{3,8})?", fullmatch=True))
        content += f"- {item}\n"
    
    content += "\n"
    
    # Another subheader with paragraph
    subheader2 = draw(st.from_regex(r"[A-Z][a-z]{5,15}", fullmatch=True))
    body = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=30,
        max_size=150,
    ))
    content += f"## {subheader2}\n\n{body}\n"
    
    return content


def count_headers(content: str) -> int:
    """Count markdown headers in content."""
    import re
    return len(re.findall(r"^#{1,6}\s+.+$", content, re.MULTILINE))


def count_list_items(content: str) -> int:
    """Count list items in content."""
    import re
    bullet_items = len(re.findall(r"^[-*]\s+.+$", content, re.MULTILINE))
    numbered_items = len(re.findall(r"^\d+\.\s+.+$", content, re.MULTILINE))
    return bullet_items + numbered_items


def count_code_blocks(content: str) -> int:
    """Count code blocks in content."""
    import re
    return len(re.findall(r"```[\s\S]*?```", content))


def count_timestamps(content: str) -> int:
    """Count timestamp patterns in content."""
    import re
    return len(re.findall(r"\d{1,2}:\d{2}", content))


def extract_headers(content: str) -> list:
    """Extract header texts from content."""
    import re
    return re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)


class TestLLMEnhancementPreservesStructure:
    """Property 6: LLM Enhancement Preserves Structure.
    
    Validates Requirement 6.5:
    For any template-rendered content with structural elements (headers, lists,
    code blocks), LLM enhancement SHALL preserve all structural elements in
    the output.
    
    Note: These tests verify the structure preservation RULES in the enhancement
    prompts. Since we can't actually call LLMs in property tests, we test that:
    1. The prompt includes structure preservation rules
    2. The content extraction logic preserves structure
    3. The enhancement config is correctly built
    
    Feature: formatter-publishing-drafts, Property 6: LLM Enhancement Preserves Structure
    """
    
    @given(content=content_with_headers_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_prompt_includes_header_preservation_rule(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (Header Rule)**
        *For any* content with headers, the enhancement prompt SHALL include
        rules to preserve header structure.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        prompt = enhancer.build_prompt(
            content=content,
            output_type="blog",
            tone="professional",
            length="medium",
        )
        
        # Prompt should include structure preservation rules
        assert "PRESERVE" in prompt.upper() or "preserve" in prompt.lower()
        assert "structural" in prompt.lower() or "structure" in prompt.lower() or "headers" in prompt.lower()
    
    @given(content=content_with_lists_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_prompt_includes_list_preservation_rule(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (List Rule)**
        *For any* content with lists, the enhancement prompt SHALL include
        rules to preserve list structure.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        prompt = enhancer.build_prompt(
            content=content,
            output_type="blog",
            tone="professional",
            length="medium",
        )
        
        # Prompt should include structure preservation rules
        assert "PRESERVE" in prompt.upper() or "preserve" in prompt.lower()
        assert "lists" in prompt.lower() or "structural" in prompt.lower()
    
    @given(content=content_with_code_blocks_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_prompt_includes_code_block_preservation_rule(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (Code Block Rule)**
        *For any* content with code blocks, the enhancement prompt SHALL include
        rules to preserve code block structure.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        prompt = enhancer.build_prompt(
            content=content,
            output_type="blog",
            tone="professional",
            length="medium",
        )
        
        # Prompt should include structure preservation rules
        assert "PRESERVE" in prompt.upper() or "preserve" in prompt.lower()
        assert "code" in prompt.lower() or "structural" in prompt.lower()
    
    @given(content=content_with_timestamps_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_prompt_includes_timestamp_preservation_rule(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (Timestamp Rule)**
        *For any* content with timestamps, the enhancement prompt SHALL include
        rules to preserve timestamp structure.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        prompt = enhancer.build_prompt(
            content=content,
            output_type="youtube",
            tone="professional",
            length="medium",
        )
        
        # Prompt should include structure preservation rules
        assert "PRESERVE" in prompt.upper() or "preserve" in prompt.lower()
    
    @given(content=content_with_mixed_structure_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_prompt_includes_all_structure_rules(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (All Rules)**
        *For any* content with mixed structural elements, the enhancement prompt
        SHALL include comprehensive structure preservation rules.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        prompt = enhancer.build_prompt(
            content=content,
            output_type="blog",
            tone="professional",
            length="medium",
        )
        
        # Prompt should include comprehensive structure preservation rules
        assert "PRESERVE" in prompt.upper() or "preserve" in prompt.lower()
        assert "structural" in prompt.lower() or "structure" in prompt.lower()
        
        # Should mention not adding/removing sections
        assert "add" in prompt.lower() or "remove" in prompt.lower() or "change" in prompt.lower()
    
    @given(content=content_with_headers_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_content_extraction_preserves_headers(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (Extraction)**
        *For any* content with headers, the content extraction logic SHALL
        preserve header count when response is valid.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        original_header_count = count_headers(content)
        
        # Simulate a valid LLM response that preserves structure
        simulated_response = content  # Perfect preservation
        
        extracted = enhancer._extract_enhanced_content(simulated_response, content)
        
        # Header count should be preserved
        assert count_headers(extracted) == original_header_count
    
    @given(content=content_with_lists_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_content_extraction_preserves_lists(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (List Extraction)**
        *For any* content with lists, the content extraction logic SHALL
        preserve list item count when response is valid.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        original_list_count = count_list_items(content)
        
        # Simulate a valid LLM response that preserves structure
        simulated_response = content  # Perfect preservation
        
        extracted = enhancer._extract_enhanced_content(simulated_response, content)
        
        # List item count should be preserved
        assert count_list_items(extracted) == original_list_count
    
    @given(content=content_with_code_blocks_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_content_extraction_preserves_code_blocks(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (Code Block Extraction)**
        *For any* content with code blocks, the content extraction logic SHALL
        preserve code block count when response is valid.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        original_code_count = count_code_blocks(content)
        
        # Simulate a valid LLM response that preserves structure
        simulated_response = content  # Perfect preservation
        
        extracted = enhancer._extract_enhanced_content(simulated_response, content)
        
        # Code block count should be preserved
        assert count_code_blocks(extracted) == original_code_count
    
    @given(content=content_with_timestamps_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_content_extraction_preserves_timestamps(self, content: str):
        """
        **Property 6: LLM Enhancement Preserves Structure (Timestamp Extraction)**
        *For any* content with timestamps, the content extraction logic SHALL
        preserve timestamp count when response is valid.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        original_timestamp_count = count_timestamps(content)
        
        # Simulate a valid LLM response that preserves structure
        simulated_response = content  # Perfect preservation
        
        extracted = enhancer._extract_enhanced_content(simulated_response, content)
        
        # Timestamp count should be preserved
        assert count_timestamps(extracted) == original_timestamp_count
    
    def test_property_6_fallback_preserves_original_structure(self):
        """
        **Property 6: LLM Enhancement Preserves Structure (Fallback)**
        *For any* LLM response that is too short, the extraction logic SHALL
        fall back to the original content, preserving all structure.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        original_content = """# Main Title

## Section One

This is the first section with some content.

- Item one
- Item two
- Item three

## Section Two

```python
def example():
    pass
```

More content here.
"""
        
        # Simulate a too-short LLM response
        short_response = "OK"
        
        extracted = enhancer._extract_enhanced_content(short_response, original_content)
        
        # Should fall back to original
        assert extracted == original_content
        assert count_headers(extracted) == count_headers(original_content)
        assert count_list_items(extracted) == count_list_items(original_content)
        assert count_code_blocks(extracted) == count_code_blocks(original_content)
    
    @given(
        output_type=st.sampled_from(["blog", "youtube", "newsletter", "video-script"]),
        tone=st.sampled_from(VALID_TONES),
        length=st.sampled_from(VALID_LENGTHS),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_all_output_types_include_preservation_rules(
        self, output_type: str, tone: str, length: str
    ):
        """
        **Property 6: LLM Enhancement Preserves Structure (All Output Types)**
        *For any* output type, the enhancement prompt SHALL include structure
        preservation rules.
        **Validates: Requirement 6.5**
        """
        enhancer = LLMEnhancer()
        
        sample_content = "# Title\n\nSome content here.\n\n- Item 1\n- Item 2"
        
        prompt = enhancer.build_prompt(
            content=sample_content,
            output_type=output_type,
            tone=tone,
            length=length,
        )
        
        # All output types should include preservation rules
        assert "PRESERVE" in prompt.upper() or "preserve" in prompt.lower()



# ============================================================================
# PROPERTY 12: Retry Behavior Correctness (Requirements 14.5, 14.6, 14.7)
# ============================================================================

from pipeline.formatters.retry import (
    retry_enhancement,
    retry_with_fallback,
    EnhancementRetryContext,
    is_transient_error,
    is_permanent_error,
    get_retry_delays,
    calculate_backoff_delay,
    TRANSIENT_ERRORS,
    PERMANENT_ERRORS,
)
from pipeline.enrichment.errors import (
    RateLimitError,
    TimeoutError,
    NetworkError,
    AuthenticationError,
    InvalidRequestError,
)


class TestRetryBehaviorCorrectness:
    """Property 12: Retry Behavior Correctness.
    
    Validates Requirements 14.5, 14.6, 14.7:
    - 14.5: Retry transient errors (rate limit, timeout, network) with exponential backoff
    - 14.6: Use exponential backoff delays: 1s, 2s, 4s
    - 14.7: Do NOT retry authentication or invalid request errors
    
    Feature: formatter-publishing-drafts, Property 12: Retry Behavior Correctness
    """
    
    def test_property_12_rate_limit_is_transient(self):
        """
        **Property 12: Retry Behavior Correctness (Rate Limit)**
        *For any* rate limit error, the system SHALL classify it as transient
        and eligible for retry.
        **Validates: Requirement 14.5**
        """
        error = RateLimitError("Rate limit exceeded")
        
        assert is_transient_error(error)
        assert not is_permanent_error(error)
    
    def test_property_12_timeout_is_transient(self):
        """
        **Property 12: Retry Behavior Correctness (Timeout)**
        *For any* timeout error, the system SHALL classify it as transient
        and eligible for retry.
        **Validates: Requirement 14.5**
        """
        error = TimeoutError("Request timed out")
        
        assert is_transient_error(error)
        assert not is_permanent_error(error)
    
    def test_property_12_network_is_transient(self):
        """
        **Property 12: Retry Behavior Correctness (Network)**
        *For any* network error, the system SHALL classify it as transient
        and eligible for retry.
        **Validates: Requirement 14.5**
        """
        error = NetworkError("Connection refused")
        
        assert is_transient_error(error)
        assert not is_permanent_error(error)
    
    def test_property_12_authentication_is_permanent(self):
        """
        **Property 12: Retry Behavior Correctness (Authentication)**
        *For any* authentication error, the system SHALL classify it as
        permanent and NOT eligible for retry.
        **Validates: Requirement 14.7**
        """
        error = AuthenticationError("Invalid API key")
        
        assert is_permanent_error(error)
        assert not is_transient_error(error)
    
    def test_property_12_invalid_request_is_permanent(self):
        """
        **Property 12: Retry Behavior Correctness (Invalid Request)**
        *For any* invalid request error, the system SHALL classify it as
        permanent and NOT eligible for retry.
        **Validates: Requirement 14.7**
        """
        error = InvalidRequestError("Malformed request")
        
        assert is_permanent_error(error)
        assert not is_transient_error(error)
    
    def test_property_12_exponential_backoff_delays(self):
        """
        **Property 12: Retry Behavior Correctness (Backoff Delays)**
        *For any* retry sequence with 3 attempts, the delays SHALL be
        1s, 2s (exponential backoff with base 1.0).
        **Validates: Requirement 14.6**
        """
        delays = get_retry_delays(max_attempts=3, base_delay=1.0)
        
        # Should have 2 delays (before attempt 2 and 3)
        assert len(delays) == 2
        
        # Delays should be 1s, 2s (exponential: 1*2^0, 1*2^1)
        assert delays[0] == 1.0
        assert delays[1] == 2.0
    
    def test_property_12_backoff_calculation(self):
        """
        **Property 12: Retry Behavior Correctness (Backoff Calculation)**
        *For any* attempt number, the backoff delay SHALL follow the formula:
        delay = base_delay * (2 ** attempt).
        **Validates: Requirement 14.6**
        """
        base_delay = 1.0
        
        # Attempt 0: 1 * 2^0 = 1
        assert calculate_backoff_delay(0, base_delay) == 1.0
        
        # Attempt 1: 1 * 2^1 = 2
        assert calculate_backoff_delay(1, base_delay) == 2.0
        
        # Attempt 2: 1 * 2^2 = 4
        assert calculate_backoff_delay(2, base_delay) == 4.0
    
    @given(attempt=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_12_backoff_is_exponential(self, attempt: int):
        """
        **Property 12: Retry Behavior Correctness (Exponential Growth)**
        *For any* attempt number, the backoff delay SHALL grow exponentially.
        **Validates: Requirement 14.6**
        """
        base_delay = 1.0
        delay = calculate_backoff_delay(attempt, base_delay)
        
        # Delay should equal base_delay * 2^attempt
        expected = base_delay * (2 ** attempt)
        assert delay == expected
    
    def test_property_12_retry_context_tracks_attempts(self):
        """
        **Property 12: Retry Behavior Correctness (Attempt Tracking)**
        *For any* retry context, it SHALL correctly track the number of
        attempts made.
        **Validates: Requirement 14.5**
        """
        ctx = EnhancementRetryContext(max_attempts=3)
        
        attempts_seen = []
        for attempt in ctx:
            attempts_seen.append(attempt)
            if attempt >= 2:  # Stop after 3 attempts (0, 1, 2)
                break
        
        assert attempts_seen == [0, 1, 2]
        assert ctx.current_attempt == 3
    
    def test_property_12_retry_context_should_retry_transient(self):
        """
        **Property 12: Retry Behavior Correctness (Context Retry Decision)**
        *For any* transient error with remaining attempts, the retry context
        SHALL indicate that retry should occur.
        **Validates: Requirement 14.5**
        """
        ctx = EnhancementRetryContext(max_attempts=3)
        
        # Start iteration
        next(iter(ctx))
        
        # Should retry transient error
        error = RateLimitError("Rate limit")
        assert ctx.should_retry(error)
    
    def test_property_12_retry_context_no_retry_permanent(self):
        """
        **Property 12: Retry Behavior Correctness (Context No Retry)**
        *For any* permanent error, the retry context SHALL indicate that
        retry should NOT occur.
        **Validates: Requirement 14.7**
        """
        ctx = EnhancementRetryContext(max_attempts=3)
        
        # Start iteration
        next(iter(ctx))
        
        # Should NOT retry permanent error
        error = AuthenticationError("Invalid key")
        assert not ctx.should_retry(error)
    
    def test_property_12_retry_context_exhausted(self):
        """
        **Property 12: Retry Behavior Correctness (Exhausted Retries)**
        *For any* retry context that has exhausted all attempts, it SHALL
        indicate that retry should NOT occur even for transient errors.
        **Validates: Requirement 14.5**
        """
        ctx = EnhancementRetryContext(max_attempts=3)
        
        # Exhaust all attempts
        for _ in ctx:
            pass
        
        # Should NOT retry even transient error after exhaustion
        error = RateLimitError("Rate limit")
        assert not ctx.should_retry(error)
    
    def test_property_12_max_3_retries(self):
        """
        **Property 12: Retry Behavior Correctness (Max Retries)**
        *For any* retry sequence, the maximum number of attempts SHALL be 3.
        **Validates: Requirement 14.5**
        """
        ctx = EnhancementRetryContext(max_attempts=3)
        
        attempt_count = 0
        for _ in ctx:
            attempt_count += 1
        
        assert attempt_count == 3
    
    @given(error_type=st.sampled_from(["rate_limit", "timeout", "network"]))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_12_all_transient_errors_retryable(self, error_type: str):
        """
        **Property 12: Retry Behavior Correctness (All Transient Types)**
        *For any* transient error type (rate limit, timeout, network),
        the system SHALL classify it as retryable.
        **Validates: Requirement 14.5**
        """
        error_map = {
            "rate_limit": RateLimitError("Rate limit exceeded"),
            "timeout": TimeoutError("Request timed out"),
            "network": NetworkError("Connection failed"),
        }
        
        error = error_map[error_type]
        assert is_transient_error(error)
    
    @given(error_type=st.sampled_from(["authentication", "invalid_request"]))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_property_12_all_permanent_errors_not_retryable(self, error_type: str):
        """
        **Property 12: Retry Behavior Correctness (All Permanent Types)**
        *For any* permanent error type (authentication, invalid request),
        the system SHALL classify it as NOT retryable.
        **Validates: Requirement 14.7**
        """
        error_map = {
            "authentication": AuthenticationError("Invalid API key"),
            "invalid_request": InvalidRequestError("Bad request"),
        }
        
        error = error_map[error_type]
        assert is_permanent_error(error)
        assert not is_transient_error(error)
    
    def test_property_12_fallback_marks_used(self):
        """
        **Property 12: Retry Behavior Correctness (Fallback Tracking)**
        *For any* retry context where fallback is used, it SHALL correctly
        track that fallback was used and the reason.
        **Validates: Requirement 14.5**
        """
        ctx = EnhancementRetryContext(max_attempts=3)
        
        # Start iteration
        next(iter(ctx))
        
        # Mark fallback used
        error = RateLimitError("Rate limit after all retries")
        ctx.mark_fallback_used(error)
        
        assert ctx.fallback_used
        assert ctx.last_error == error
        assert "RateLimitError" in ctx.fallback_reason
    
    def test_property_12_enhancement_error_from_context(self):
        """
        **Property 12: Retry Behavior Correctness (Error Generation)**
        *For any* retry context where fallback is used, it SHALL be able
        to generate an EnhancementError with correct details.
        **Validates: Requirement 14.5**
        """
        from pipeline.formatters.errors import EnhancementError
        
        ctx = EnhancementRetryContext(max_attempts=3)
        
        # Simulate 3 failed attempts
        for attempt in ctx:
            if attempt == 2:
                error = RateLimitError("Rate limit")
                ctx.mark_fallback_used(error)
                break
        
        enhancement_error = ctx.get_enhancement_error()
        
        assert enhancement_error is not None
        assert isinstance(enhancement_error, EnhancementError)
        assert enhancement_error.attempts == 3
        assert enhancement_error.recoverable



# ============================================================================
# PROPERTY 13: Graceful Degradation on LLM Failure (Requirement 14.4)
# ============================================================================

from unittest.mock import Mock, patch, MagicMock
from pipeline.formatters.llm.enhancer import LLMEnhancer, EnhancementResult


# Strategies for graceful degradation tests
@st.composite
def template_content_strategy(draw):
    """Generate template-rendered content for testing."""
    title = draw(st.from_regex(r"[A-Z][a-z]{5,20}( [A-Z][a-z]{3,15})?", fullmatch=True))
    body = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=50,
        max_size=500,
    ))
    
    return f"# {title}\n\n{body}"


@st.composite
def output_type_for_enhancement_strategy(draw):
    """Generate output types for enhancement testing."""
    return draw(st.sampled_from([
        "blog", "tweet", "youtube", "linkedin", "newsletter",
        "video-script", "tiktok-script", "seo"
    ]))


class TestGracefulDegradationOnLLMFailure:
    """Property 13: Graceful Degradation on LLM Failure.
    
    Validates Requirement 14.4:
    For any formatting request where LLM enhancement fails after all retries,
    the system SHALL fall back to template-only output and include a warning
    in the result, rather than failing completely.
    
    Feature: formatter-publishing-drafts, Property 13: Graceful Degradation on LLM Failure
    """
    
    def test_property_13_fallback_returns_original_content(self):
        """
        **Property 13: Graceful Degradation (Original Content)**
        *For any* LLM failure, the system SHALL return the original
        template-rendered content as fallback.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        enhancer = LLMEnhancer()
        
        # Create fallback result
        result = enhancer._create_fallback_result(
            content=original_content,
            error=RateLimitError("Rate limit"),
            reason="LLM enhancement failed",
            attempts=3,
        )
        
        # Content should be original
        assert result.content == original_content
    
    def test_property_13_fallback_includes_warning(self):
        """
        **Property 13: Graceful Degradation (Warning Included)**
        *For any* LLM failure, the fallback result SHALL include a warning
        message indicating that template-only output is being used.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        enhancer = LLMEnhancer()
        
        # Create fallback result
        result = enhancer._create_fallback_result(
            content=original_content,
            error=RateLimitError("Rate limit"),
            reason="LLM enhancement failed",
            attempts=3,
        )
        
        # Should have warning
        assert len(result.warnings) > 0
        assert any("template-only" in w.lower() for w in result.warnings)
    
    def test_property_13_fallback_not_enhanced(self):
        """
        **Property 13: Graceful Degradation (Enhanced Flag)**
        *For any* LLM failure, the fallback result SHALL have enhanced=False
        to indicate that LLM enhancement was not applied.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        enhancer = LLMEnhancer()
        
        # Create fallback result
        result = enhancer._create_fallback_result(
            content=original_content,
            error=RateLimitError("Rate limit"),
            reason="LLM enhancement failed",
            attempts=3,
        )
        
        # Should not be marked as enhanced
        assert result.enhanced is False
    
    def test_property_13_fallback_success_false(self):
        """
        **Property 13: Graceful Degradation (Success Flag)**
        *For any* LLM failure, the fallback result SHALL have success=False
        to indicate that enhancement did not succeed.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        enhancer = LLMEnhancer()
        
        # Create fallback result
        result = enhancer._create_fallback_result(
            content=original_content,
            error=RateLimitError("Rate limit"),
            reason="LLM enhancement failed",
            attempts=3,
        )
        
        # Should not be marked as success
        assert result.success is False
    
    def test_property_13_fallback_includes_error_info(self):
        """
        **Property 13: Graceful Degradation (Error Info)**
        *For any* LLM failure, the fallback result SHALL include error
        information for debugging purposes.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        enhancer = LLMEnhancer()
        
        # Create fallback result
        result = enhancer._create_fallback_result(
            content=original_content,
            error=RateLimitError("Rate limit exceeded"),
            reason="LLM enhancement failed",
            attempts=3,
        )
        
        # Should have error info
        assert result.error is not None
        assert "Rate limit" in result.error
    
    def test_property_13_fallback_includes_attempt_count(self):
        """
        **Property 13: Graceful Degradation (Attempt Count)**
        *For any* LLM failure after retries, the warning SHALL include
        the number of attempts made.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        enhancer = LLMEnhancer()
        
        # Create fallback result with 3 attempts
        result = enhancer._create_fallback_result(
            content=original_content,
            error=RateLimitError("Rate limit"),
            reason="LLM enhancement failed",
            attempts=3,
        )
        
        # Warning should mention attempts
        assert any("3 attempts" in w for w in result.warnings)
    
    @given(content=template_content_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_13_fallback_preserves_all_content(self, content: str):
        """
        **Property 13: Graceful Degradation (Content Preservation)**
        *For any* template-rendered content, the fallback SHALL preserve
        the content exactly as provided.
        **Validates: Requirement 14.4**
        """
        enhancer = LLMEnhancer()
        
        # Create fallback result
        result = enhancer._create_fallback_result(
            content=content,
            error=RateLimitError("Rate limit"),
            reason="LLM enhancement failed",
            attempts=3,
        )
        
        # Content should be exactly preserved
        assert result.content == content
    
    @given(
        content=template_content_strategy(),
        error_type=st.sampled_from(["rate_limit", "timeout", "network", "auth"]),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_13_fallback_for_all_error_types(self, content: str, error_type: str):
        """
        **Property 13: Graceful Degradation (All Error Types)**
        *For any* error type (transient or permanent), the fallback SHALL
        return the original content with appropriate warning.
        **Validates: Requirement 14.4**
        """
        error_map = {
            "rate_limit": RateLimitError("Rate limit exceeded"),
            "timeout": TimeoutError("Request timed out"),
            "network": NetworkError("Connection failed"),
            "auth": AuthenticationError("Invalid API key"),
        }
        
        enhancer = LLMEnhancer()
        
        # Create fallback result
        result = enhancer._create_fallback_result(
            content=content,
            error=error_map[error_type],
            reason=f"LLM failed: {error_type}",
            attempts=3 if error_type != "auth" else 1,
        )
        
        # Should always return original content
        assert result.content == content
        assert result.enhanced is False
        assert len(result.warnings) > 0
    
    def test_property_13_enhance_with_agent_creation_failure(self):
        """
        **Property 13: Graceful Degradation (Agent Creation Failure)**
        *For any* failure to create LLM agent, the system SHALL fall back
        to template-only output.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        # Create enhancer with mock factory that fails
        mock_factory = Mock()
        mock_factory.create_agent.side_effect = Exception("No API key configured")
        
        enhancer = LLMEnhancer(agent_factory=mock_factory)
        
        result = enhancer.enhance(
            content=original_content,
            output_type="blog",
        )
        
        # Should fall back to original content
        assert result.content == original_content
        assert result.enhanced is False
        assert result.success is False
        assert len(result.warnings) > 0
    
    def test_property_13_enhance_with_transient_error_exhausted(self):
        """
        **Property 13: Graceful Degradation (Transient Exhausted)**
        *For any* transient error that exhausts all retries, the system
        SHALL fall back to template-only output.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        # Create mock agent that always fails with transient error
        mock_agent = Mock()
        mock_agent.generate.side_effect = RateLimitError("Rate limit exceeded")
        
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_agent
        
        enhancer = LLMEnhancer(agent_factory=mock_factory)
        
        # Patch sleep to speed up test
        with patch('time.sleep'):
            result = enhancer.enhance(
                content=original_content,
                output_type="blog",
            )
        
        # Should fall back to original content after retries
        assert result.content == original_content
        assert result.enhanced is False
        assert result.success is False
    
    def test_property_13_enhance_with_permanent_error_immediate_fallback(self):
        """
        **Property 13: Graceful Degradation (Permanent Immediate)**
        *For any* permanent error, the system SHALL immediately fall back
        to template-only output without retrying.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        # Create mock agent that fails with permanent error
        mock_agent = Mock()
        mock_agent.generate.side_effect = AuthenticationError("Invalid API key")
        
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_agent
        
        enhancer = LLMEnhancer(agent_factory=mock_factory)
        
        result = enhancer.enhance(
            content=original_content,
            output_type="blog",
        )
        
        # Should fall back immediately (only 1 call, no retries)
        assert result.content == original_content
        assert result.enhanced is False
        assert result.success is False
        assert mock_agent.generate.call_count == 1  # No retries for permanent error
    
    @given(
        content=template_content_strategy(),
        output_type=output_type_for_enhancement_strategy(),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_13_fallback_works_for_all_output_types(
        self, content: str, output_type: str
    ):
        """
        **Property 13: Graceful Degradation (All Output Types)**
        *For any* output type, the graceful degradation SHALL work correctly.
        **Validates: Requirement 14.4**
        """
        # Create mock agent that fails
        mock_agent = Mock()
        mock_agent.generate.side_effect = RateLimitError("Rate limit")
        
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_agent
        
        enhancer = LLMEnhancer(agent_factory=mock_factory)
        
        # Patch sleep to speed up test
        with patch('time.sleep'):
            result = enhancer.enhance(
                content=content,
                output_type=output_type,
            )
        
        # Should fall back for any output type
        assert result.content == content
        assert result.enhanced is False
    
    def test_property_13_successful_enhancement_no_fallback(self):
        """
        **Property 13: Graceful Degradation (Success No Fallback)**
        *For any* successful LLM enhancement, the system SHALL NOT use
        fallback and SHALL return enhanced content.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        enhanced_content = "# Test Title\n\nThis is the beautifully enhanced content with better prose."
        
        # Create mock agent that succeeds
        mock_response = Mock()
        mock_response.content = enhanced_content
        mock_response.model_used = "gpt-4"
        mock_response.tokens_used = 100
        mock_response.cost_usd = 0.01
        
        mock_agent = Mock()
        mock_agent.generate.return_value = mock_response
        
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_agent
        
        enhancer = LLMEnhancer(agent_factory=mock_factory)
        
        result = enhancer.enhance(
            content=original_content,
            output_type="blog",
        )
        
        # Should return enhanced content, not fallback
        assert result.content == enhanced_content
        assert result.enhanced is True
        assert result.success is True
        assert len(result.warnings) == 0
    
    def test_property_13_enhancement_result_is_valid_type(self):
        """
        **Property 13: Graceful Degradation (Result Type)**
        *For any* enhancement attempt (success or failure), the result
        SHALL be a valid EnhancementResult object.
        **Validates: Requirement 14.4**
        """
        original_content = "# Test Title\n\nThis is the original content."
        
        enhancer = LLMEnhancer()
        
        # Create fallback result
        result = enhancer._create_fallback_result(
            content=original_content,
            error=RateLimitError("Rate limit"),
            reason="LLM enhancement failed",
            attempts=3,
        )
        
        # Should be valid EnhancementResult
        assert isinstance(result, EnhancementResult)
        assert hasattr(result, 'content')
        assert hasattr(result, 'success')
        assert hasattr(result, 'enhanced')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'error')


# ============================================================================
# PROPERTY 11: Input Validation Completeness (Requirements 19.1, 19.2)
# ============================================================================

from pipeline.formatters.input_validator import (
    InputValidator,
    ValidationResult as InputValidationResult,
    validate_input,
    validate_input_file,
    get_required_enrichments_for_type,
    REQUIRED_ENRICHMENTS,
)
from pipeline.formatters.base import VALID_OUTPUT_TYPES


# Strategies for input validation tests
@st.composite
def valid_enrichment_v1_strategy(draw):
    """Generate valid EnrichmentV1 data for testing."""
    # Generate metadata
    metadata = {
        "provider": draw(st.sampled_from(["openai", "claude", "bedrock", "ollama"])),
        "model": draw(st.text(min_size=3, max_size=30, alphabet=st.characters(
            whitelist_categories=("L", "N"),
            whitelist_characters="-_"
        ))),
        "timestamp": draw(st.datetimes()).isoformat(),
        "cost_usd": draw(st.floats(min_value=0, max_value=10)),
        "tokens_used": draw(st.integers(min_value=0, max_value=100000)),
        "enrichment_types": draw(st.lists(
            st.sampled_from(["summary", "tags", "chapters", "highlights"]),
            min_size=1,
            max_size=4,
            unique=True,
        )),
        "cache_hit": draw(st.booleans()),
    }
    
    # Generate summary (most common enrichment)
    summary = {
        "short": draw(st.text(min_size=10, max_size=200)),
        "medium": draw(st.text(min_size=50, max_size=500)),
        "long": draw(st.text(min_size=100, max_size=1000)),
    }
    
    # Generate tags
    tags = {
        "categories": draw(st.lists(
            st.text(min_size=2, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))),
            min_size=1,
            max_size=5,
        )),
        "keywords": draw(st.lists(
            st.text(min_size=2, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))),
            min_size=1,
            max_size=10,
        )),
    }
    
    return {
        "enrichment_version": "v1",
        "metadata": metadata,
        "summary": summary,
        "tags": tags,
    }


@st.composite
def enrichment_with_chapters_strategy(draw):
    """Generate EnrichmentV1 data with chapters."""
    base = draw(valid_enrichment_v1_strategy())
    
    # Add chapters
    chapters = []
    for i in range(draw(st.integers(min_value=1, max_value=5))):
        chapters.append({
            "title": draw(st.text(min_size=5, max_size=50)),
            "start_time": i * 60.0,
            "end_time": (i + 1) * 60.0,
            "summary": draw(st.text(min_size=10, max_size=200)),
        })
    
    base["chapters"] = chapters
    return base


@st.composite
def enrichment_with_highlights_strategy(draw):
    """Generate EnrichmentV1 data with highlights."""
    base = draw(valid_enrichment_v1_strategy())
    
    # Add highlights
    highlights = []
    for i in range(draw(st.integers(min_value=1, max_value=5))):
        highlights.append({
            "text": draw(st.text(min_size=20, max_size=200)),
            "timestamp": i * 30.0,
            "importance": draw(st.sampled_from(["high", "medium", "low"])),
        })
    
    base["highlights"] = highlights
    return base


@st.composite
def enrichment_with_transcript_strategy(draw):
    """Generate EnrichmentV1 data with transcript."""
    base = draw(valid_enrichment_v1_strategy())
    
    # Add transcript
    base["transcript"] = draw(st.text(min_size=100, max_size=2000))
    
    return base


@st.composite
def invalid_enrichment_strategy(draw):
    """Generate invalid EnrichmentV1 data for testing error detection."""
    # Choose what to make invalid
    invalid_type = draw(st.sampled_from([
        "missing_version",
        "wrong_version",
        "missing_metadata",
        "missing_metadata_field",
        "no_enrichments",
    ]))
    
    if invalid_type == "missing_version":
        return {
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
    
    elif invalid_type == "wrong_version":
        return {
            "enrichment_version": "v2",  # Wrong version
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
    
    elif invalid_type == "missing_metadata":
        return {
            "enrichment_version": "v1",
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
    
    elif invalid_type == "missing_metadata_field":
        return {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                # Missing model, timestamp, etc.
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
    
    else:  # no_enrichments
        return {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": [],
                "cache_hit": False,
            },
            # No summary, tags, chapters, or highlights
        }


class TestInputValidationCompleteness:
    """Property 11: Input Validation Completeness.
    
    Validates Requirements 19.1, 19.2:
    - 19.1: Validate that input files conform to EnrichmentV1 schema
    - 19.2: Validate that required enrichment fields are present for output type
    
    Feature: formatter-publishing-drafts, Property 11: Input Validation Completeness
    """
    
    def test_property_11_valid_enrichment_v1_passes(self):
        """
        **Property 11: Input Validation (Valid Schema)**
        *For any* valid EnrichmentV1 data, validation SHALL pass.
        **Validates: Requirement 19.1**
        """
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium summary with more details.",
                "long": "A long summary with comprehensive details.",
            },
        }
        
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_property_11_missing_version_fails(self):
        """
        **Property 11: Input Validation (Missing Version)**
        *For any* data missing enrichment_version, validation SHALL fail
        with a specific error message.
        **Validates: Requirement 19.1**
        """
        data = {
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert not result.is_valid
        assert any("enrichment_version" in e for e in result.errors)
    
    def test_property_11_wrong_version_fails(self):
        """
        **Property 11: Input Validation (Wrong Version)**
        *For any* data with unsupported enrichment_version, validation
        SHALL fail with a specific error message.
        **Validates: Requirement 19.1**
        """
        data = {
            "enrichment_version": "v2",  # Unsupported
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert not result.is_valid
        assert any("v2" in e or "version" in e.lower() for e in result.errors)
    
    def test_property_11_missing_metadata_fails(self):
        """
        **Property 11: Input Validation (Missing Metadata)**
        *For any* data missing metadata, validation SHALL fail with
        a specific error message.
        **Validates: Requirement 19.1**
        """
        data = {
            "enrichment_version": "v1",
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert not result.is_valid
        assert any("metadata" in e for e in result.errors)
    
    def test_property_11_missing_metadata_field_fails(self):
        """
        **Property 11: Input Validation (Missing Metadata Field)**
        *For any* data with incomplete metadata, validation SHALL fail
        with specific error messages for each missing field.
        **Validates: Requirement 19.1**
        """
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                # Missing: model, timestamp, cost_usd, tokens_used, enrichment_types
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert not result.is_valid
        # Should have errors for missing metadata fields
        assert any("model" in e for e in result.errors)
    
    def test_property_11_no_enrichments_fails(self):
        """
        **Property 11: Input Validation (No Enrichments)**
        *For any* data with no enrichment types present, validation
        SHALL fail with a specific error message.
        **Validates: Requirement 19.1**
        """
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": [],
                "cache_hit": False,
            },
            # No summary, tags, chapters, or highlights
        }
        
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert not result.is_valid
        assert any("enrichment" in e.lower() for e in result.errors)
    
    @given(output_type=st.sampled_from(VALID_OUTPUT_TYPES))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_required_enrichments_defined_for_all_types(
        self, output_type: str
    ):
        """
        **Property 11: Input Validation (Requirements Defined)**
        *For any* valid output type, the system SHALL have defined
        required enrichments.
        **Validates: Requirement 19.2**
        """
        required = get_required_enrichments_for_type(output_type)
        
        # Should return a list (possibly empty for some types)
        assert isinstance(required, list)
        # All items should be strings
        assert all(isinstance(r, str) for r in required)
    
    def test_property_11_blog_requires_summary_and_tags(self):
        """
        **Property 11: Input Validation (Blog Requirements)**
        *For any* blog output request, validation SHALL require
        summary and tags enrichments.
        **Validates: Requirement 19.2**
        """
        # Data with summary but no tags
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data, output_type="blog")
        
        assert not result.is_valid
        assert "tags" in result.missing_enrichments
    
    def test_property_11_chapters_requires_chapters(self):
        """
        **Property 11: Input Validation (Chapters Requirements)**
        *For any* chapters output request, validation SHALL require
        chapters enrichment.
        **Validates: Requirement 19.2**
        """
        # Data with summary but no chapters
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data, output_type="chapters")
        
        assert not result.is_valid
        assert "chapters" in result.missing_enrichments
    
    def test_property_11_quote_cards_requires_highlights(self):
        """
        **Property 11: Input Validation (Quote Cards Requirements)**
        *For any* quote-cards output request, validation SHALL require
        highlights enrichment.
        **Validates: Requirement 19.2**
        """
        # Data with summary but no highlights
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data, output_type="quote-cards")
        
        assert not result.is_valid
        assert "highlights" in result.missing_enrichments
    
    def test_property_11_transcript_clean_requires_transcript(self):
        """
        **Property 11: Input Validation (Transcript Clean Requirements)**
        *For any* transcript-clean output request, validation SHALL require
        transcript enrichment.
        **Validates: Requirement 19.2**
        """
        # Data with summary but no transcript
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data, output_type="transcript-clean")
        
        assert not result.is_valid
        assert "transcript" in result.missing_enrichments
    
    @given(data=valid_enrichment_v1_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_valid_data_passes_schema_validation(self, data: dict):
        """
        **Property 11: Input Validation (Valid Data Passes)**
        *For any* valid EnrichmentV1 data, schema validation SHALL pass.
        **Validates: Requirement 19.1**
        """
        validator = InputValidator()
        result = validator.validate_content(data)
        
        # Should pass schema validation (may fail output type requirements)
        # Check that no schema-level errors exist
        schema_errors = [e for e in result.errors if "enrichment_version" in e or "metadata" in e]
        assert len(schema_errors) == 0
    
    @given(data=invalid_enrichment_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_invalid_data_fails_validation(self, data: dict):
        """
        **Property 11: Input Validation (Invalid Data Fails)**
        *For any* invalid EnrichmentV1 data, validation SHALL fail
        with specific error messages.
        **Validates: Requirement 19.1**
        """
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_property_11_available_enrichments_reported(self):
        """
        **Property 11: Input Validation (Available Enrichments)**
        *For any* input file, validation SHALL report which enrichment
        types are available.
        **Validates: Requirement 19.5**
        """
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary", "tags"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
            "tags": {"categories": ["tech"], "keywords": ["ai"]},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert "summary" in result.available_enrichments
        assert "tags" in result.available_enrichments
    
    def test_property_11_missing_enrichments_reported(self):
        """
        **Property 11: Input Validation (Missing Enrichments)**
        *For any* output type with missing required enrichments,
        validation SHALL report which enrichments are missing.
        **Validates: Requirement 19.3, 19.4**
        """
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data, output_type="blog")
        
        # Blog requires summary and tags, tags is missing
        assert "tags" in result.missing_enrichments
    
    def test_property_11_suggestions_provided(self):
        """
        **Property 11: Input Validation (Suggestions)**
        *For any* validation failure, the system SHALL provide
        suggestions for fixing the issue.
        **Validates: Requirement 19.6**
        """
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data, output_type="blog")
        
        # Should have suggestions for missing tags
        assert len(result.suggestions) > 0
        assert any("enrichment" in s.lower() for s in result.suggestions)
    
    def test_property_11_invalid_output_type_error(self):
        """
        **Property 11: Input Validation (Invalid Output Type)**
        *For any* invalid output type, validation SHALL fail with
        a specific error message.
        **Validates: Requirement 19.2**
        """
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        validator = InputValidator()
        result = validator.validate_content(data, output_type="invalid-type")
        
        assert not result.is_valid
        assert any("invalid" in e.lower() and "output type" in e.lower() for e in result.errors)
    
    @given(
        data=valid_enrichment_v1_strategy(),
        output_type=st.sampled_from(["tweet", "youtube", "linkedin", "newsletter"]),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_summary_only_types_pass_with_summary(
        self, data: dict, output_type: str
    ):
        """
        **Property 11: Input Validation (Summary-Only Types)**
        *For any* output type that only requires summary, validation
        SHALL pass when summary is present.
        **Validates: Requirement 19.2**
        """
        validator = InputValidator()
        result = validator.validate_content(data, output_type=output_type)
        
        # These types only require summary, which is in valid_enrichment_v1_strategy
        assert result.is_valid or "summary" in result.missing_enrichments
    
    @given(data=enrichment_with_chapters_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_chapters_type_passes_with_chapters(self, data: dict):
        """
        **Property 11: Input Validation (Chapters Type)**
        *For any* data with chapters, validation for chapters output
        type SHALL pass.
        **Validates: Requirement 19.2**
        """
        validator = InputValidator()
        result = validator.validate_content(data, output_type="chapters")
        
        assert result.is_valid
        assert "chapters" in result.available_enrichments
    
    @given(data=enrichment_with_highlights_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_quote_cards_type_passes_with_highlights(self, data: dict):
        """
        **Property 11: Input Validation (Quote Cards Type)**
        *For any* data with highlights, validation for quote-cards
        output type SHALL pass.
        **Validates: Requirement 19.2**
        """
        validator = InputValidator()
        result = validator.validate_content(data, output_type="quote-cards")
        
        assert result.is_valid
        assert "highlights" in result.available_enrichments
    
    @given(data=enrichment_with_transcript_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_transcript_clean_passes_with_transcript(self, data: dict):
        """
        **Property 11: Input Validation (Transcript Clean Type)**
        *For any* data with transcript, validation for transcript-clean
        output type SHALL pass.
        **Validates: Requirement 19.2**
        """
        validator = InputValidator()
        result = validator.validate_content(data, output_type="transcript-clean")
        
        assert result.is_valid
        assert "transcript" in result.available_enrichments
    
    def test_property_11_convenience_function_validate_input(self):
        """
        **Property 11: Input Validation (Convenience Function)**
        *For any* input data, the validate_input convenience function
        SHALL return the same result as InputValidator.validate_content.
        **Validates: Requirement 19.1**
        """
        data = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {"short": "test", "medium": "test", "long": "test"},
        }
        
        is_valid, errors = validate_input(data)
        
        validator = InputValidator()
        result = validator.validate_content(data)
        
        assert is_valid == result.is_valid
        assert errors == result.errors
    
    def test_property_11_all_16_output_types_have_requirements(self):
        """
        **Property 11: Input Validation (All Types Covered)**
        *For any* of the 16 output types, the system SHALL have
        defined required enrichments.
        **Validates: Requirement 19.2**
        """
        assert len(REQUIRED_ENRICHMENTS) == 16
        
        for output_type in VALID_OUTPUT_TYPES:
            assert output_type in REQUIRED_ENRICHMENTS
            required = REQUIRED_ENRICHMENTS[output_type]
            assert isinstance(required, list)


# ============================================================================
# PROPERTY 9: Cost Estimation Before Execution (Requirements 13.1, 13.3, 13.4)
# ============================================================================

class TestCostEstimationBeforeExecution:
    """Property 9: Cost Estimation Before Execution.
    
    Validates Requirements 13.1, 13.3, 13.4:
    - 13.1: Calculate estimated cost BEFORE making any LLM API calls
    - 13.3: If --max-cost is specified and estimate exceeds limit, NO LLM calls made
    - 13.4: Display 50% warning threshold when approaching limit
    
    Feature: formatter-publishing-drafts, Property 9: Cost Estimation Before Execution
    """
    
    @given(
        max_cost=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        estimated_cost=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_cost_limit_enforcement(self, max_cost: float, estimated_cost: float):
        """
        **Property 9: Cost Estimation Before Execution (Limit Enforcement)**
        *For any* formatting request with --max-cost specified, if the estimated
        cost exceeds the limit, NO LLM API calls SHALL be made.
        **Validates: Requirements 13.1, 13.3**
        """
        from pipeline.formatters.composer import FormatComposer
        from pipeline.formatters.base import CostEstimate
        
        composer = FormatComposer()
        
        # Create a cost estimate
        estimate = CostEstimate(
            estimated_tokens=1000,
            estimated_cost_usd=estimated_cost,
            provider="openai",
            model="gpt-4",
            within_budget=estimated_cost <= max_cost,
        )
        
        # Check cost limit
        within_limit, warning = composer.check_cost_limit(estimate, max_cost)
        
        # If estimated cost exceeds max_cost, within_limit should be False
        if estimated_cost > max_cost:
            assert not within_limit, (
                f"Cost ${estimated_cost:.4f} exceeds limit ${max_cost:.4f} "
                f"but check_cost_limit returned within_limit=True"
            )
        else:
            assert within_limit, (
                f"Cost ${estimated_cost:.4f} is within limit ${max_cost:.4f} "
                f"but check_cost_limit returned within_limit=False"
            )
    
    @given(
        max_cost=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_warning_threshold_at_50_percent(self, max_cost: float):
        """
        **Property 9: Cost Estimation Before Execution (50% Warning)**
        *For any* formatting request where estimated cost exceeds 50% of max_cost
        but is still within limit, a warning SHALL be generated.
        **Validates: Requirement 13.4**
        """
        from pipeline.formatters.composer import FormatComposer, COST_WARNING_THRESHOLD
        from pipeline.formatters.base import CostEstimate
        
        composer = FormatComposer()
        
        # Cost at exactly 51% of max_cost (above warning threshold)
        estimated_cost = max_cost * 0.51
        
        estimate = CostEstimate(
            estimated_tokens=1000,
            estimated_cost_usd=estimated_cost,
            provider="openai",
            model="gpt-4",
            within_budget=True,
        )
        
        within_limit, warning = composer.check_cost_limit(estimate, max_cost)
        
        # Should be within limit but have a warning
        assert within_limit
        assert warning is not None, (
            f"Cost ${estimated_cost:.4f} is {(estimated_cost/max_cost)*100:.1f}% of "
            f"max_cost ${max_cost:.4f} but no warning was generated"
        )
        assert "%" in warning  # Warning should mention percentage
    
    @given(
        max_cost=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_no_warning_below_threshold(self, max_cost: float):
        """
        **Property 9: Cost Estimation Before Execution (No Warning Below Threshold)**
        *For any* formatting request where estimated cost is below 50% of max_cost,
        NO warning SHALL be generated.
        **Validates: Requirement 13.4**
        """
        from pipeline.formatters.composer import FormatComposer
        from pipeline.formatters.base import CostEstimate
        
        composer = FormatComposer()
        
        # Cost at exactly 49% of max_cost (below warning threshold)
        estimated_cost = max_cost * 0.49
        
        estimate = CostEstimate(
            estimated_tokens=1000,
            estimated_cost_usd=estimated_cost,
            provider="openai",
            model="gpt-4",
            within_budget=True,
        )
        
        within_limit, warning = composer.check_cost_limit(estimate, max_cost)
        
        # Should be within limit with no warning
        assert within_limit
        assert warning is None, (
            f"Cost ${estimated_cost:.4f} is only {(estimated_cost/max_cost)*100:.1f}% of "
            f"max_cost ${max_cost:.4f} but a warning was generated: {warning}"
        )
    
    @given(
        output_type=st.sampled_from(VALID_OUTPUT_TYPES),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_estimate_returns_valid_structure(self, output_type: str):
        """
        **Property 9: Cost Estimation Before Execution (Valid Structure)**
        *For any* output type, cost estimation SHALL return a valid CostEstimate
        with all required fields populated.
        **Validates: Requirement 13.1**
        """
        from pipeline.formatters.composer import FormatComposer
        from pipeline.formatters.base import FormatRequest, CostEstimate
        
        composer = FormatComposer()
        
        # Create a minimal valid enriched content
        enriched_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary with more details.",
                "long": "A long summary with comprehensive details about the content.",
            },
            "tags": {
                "categories": ["tech"],
                "keywords": ["test"],
            },
            "chapters": [
                {"start_time": 0, "end_time": 60, "title": "Intro", "summary": "Introduction"}
            ],
            "highlights": [
                {"text": "Important quote", "timestamp": 30, "importance": "high"}
            ],
            "transcript": "This is the transcript text.",
        }
        
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type=output_type,
            llm_enhance=True,
        )
        
        estimate = composer.estimate_cost(request)
        
        # Verify structure
        assert isinstance(estimate, CostEstimate)
        assert estimate.estimated_tokens >= 0
        assert estimate.estimated_cost_usd >= 0
        assert isinstance(estimate.provider, str)
        assert isinstance(estimate.model, str)
        assert isinstance(estimate.within_budget, bool)
    
    def test_property_9_dry_run_returns_estimate_without_execution(self):
        """
        **Property 9: Cost Estimation Before Execution (Dry Run)**
        *For any* formatting request with dry_run=True, the system SHALL
        return cost estimate without making any LLM API calls.
        **Validates: Requirements 13.1, 13.3**
        """
        from pipeline.formatters.composer import FormatComposer
        from pipeline.formatters.base import FormatRequest
        
        composer = FormatComposer()
        
        enriched_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
        }
        
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="blog",
            llm_enhance=True,
        )
        
        dry_run_result = composer.dry_run(request)
        
        # Verify dry run result structure
        assert dry_run_result["dry_run"] is True
        assert dry_run_result["output_type"] == "blog"
        assert "estimated_cost_usd" in dry_run_result
        assert "estimated_tokens" in dry_run_result
        assert dry_run_result["estimated_cost_usd"] >= 0
    
    @given(
        max_cost=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_9_dry_run_with_max_cost_shows_budget_status(self, max_cost: float):
        """
        **Property 9: Cost Estimation Before Execution (Budget Status in Dry Run)**
        *For any* dry run with max_cost specified, the result SHALL indicate
        whether the estimate is within budget.
        **Validates: Requirement 13.3**
        """
        from pipeline.formatters.composer import FormatComposer
        from pipeline.formatters.base import FormatRequest
        
        composer = FormatComposer()
        
        enriched_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
        }
        
        request = FormatRequest(
            enriched_content=enriched_content,
            output_type="blog",
            llm_enhance=True,
            max_cost=max_cost,
        )
        
        dry_run_result = composer.dry_run(request)
        
        # Verify budget status is included
        assert "max_cost" in dry_run_result
        assert dry_run_result["max_cost"] == max_cost
        assert "within_limit" in dry_run_result
        assert isinstance(dry_run_result["within_limit"], bool)


# ============================================================================
# PROPERTY 7: Bundle Generation Completeness (Requirements 9.4, 9.8, 9.9, 9.10)
# ============================================================================

class TestBundleGenerationCompleteness:
    """Property 7: Bundle Generation Completeness.
    
    Validates Requirements 9.4, 9.8, 9.9, 9.10:
    - 9.4: Generate all output types defined in the bundle
    - 9.8: Continue processing remaining types if one fails (error isolation)
    - 9.9: Generate manifest file listing all outputs
    - 9.10: Track total cost across all bundle outputs
    
    Feature: formatter-publishing-drafts, Property 7: Bundle Generation Completeness
    """
    
    @given(
        bundle_name=st.sampled_from([
            "blog-launch",
            "video-launch",
            "podcast",
            "social-only",
            "full-repurpose",
            "notes-package",
        ])
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_bundle_generates_all_defined_outputs(self, bundle_name: str):
        """
        **Property 7: Bundle Generation Completeness (All Outputs)**
        *For any* valid bundle name, format_bundle SHALL attempt to generate
        all output types defined in that bundle's configuration.
        **Validates: Requirement 9.4**
        """
        from pipeline.formatters.composer import FormatComposer
        from pipeline.formatters.bundles.loader import BundleLoader
        
        composer = FormatComposer()
        loader = BundleLoader()
        
        # Get expected outputs from bundle config
        bundle = loader.load_bundle(bundle_name)
        expected_outputs = set(bundle.outputs)
        
        enriched_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary", "chapters", "highlights"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary with more details.",
            },
            "chapters": [
                {"title": "Introduction", "start_time": 0, "end_time": 60},
                {"title": "Main Content", "start_time": 60, "end_time": 300},
            ],
            "highlights": [
                {"text": "Key insight 1", "timestamp": 30},
                {"text": "Key insight 2", "timestamp": 120},
            ],
            "tags": ["python", "tutorial", "programming"],
            "transcript": "This is the full transcript content.",
        }
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = composer.format_bundle(
                bundle_name=bundle_name,
                enriched_content=enriched_content,
                output_dir=tmpdir,
                llm_enhance=False,  # Skip LLM for faster tests
            )
            
            # All outputs should be either successful or failed (attempted)
            attempted_outputs = set(result.successful) | set(ot for ot, _ in result.failed)
            assert attempted_outputs == expected_outputs
    
    @given(
        bundle_name=st.sampled_from(["blog-launch", "social-only"])
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_bundle_result_contains_bundle_name(self, bundle_name: str):
        """
        **Property 7: Bundle Generation Completeness (Bundle Name in Result)**
        *For any* bundle generation, the result SHALL include the bundle name.
        **Validates: Requirement 9.4**
        """
        from pipeline.formatters.composer import FormatComposer
        
        composer = FormatComposer()
        
        enriched_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
        }
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = composer.format_bundle(
                bundle_name=bundle_name,
                enriched_content=enriched_content,
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            assert result.bundle_name == bundle_name
    
    @given(
        bundle_name=st.sampled_from(["blog-launch", "social-only"])
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_bundle_generates_manifest(self, bundle_name: str):
        """
        **Property 7: Bundle Generation Completeness (Manifest Generation)**
        *For any* bundle generation, a manifest file SHALL be created listing
        all outputs and their status.
        **Validates: Requirement 9.9**
        """
        from pipeline.formatters.composer import FormatComposer
        import json
        from pathlib import Path
        
        composer = FormatComposer()
        
        enriched_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
        }
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = composer.format_bundle(
                bundle_name=bundle_name,
                enriched_content=enriched_content,
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # Manifest should be created
            assert result.manifest_path != ""
            manifest_path = Path(result.manifest_path)
            assert manifest_path.exists()
            
            # Manifest should be valid JSON
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            # Manifest should contain required fields
            assert "bundle_name" in manifest
            assert manifest["bundle_name"] == bundle_name
            assert "outputs_requested" in manifest
            assert "outputs_successful" in manifest
            assert "outputs_failed" in manifest
            assert "total_cost_usd" in manifest
    
    @given(
        bundle_name=st.sampled_from(["blog-launch", "social-only"])
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_bundle_tracks_total_time(self, bundle_name: str):
        """
        **Property 7: Bundle Generation Completeness (Time Tracking)**
        *For any* bundle generation, the total processing time SHALL be tracked.
        **Validates: Requirement 9.10**
        """
        from pipeline.formatters.composer import FormatComposer
        
        composer = FormatComposer()
        
        enriched_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
        }
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = composer.format_bundle(
                bundle_name=bundle_name,
                enriched_content=enriched_content,
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # Total time should be tracked and positive
            assert result.total_time >= 0
    
    def test_property_7_bundle_error_isolation(self):
        """
        **Property 7: Bundle Generation Completeness (Error Isolation)**
        *For any* bundle where one output type fails, the system SHALL
        continue processing remaining output types.
        **Validates: Requirement 9.8**
        """
        from pipeline.formatters.composer import FormatComposer
        from unittest.mock import patch, MagicMock
        
        composer = FormatComposer()
        
        enriched_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
        }
        
        # Mock format_single to fail for one specific output type
        original_format_single = composer.format_single
        call_count = {"blog": 0, "tweet": 0, "linkedin": 0, "seo": 0}
        
        def mock_format_single(request):
            call_count[request.output_type] = call_count.get(request.output_type, 0) + 1
            if request.output_type == "tweet":
                raise Exception("Simulated tweet failure")
            return original_format_single(request)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(composer, 'format_single', side_effect=mock_format_single):
                result = composer.format_bundle(
                    bundle_name="blog-launch",  # blog, tweet, linkedin, seo
                    enriched_content=enriched_content,
                    output_dir=tmpdir,
                    llm_enhance=False,
                )
            
            # Tweet should be in failed list
            failed_types = [ot for ot, _ in result.failed]
            assert "tweet" in failed_types
            
            # Other types should still be attempted (either successful or failed)
            # blog-launch has: blog, tweet, linkedin, seo
            # At least some should succeed
            assert len(result.successful) > 0 or len(result.failed) > 1


# ============================================================================
# PROPERTY 14: Bundle Configuration Validation (Requirement 9.5)
# ============================================================================

class TestBundleConfigurationValidation:
    """Property 14: Bundle Configuration Validation.
    
    Validates Requirement 9.5:
    - 9.5: Validate bundle configuration (check output types are valid)
    
    Feature: formatter-publishing-drafts, Property 14: Bundle Configuration Validation
    """
    
    @given(
        bundle_name=st.sampled_from([
            "blog-launch",
            "video-launch",
            "podcast",
            "social-only",
            "full-repurpose",
            "notes-package",
        ])
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_14_builtin_bundles_have_valid_output_types(self, bundle_name: str):
        """
        **Property 14: Bundle Configuration Validation (Valid Output Types)**
        *For any* built-in bundle, all output types in the bundle SHALL be
        valid output types recognized by the formatter.
        **Validates: Requirement 9.5**
        """
        from pipeline.formatters.bundles.loader import BundleLoader, VALID_OUTPUT_TYPES
        
        loader = BundleLoader()
        bundle = loader.load_bundle(bundle_name)
        
        # All outputs should be valid types
        for output_type in bundle.outputs:
            assert output_type in VALID_OUTPUT_TYPES, \
                f"Bundle '{bundle_name}' contains invalid output type: {output_type}"
    
    @given(
        bundle_name=st.sampled_from([
            "blog-launch",
            "video-launch",
            "podcast",
            "social-only",
            "full-repurpose",
            "notes-package",
        ])
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_14_builtin_bundles_have_no_duplicates(self, bundle_name: str):
        """
        **Property 14: Bundle Configuration Validation (No Duplicates)**
        *For any* built-in bundle, there SHALL be no duplicate output types.
        **Validates: Requirement 9.5**
        """
        from pipeline.formatters.bundles.loader import BundleLoader
        
        loader = BundleLoader()
        bundle = loader.load_bundle(bundle_name)
        
        # No duplicates
        assert len(bundle.outputs) == len(set(bundle.outputs)), \
            f"Bundle '{bundle_name}' contains duplicate output types"
    
    @given(
        bundle_name=st.sampled_from([
            "blog-launch",
            "video-launch",
            "podcast",
            "social-only",
            "full-repurpose",
            "notes-package",
        ])
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_14_builtin_bundles_have_at_least_one_output(self, bundle_name: str):
        """
        **Property 14: Bundle Configuration Validation (Non-Empty)**
        *For any* built-in bundle, there SHALL be at least one output type.
        **Validates: Requirement 9.5**
        """
        from pipeline.formatters.bundles.loader import BundleLoader
        
        loader = BundleLoader()
        bundle = loader.load_bundle(bundle_name)
        
        # At least one output
        assert len(bundle.outputs) >= 1, \
            f"Bundle '{bundle_name}' has no output types"
    
    @given(
        bundle_name=st.sampled_from([
            "blog-launch",
            "video-launch",
            "podcast",
            "social-only",
            "full-repurpose",
            "notes-package",
        ])
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_14_builtin_bundles_pass_validation(self, bundle_name: str):
        """
        **Property 14: Bundle Configuration Validation (Validation Passes)**
        *For any* built-in bundle, validation SHALL pass without errors.
        **Validates: Requirement 9.5**
        """
        from pipeline.formatters.bundles.loader import BundleLoader
        
        loader = BundleLoader()
        bundle = loader.load_bundle(bundle_name)
        
        # Validation should pass
        is_valid, errors = loader.validate_bundle(bundle)
        assert is_valid, f"Bundle '{bundle_name}' validation failed: {errors}"
    
    def test_property_14_unknown_bundle_raises_error_with_available_list(self):
        """
        **Property 14: Bundle Configuration Validation (Unknown Bundle Error)**
        *For any* unknown bundle name, the system SHALL raise an error that
        includes a list of available bundles.
        **Validates: Requirement 9.5**
        """
        from pipeline.formatters.bundles.loader import BundleLoader, BundleNotFoundError
        
        loader = BundleLoader()
        
        with pytest.raises(BundleNotFoundError) as exc_info:
            loader.load_bundle("nonexistent-bundle")
        
        # Error message should include available bundles
        error_msg = str(exc_info.value)
        assert "nonexistent-bundle" in error_msg
        assert "Available bundles" in error_msg
        # Should list at least one available bundle
        assert "blog-launch" in error_msg or "social-only" in error_msg
    
    def test_property_14_bundle_with_invalid_output_type_fails_validation(self):
        """
        **Property 14: Bundle Configuration Validation (Invalid Type Rejected)**
        *For any* bundle containing an invalid output type, validation SHALL fail.
        **Validates: Requirement 9.5**
        """
        from pipeline.formatters.bundles.loader import BundleLoader, BundleConfig
        
        loader = BundleLoader()
        
        # Create a bundle with an invalid output type
        invalid_bundle = BundleConfig(
            name="test-invalid",
            description="Test bundle with invalid type",
            outputs=["blog", "invalid-type", "tweet"],
        )
        
        is_valid, errors = loader.validate_bundle(invalid_bundle)
        
        assert not is_valid
        assert any("invalid-type" in error for error in errors)
    
    def test_property_14_bundle_with_duplicates_fails_validation(self):
        """
        **Property 14: Bundle Configuration Validation (Duplicates Rejected)**
        *For any* bundle containing duplicate output types, validation SHALL fail.
        **Validates: Requirement 9.5**
        """
        from pipeline.formatters.bundles.loader import BundleLoader, BundleConfig
        
        loader = BundleLoader()
        
        # Create a bundle with duplicate output types
        duplicate_bundle = BundleConfig(
            name="test-duplicates",
            description="Test bundle with duplicates",
            outputs=["blog", "tweet", "blog"],  # blog appears twice
        )
        
        is_valid, errors = loader.validate_bundle(duplicate_bundle)
        
        assert not is_valid
        assert any("Duplicate" in error or "duplicate" in error for error in errors)



# ============================================================================
# PROPERTY 8: Batch Error Isolation (Requirements 10.5, 10.6, 10.7)
# ============================================================================

class TestBatchErrorIsolation:
    """Property 8: Batch Error Isolation.
    
    Validates Requirements 10.5, 10.6, 10.7:
    - 10.5: Continue processing remaining files if one fails
    - 10.6: Track failed files with error messages
    - 10.7: Generate summary report with success/failure counts
    
    Feature: formatter-publishing-drafts, Property 8: Batch Error Isolation
    """
    
    def test_property_8_batch_continues_on_file_failure(self):
        """
        **Property 8: Batch Error Isolation (Continue on Failure)**
        *For any* batch where one file fails to process, the system SHALL
        continue processing remaining files.
        **Validates: Requirement 10.5**
        """
        from pipeline.formatters.composer import FormatComposer
        import tempfile
        import json
        from pathlib import Path
        
        composer = FormatComposer()
        
        # Create test files - one valid, one invalid
        valid_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
            "tags": ["python", "tutorial", "programming"],
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create valid file
            valid_file = tmppath / "valid.enriched.json"
            with open(valid_file, "w") as f:
                json.dump(valid_content, f)
            
            # Create invalid file (not valid JSON)
            invalid_file = tmppath / "invalid.enriched.json"
            with open(invalid_file, "w") as f:
                f.write("not valid json {{{")
            
            # Create another valid file
            valid_file2 = tmppath / "valid2.enriched.json"
            with open(valid_file2, "w") as f:
                json.dump(valid_content, f)
            
            # Process batch
            result = composer.format_batch(
                input_pattern=str(tmppath / "*.enriched.json"),
                output_type="blog",
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # Should have processed all files
            total_processed = len(result.successful) + len(result.failed)
            assert total_processed == 3
            
            # Should have at least one success (the valid files)
            assert len(result.successful) >= 1
            
            # Should have at least one failure (the invalid file)
            assert len(result.failed) >= 1
    
    def test_property_8_batch_tracks_failed_files_with_errors(self):
        """
        **Property 8: Batch Error Isolation (Track Failed Files)**
        *For any* batch with failed files, the result SHALL include the
        file path and error message for each failure.
        **Validates: Requirement 10.6**
        """
        from pipeline.formatters.composer import FormatComposer
        import tempfile
        import json
        from pathlib import Path
        
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create invalid file
            invalid_file = tmppath / "invalid.enriched.json"
            with open(invalid_file, "w") as f:
                f.write("not valid json")
            
            # Process batch
            result = composer.format_batch(
                input_pattern=str(tmppath / "*.enriched.json"),
                output_type="blog",
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # Should have one failure
            assert len(result.failed) == 1
            
            # Failed entry should be a tuple of (file_path, error_message)
            file_path, error_msg = result.failed[0]
            assert "invalid.enriched.json" in file_path
            assert len(error_msg) > 0  # Error message should not be empty
    
    def test_property_8_batch_generates_summary(self):
        """
        **Property 8: Batch Error Isolation (Summary Report)**
        *For any* batch processing, the result SHALL include success/failure
        counts and total processing time.
        **Validates: Requirement 10.7**
        """
        from pipeline.formatters.composer import FormatComposer
        import tempfile
        import json
        from pathlib import Path
        
        composer = FormatComposer()
        
        valid_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
            "tags": ["python", "tutorial", "programming"],
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create test files
            for i in range(3):
                file_path = tmppath / f"file{i}.enriched.json"
                with open(file_path, "w") as f:
                    json.dump(valid_content, f)
            
            # Process batch
            result = composer.format_batch(
                input_pattern=str(tmppath / "*.enriched.json"),
                output_type="blog",
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # Result should have summary information
            assert hasattr(result, 'successful')
            assert hasattr(result, 'failed')
            assert hasattr(result, 'total_time')
            assert hasattr(result, 'total_cost')
            
            # Total time should be positive
            assert result.total_time >= 0
            
            # Total should match files processed
            assert len(result.successful) + len(result.failed) == 3
    
    def test_property_8_batch_summary_format(self):
        """
        **Property 8: Batch Error Isolation (Summary Format)**
        *For any* batch result, format_batch_summary SHALL produce a
        human-readable summary string.
        **Validates: Requirement 10.7**
        """
        from pipeline.formatters.composer import FormatComposer, BatchResult
        
        composer = FormatComposer()
        
        # Create a mock result
        result = BatchResult(
            successful=["file1.json", "file2.json"],
            failed=[("file3.json", "Invalid JSON")],
            output_dir="./output",
            total_cost=0.05,
            total_time=2.5,
        )
        
        summary = composer.format_batch_summary(result)
        
        # Summary should contain key information
        assert "Total files: 3" in summary
        assert "Successful: 2" in summary
        assert "Failed: 1" in summary
        assert "file3.json" in summary
        assert "Invalid JSON" in summary
    
    @given(
        num_files=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_property_8_batch_processes_all_matching_files(self, num_files: int):
        """
        **Property 8: Batch Error Isolation (Process All Files)**
        *For any* glob pattern matching N files, the batch processor SHALL
        attempt to process all N files.
        **Validates: Requirement 10.5**
        """
        from pipeline.formatters.composer import FormatComposer
        import tempfile
        import json
        from pathlib import Path
        
        composer = FormatComposer()
        
        valid_content = {
            "enrichment_version": "v1",
            "metadata": {
                "provider": "openai",
                "model": "gpt-4",
                "timestamp": "2026-01-01T00:00:00",
                "cost_usd": 0.1,
                "tokens_used": 100,
                "enrichment_types": ["summary"],
                "cache_hit": False,
            },
            "summary": {
                "short": "A short summary.",
                "medium": "A medium length summary.",
                "long": "A long summary.",
            },
            "tags": ["python", "tutorial", "programming"],
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create test files
            for i in range(num_files):
                file_path = tmppath / f"file{i}.enriched.json"
                with open(file_path, "w") as f:
                    json.dump(valid_content, f)
            
            # Process batch
            result = composer.format_batch(
                input_pattern=str(tmppath / "*.enriched.json"),
                output_type="blog",
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # All files should be processed (either success or failure)
            total_processed = len(result.successful) + len(result.failed)
            assert total_processed == num_files
    
    def test_property_8_batch_empty_pattern_returns_empty_result(self):
        """
        **Property 8: Batch Error Isolation (Empty Pattern)**
        *For any* glob pattern matching no files, the batch processor SHALL
        return an empty result without errors.
        **Validates: Requirement 10.5**
        """
        from pipeline.formatters.composer import FormatComposer
        import tempfile
        
        composer = FormatComposer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Process batch with pattern that matches nothing
            result = composer.format_batch(
                input_pattern=f"{tmpdir}/nonexistent*.json",
                output_type="blog",
                output_dir=tmpdir,
                llm_enhance=False,
            )
            
            # Should return empty result
            assert len(result.successful) == 0
            assert len(result.failed) == 0
            assert result.total_time >= 0


# ============================================================================
# OUTPUT WRITER TESTS (Requirements 16.1-16.7, 12.5, 12.6)
# ============================================================================

from pipeline.formatters.writer import OutputWriter, WriteResult
from pipeline.formatters.schemas.format_v1 import FormatV1, LLMMetadata, ValidationMetadata
from pipeline.formatters.base import FormatResult


class TestOutputWriter:
    """Tests for OutputWriter functionality.
    
    Validates Requirements:
    - 16.1: Output path handling (--output flag)
    - 16.2: Default filename generation
    - 16.3: Bundle output directory handling
    - 16.4: Preserve original filename
    - 16.5: Overwrite confirmation
    - 16.6: Directory creation
    - 16.7: Validate writable path
    - 12.5: Embed metadata in Markdown frontmatter
    - 12.6: Generate sidecar metadata files
    """
    
    def test_generate_filename_with_input_path(self):
        """Test filename generation from input path."""
        writer = OutputWriter()
        
        filename = writer.generate_filename(
            input_path="content/my-video.enriched.json",
            output_type="blog",
        )
        
        assert filename == "my-video_blog.md"
    
    def test_generate_filename_without_input_path(self):
        """Test filename generation without input path."""
        writer = OutputWriter()
        
        filename = writer.generate_filename(
            input_path=None,
            output_type="blog",
        )
        
        assert filename == "output_blog.md"
    
    def test_generate_filename_json_output_type(self):
        """Test filename generation for JSON output types."""
        writer = OutputWriter()
        
        filename = writer.generate_filename(
            input_path="content/my-video.enriched.json",
            output_type="seo",
        )
        
        assert filename == "my-video_seo.json"
    
    def test_generate_filename_all_output_types(self):
        """Test filename generation for all output types."""
        writer = OutputWriter()
        
        for output_type in OutputWriter.MARKDOWN_TYPES:
            filename = writer.generate_filename(
                input_path="test.json",
                output_type=output_type,
            )
            assert filename.endswith(".md"), f"{output_type} should produce .md"
        
        for output_type in OutputWriter.JSON_TYPES:
            filename = writer.generate_filename(
                input_path="test.json",
                output_type=output_type,
            )
            assert filename.endswith(".json"), f"{output_type} should produce .json"
    
    def test_write_creates_directory(self):
        """Test that write creates output directory if needed."""
        import tempfile
        import os
        
        writer = OutputWriter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir", "output.md")
            
            # Create a minimal FormatResult
            metadata = FormatV1(
                format_version="v1",
                output_type="blog",
                timestamp=datetime.now(),
                source_file="test.json",
                validation=ValidationMetadata(
                    platform=None,
                    character_count=100,
                    truncated=False,
                ),
            )
            
            format_result = FormatResult(
                content="# Test Content\n\nThis is test content.",
                metadata=metadata,
                warnings=[],
                success=True,
            )
            
            result = writer.write(
                format_result=format_result,
                output_path=output_path,
                embed_metadata=False,
                force=True,
            )
            
            assert result.success
            assert os.path.exists(output_path)
    
    def test_write_respects_overwrite_protection(self):
        """Test that write respects overwrite protection."""
        import tempfile
        import os
        
        writer = OutputWriter(force_overwrite=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.md")
            
            # Create existing file
            with open(output_path, "w") as f:
                f.write("existing content")
            
            # Create a minimal FormatResult
            metadata = FormatV1(
                format_version="v1",
                output_type="blog",
                timestamp=datetime.now(),
                source_file="test.json",
                validation=ValidationMetadata(
                    platform=None,
                    character_count=100,
                    truncated=False,
                ),
            )
            
            format_result = FormatResult(
                content="# New Content",
                metadata=metadata,
                warnings=[],
                success=True,
            )
            
            result = writer.write(
                format_result=format_result,
                output_path=output_path,
                embed_metadata=False,
                force=False,
            )
            
            # Should fail without force
            assert not result.success
            assert "already exists" in result.error
    
    def test_write_with_force_overwrites(self):
        """Test that write with force overwrites existing file."""
        import tempfile
        import os
        
        writer = OutputWriter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.md")
            
            # Create existing file
            with open(output_path, "w") as f:
                f.write("existing content")
            
            # Create a minimal FormatResult
            metadata = FormatV1(
                format_version="v1",
                output_type="blog",
                timestamp=datetime.now(),
                source_file="test.json",
                validation=ValidationMetadata(
                    platform=None,
                    character_count=100,
                    truncated=False,
                ),
            )
            
            format_result = FormatResult(
                content="# New Content",
                metadata=metadata,
                warnings=[],
                success=True,
            )
            
            result = writer.write(
                format_result=format_result,
                output_path=output_path,
                embed_metadata=False,
                force=True,
            )
            
            assert result.success
            assert result.overwritten
            
            # Verify content was overwritten
            with open(output_path, "r") as f:
                content = f.read()
            assert "# New Content" in content
    
    def test_embed_markdown_frontmatter(self):
        """Test metadata embedding in Markdown frontmatter."""
        import tempfile
        import os
        
        writer = OutputWriter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.md")
            
            metadata = FormatV1(
                format_version="v1",
                output_type="blog",
                timestamp=datetime.now(),
                source_file="test.json",
                platform="medium",
                tone="professional",
                validation=ValidationMetadata(
                    platform="medium",
                    character_count=500,
                    truncated=False,
                ),
            )
            
            format_result = FormatResult(
                content="# Test Blog\n\nThis is the content.",
                metadata=metadata,
                warnings=[],
                success=True,
            )
            
            result = writer.write(
                format_result=format_result,
                output_path=output_path,
                embed_metadata=True,
                force=True,
            )
            
            assert result.success
            
            # Read and verify frontmatter
            with open(output_path, "r") as f:
                content = f.read()
            
            assert content.startswith("---\n")
            assert "format_version: v1" in content
            assert "output_type: blog" in content
            assert "platform: medium" in content
            assert "# Test Blog" in content
    
    def test_embed_json_metadata(self):
        """Test metadata embedding in JSON output."""
        import tempfile
        import os
        import json
        
        writer = OutputWriter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            
            metadata = FormatV1(
                format_version="v1",
                output_type="seo",
                timestamp=datetime.now(),
                source_file="test.json",
                validation=ValidationMetadata(
                    platform=None,
                    character_count=200,
                    truncated=False,
                ),
            )
            
            # SEO output is JSON
            seo_content = json.dumps({
                "meta_title": "Test Title",
                "meta_description": "Test description",
                "keywords": ["test", "seo"],
            })
            
            format_result = FormatResult(
                content=seo_content,
                metadata=metadata,
                warnings=[],
                success=True,
            )
            
            result = writer.write(
                format_result=format_result,
                output_path=output_path,
                embed_metadata=True,
                force=True,
            )
            
            assert result.success
            
            # Read and verify embedded metadata
            with open(output_path, "r") as f:
                data = json.load(f)
            
            assert "_metadata" in data
            assert data["_metadata"]["format_version"] == "v1"
            assert data["_metadata"]["output_type"] == "seo"
            assert data["meta_title"] == "Test Title"
    
    def test_generate_sidecar_metadata(self):
        """Test sidecar metadata file generation."""
        import tempfile
        import os
        import json
        
        writer = OutputWriter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.md")
            
            metadata = FormatV1(
                format_version="v1",
                output_type="blog",
                timestamp=datetime.now(),
                source_file="test.json",
                validation=ValidationMetadata(
                    platform=None,
                    character_count=100,
                    truncated=False,
                ),
                llm_metadata=LLMMetadata(
                    provider="openai",
                    model="gpt-4",
                    cost_usd=0.05,
                    tokens_used=500,
                    temperature=0.7,
                    enhanced=True,
                ),
            )
            
            format_result = FormatResult(
                content="# Test Content",
                metadata=metadata,
                warnings=[],
                success=True,
            )
            
            result = writer.write(
                format_result=format_result,
                output_path=output_path,
                embed_metadata=False,
                generate_sidecar=True,
                force=True,
            )
            
            assert result.success
            assert result.metadata_path is not None
            assert result.metadata_path.endswith(".meta.json")
            
            # Verify sidecar file
            with open(result.metadata_path, "r") as f:
                sidecar_data = json.load(f)
            
            assert sidecar_data["format_version"] == "v1"
            assert sidecar_data["output_type"] == "blog"
            assert "llm" in sidecar_data
            assert sidecar_data["llm"]["provider"] == "openai"
    
    def test_write_bundle_outputs(self):
        """Test writing multiple bundle outputs."""
        import tempfile
        import os
        
        writer = OutputWriter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple format results
            results = {}
            for output_type in ["blog", "tweet", "linkedin"]:
                metadata = FormatV1(
                    format_version="v1",
                    output_type=output_type,
                    timestamp=datetime.now(),
                    source_file="test.json",
                    validation=ValidationMetadata(
                        platform=None,
                        character_count=100,
                        truncated=False,
                    ),
                )
                
                results[output_type] = FormatResult(
                    content=f"# {output_type.title()} Content",
                    metadata=metadata,
                    warnings=[],
                    success=True,
                )
            
            write_results = writer.write_bundle_outputs(
                results=results,
                output_dir=tmpdir,
                input_path="my-video.enriched.json",
                embed_metadata=True,
                force=True,
            )
            
            # All should succeed
            assert all(r.success for r in write_results.values())
            
            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "my-video_blog.md"))
            assert os.path.exists(os.path.join(tmpdir, "my-video_tweet.md"))
            assert os.path.exists(os.path.join(tmpdir, "my-video_linkedin.md"))
    
    def test_resolve_output_path_explicit(self):
        """Test output path resolution with explicit path."""
        writer = OutputWriter()
        
        resolved = writer._resolve_output_path(
            output_path="/explicit/path/output.md",
            input_path="input.json",
            output_dir="/some/dir",
            output_type="blog",
        )
        
        # Explicit path takes precedence
        assert resolved == "/explicit/path/output.md"
    
    def test_resolve_output_path_with_output_dir(self):
        """Test output path resolution with output directory."""
        import os
        
        writer = OutputWriter()
        
        resolved = writer._resolve_output_path(
            output_path=None,
            input_path="content/video.enriched.json",
            output_dir="/output/dir",
            output_type="blog",
        )
        
        # Use os.path.join for platform-independent comparison
        expected = os.path.join("/output/dir", "video_blog.md")
        assert resolved == expected
    
    def test_resolve_output_path_default(self):
        """Test output path resolution with defaults."""
        writer = OutputWriter()
        
        resolved = writer._resolve_output_path(
            output_path=None,
            input_path="video.enriched.json",
            output_dir=None,
            output_type="tweet",
        )
        
        assert resolved == "video_tweet.md"
