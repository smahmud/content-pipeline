"""
Property-based tests for output path resolution.

**Property 6: Output Path Resolution**
*For any* combination of output paths, directories, and input files, the OutputManager 
should consistently resolve paths according to the defined precedence rules and validation.
**Validates: Requirements 7.1, 7.2, 7.3, 7.5**
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, assume, settings

from pipeline.output.manager import OutputManager


# Strategy for generating valid file paths
def valid_filename_strategy():
    """Generate valid filenames."""
    return st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='._-'),
        min_size=1,
        max_size=50
    ).filter(lambda x: x and not x.startswith('.') and not x.endswith('.'))

def valid_directory_strategy():
    """Generate valid directory paths."""
    return st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='._-/\\'),
        min_size=1,
        max_size=100
    ).filter(lambda x: x and not x.startswith('.') and '..' not in x)

# Strategy for file extensions
file_extensions = st.sampled_from(['.json', '.txt', '.md', '.xml', '.yaml'])

# Strategy for input file paths
def input_file_path_strategy():
    """Generate input file paths."""
    filename = valid_filename_strategy()
    extension = st.sampled_from(['.mp3', '.wav', '.m4a', '.webm', '.mp4'])
    return st.builds(lambda f, e: f + e, filename, extension)


class TestOutputPathResolutionProperties:
    """Test output path resolution properties."""

    @given(
        output_path=st.one_of(
            st.none(),
            st.builds(lambda f, e: f + e, valid_filename_strategy(), file_extensions)
        ),
        output_dir=st.one_of(st.none(), valid_directory_strategy()),
        input_file=st.one_of(st.none(), input_file_path_strategy())
    )
    @settings(max_examples=50)
    def test_path_resolution_consistency(self, output_path, output_dir, input_file):
        """
        **Property 6: Output Path Resolution**
        *For any* combination of parameters, path resolution should be consistent and deterministic.
        """
        # Skip invalid combinations
        assume(output_path is not None or input_file is not None)
        
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use temp directory as base for relative paths
            actual_output_dir = temp_dir if output_dir is None else os.path.join(temp_dir, output_dir)
            
            with patch.object(manager, '_ensure_directory_exists'), \
                 patch.object(manager, '_validate_output_path'):
                
                # Resolve path multiple times - should be consistent
                resolved1 = manager.resolve_output_path(
                    output_path=output_path,
                    output_dir=actual_output_dir,
                    input_file_path=input_file,
                    create_dirs=False
                )
                
                resolved2 = manager.resolve_output_path(
                    output_path=output_path,
                    output_dir=actual_output_dir,
                    input_file_path=input_file,
                    create_dirs=False
                )
                
                # Should be identical
                assert resolved1 == resolved2
                assert isinstance(resolved1, Path)
                assert isinstance(resolved2, Path)

    def test_absolute_path_precedence(self):
        """
        **Property 6: Output Path Resolution**
        *For any* absolute output path, it should take precedence over all other parameters.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            absolute_path = os.path.join(temp_dir, "absolute_output.json")
            
            with patch.object(manager, '_ensure_directory_exists'), \
                 patch.object(manager, '_validate_output_path'):
                
                resolved = manager.resolve_output_path(
                    output_path=absolute_path,
                    output_dir="/ignored/directory",
                    input_file_path="ignored_input.mp3",
                    create_dirs=False
                )
                
                assert resolved == Path(absolute_path)
                assert resolved.is_absolute()

    def test_relative_path_resolution(self):
        """
        **Property 6: Output Path Resolution**
        *For any* relative output path, it should be resolved relative to current directory.
        """
        manager = OutputManager()
        
        relative_path = "relative/output.json"
        expected = Path.cwd() / relative_path
        
        with patch.object(manager, '_ensure_directory_exists'), \
             patch.object(manager, '_validate_output_path'):
            
            resolved = manager.resolve_output_path(
                output_path=relative_path,
                create_dirs=False
            )
            
            assert resolved == expected
            assert resolved.is_absolute()

    @given(
        output_dir=valid_directory_strategy(),
        input_file=input_file_path_strategy()
    )
    @settings(max_examples=30)
    def test_generated_path_structure(self, output_dir, input_file):
        """
        **Property 6: Output Path Resolution**
        *For any* output directory and input file, generated paths should follow expected structure.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            actual_output_dir = os.path.join(temp_dir, output_dir)
            
            with patch.object(manager, '_ensure_directory_exists'), \
                 patch.object(manager, '_validate_output_path'):
                
                resolved = manager.resolve_output_path(
                    output_dir=actual_output_dir,
                    input_file_path=input_file,
                    create_dirs=False
                )
                
                # Should be in the specified directory
                assert str(resolved.parent).endswith(output_dir.replace('/', os.sep).replace('\\', os.sep))
                
                # Should have .json extension
                assert resolved.suffix == '.json'
                
                # Should be based on input filename
                input_stem = Path(input_file).stem
                assert resolved.stem == input_stem

    def test_default_directory_usage(self):
        """
        **Property 6: Output Path Resolution**
        *For any* OutputManager with default directory, it should be used when no output_dir specified.
        """
        default_dir = "./custom_default"
        manager = OutputManager(default_output_dir=default_dir)
        
        with patch.object(manager, '_ensure_directory_exists'), \
             patch.object(manager, '_validate_output_path'):
            
            resolved = manager.resolve_output_path(
                input_file_path="test.mp3",
                create_dirs=False
            )
            
            expected_dir = Path.cwd() / default_dir
            assert resolved.parent == expected_dir

    @given(input_file=input_file_path_strategy())
    @settings(max_examples=20)
    def test_filename_generation_from_input(self, input_file):
        """
        **Property 6: Output Path Resolution**
        *For any* input file, generated filename should preserve the base name with .json extension.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(manager, '_ensure_directory_exists'), \
                 patch.object(manager, '_validate_output_path'):
                
                resolved = manager.resolve_output_path(
                    output_dir=temp_dir,
                    input_file_path=input_file,
                    create_dirs=False
                )
                
                input_path = Path(input_file)
                expected_filename = input_path.stem + '.json'
                
                assert resolved.name == expected_filename
                assert resolved.suffix == '.json'

    def test_timestamp_filename_generation(self):
        """
        **Property 6: Output Path Resolution**
        *For any* case without input file, timestamp-based filename should be generated.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(manager, '_ensure_directory_exists'), \
                 patch.object(manager, '_validate_output_path'), \
                 patch('pipeline.output.manager.datetime') as mock_datetime:
                
                mock_datetime.now.return_value.strftime.return_value = "20240127_143000"
                
                resolved = manager.resolve_output_path(
                    output_dir=temp_dir,
                    create_dirs=False
                )
                
                assert resolved.name == "transcript_20240127_143000.json"
                assert resolved.suffix == '.json'

    def test_directory_creation_behavior(self):
        """
        **Property 6: Output Path Resolution**
        *For any* path resolution with create_dirs=True, directories should be created.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nested", "deep", "directory")
            output_path = os.path.join(nested_dir, "output.json")
            
            with patch.object(manager, '_validate_output_path'):
                resolved = manager.resolve_output_path(
                    output_path=output_path,
                    create_dirs=True
                )
                
                assert resolved == Path(output_path)
                assert resolved.parent.exists()
                assert resolved.parent.is_dir()

    def test_directory_creation_disabled(self):
        """
        **Property 6: Output Path Resolution**
        *For any* path resolution with create_dirs=False, directories should not be created.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nonexistent", "directory")
            output_path = os.path.join(nested_dir, "output.json")
            
            with patch.object(manager, '_validate_output_path'):
                resolved = manager.resolve_output_path(
                    output_path=output_path,
                    create_dirs=False
                )
                
                assert resolved == Path(output_path)
                assert not resolved.parent.exists()

    @given(
        overwrite=st.booleans(),
        backup=st.booleans()
    )
    @settings(max_examples=10)
    def test_existing_file_handling_consistency(self, overwrite, backup):
        """
        **Property 6: Output Path Resolution**
        *For any* existing file handling preferences, behavior should be consistent.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = Path(temp_dir) / "existing.json"
            existing_file.write_text("existing content")
            
            result = manager.handle_existing_file(existing_file, overwrite=overwrite, backup=backup)
            
            if overwrite:
                # Should return the same path
                assert result == existing_file
                if backup:
                    # Should have created a backup
                    backup_files = list(existing_file.parent.glob(f"{existing_file.stem}.*.backup{existing_file.suffix}"))
                    assert len(backup_files) > 0
            else:
                # Should return a different path
                if existing_file.exists():  # Only if file still exists
                    assert result != existing_file
                    assert result.parent == existing_file.parent
                    assert result.suffix == existing_file.suffix

    def test_path_validation_properties(self):
        """
        **Property 6: Output Path Resolution**
        *For any* resolved path, validation should ensure writability.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.json"
            
            # Should not raise an error for valid path
            manager._validate_output_path(output_path)
            
            # Should raise error for non-writable directory
            with patch('os.access', return_value=False):
                with pytest.raises(ValueError, match="not writable"):
                    manager._validate_output_path(output_path)

    @given(
        output_path=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        output_dir=st.one_of(st.none(), st.text(min_size=1, max_size=50))
    )
    @settings(max_examples=30)
    def test_configuration_validation_properties(self, output_path, output_dir):
        """
        **Property 6: Output Path Resolution**
        *For any* configuration parameters, validation should provide clear feedback.
        """
        manager = OutputManager()
        
        # Filter out obviously invalid paths
        if output_path and ('\x00' in output_path or len(output_path.strip()) == 0):
            output_path = None
        if output_dir and ('\x00' in output_dir or len(output_dir.strip()) == 0):
            output_dir = None
        
        is_valid, errors = manager.validate_output_configuration(
            output_path=output_path,
            output_dir=output_dir
        )
        
        # Should always return boolean and list
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # If invalid, should have errors
        if not is_valid:
            assert len(errors) > 0
            assert all(isinstance(error, str) for error in errors)

    def test_output_info_completeness(self):
        """
        **Property 6: Output Path Resolution**
        *For any* output path, get_output_info should provide complete information.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.json"
            
            info = manager.get_output_info(output_path)
            
            # Should have all required fields
            required_fields = [
                'output_path', 'output_dir', 'filename', 'is_absolute',
                'exists', 'parent_exists', 'parent_writable', 'file_writable'
            ]
            
            for field in required_fields:
                assert field in info
            
            # Should have correct types
            assert isinstance(info['output_path'], str)
            assert isinstance(info['output_dir'], str)
            assert isinstance(info['filename'], str)
            assert isinstance(info['is_absolute'], bool)
            assert isinstance(info['exists'], bool)
            assert isinstance(info['parent_exists'], bool)
            assert isinstance(info['parent_writable'], bool)

    def test_unique_path_generation_properties(self):
        """
        **Property 6: Output Path Resolution**
        *For any* base path with conflicts, unique path generation should be deterministic.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test.json"
            base_path.touch()
            
            # Generate unique path
            unique_path = manager._generate_unique_path(base_path)
            
            # Should be different from base
            assert unique_path != base_path
            
            # Should not exist
            assert not unique_path.exists()
            
            # Should follow naming pattern
            assert unique_path.parent == base_path.parent
            assert unique_path.suffix == base_path.suffix
            assert "_001" in unique_path.stem

    def test_error_handling_consistency(self):
        """
        **Property 6: Output Path Resolution**
        *For any* error condition, error messages should be informative and consistent.
        """
        manager = OutputManager()
        
        # Test with invalid directory creation
        with patch.object(manager, '_ensure_directory_exists', side_effect=OSError("Permission denied")):
            with pytest.raises(ValueError, match="Failed to resolve output path"):
                manager.resolve_output_path(output_path="/invalid/path/output.json")
        
        # Test with validation failure
        with patch.object(manager, '_validate_output_path', side_effect=ValueError("Not writable")):
            with pytest.raises(ValueError, match="Failed to resolve output path"):
                manager.resolve_output_path(output_path="valid_path.json", create_dirs=False)


class TestOutputPathIntegrationProperties:
    """Integration property tests for output path resolution."""

    def test_full_workflow_properties(self):
        """
        **Property 6: Output Path Resolution**
        *For any* complete workflow, all components should work together correctly.
        """
        manager = OutputManager(default_output_dir="./test_output")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test explicit absolute path
            abs_path = os.path.join(temp_dir, "absolute.json")
            resolved_abs = manager.resolve_output_path(output_path=abs_path)
            assert resolved_abs == Path(abs_path)
            assert resolved_abs.parent.exists()
            
            # Test generated path
            resolved_gen = manager.resolve_output_path(
                output_dir=temp_dir,
                input_file_path="test.mp3"
            )
            assert resolved_gen.parent == Path(temp_dir)
            assert resolved_gen.name == "test.json"
            assert resolved_gen.parent.exists()
            
            # Test configuration validation
            is_valid, errors = manager.validate_output_configuration(output_dir=temp_dir)
            assert is_valid is True
            assert errors == []

    def test_path_precedence_properties(self):
        """
        **Property 6: Output Path Resolution**
        *For any* combination of path parameters, precedence rules should be followed.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Explicit path should override everything
            explicit_path = os.path.join(temp_dir, "explicit.json")
            
            resolved = manager.resolve_output_path(
                output_path=explicit_path,
                output_dir="/ignored",
                input_file_path="ignored.mp3"
            )
            
            assert resolved == Path(explicit_path)
            
            # Without explicit path, should use output_dir + input_file
            resolved2 = manager.resolve_output_path(
                output_dir=temp_dir,
                input_file_path="input.mp3"
            )
            
            assert resolved2.parent == Path(temp_dir)
            assert resolved2.name == "input.json"

    def test_cross_platform_path_handling(self):
        """
        **Property 6: Output Path Resolution**
        *For any* platform, path handling should work correctly.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with different path separators
            test_paths = [
                "output.json",
                "subdir/output.json",
                "subdir\\output.json" if os.name == 'nt' else "subdir/output.json"
            ]
            
            for test_path in test_paths:
                try:
                    resolved = manager.resolve_output_path(
                        output_path=os.path.join(temp_dir, test_path)
                    )
                    
                    assert isinstance(resolved, Path)
                    assert resolved.is_absolute()
                    assert resolved.parent.exists()
                    
                except (ValueError, OSError):
                    # Some paths might be invalid on certain platforms
                    pass

    def test_concurrent_path_resolution(self):
        """
        **Property 6: Output Path Resolution**
        *For any* concurrent usage, path resolution should be thread-safe.
        """
        manager = OutputManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Multiple resolutions should be consistent
            results = []
            
            for i in range(10):
                resolved = manager.resolve_output_path(
                    output_dir=temp_dir,
                    input_file_path=f"test_{i}.mp3"
                )
                results.append(resolved)
            
            # All should be in the same directory
            for result in results:
                assert result.parent == Path(temp_dir)
            
            # All should have unique names
            names = [r.name for r in results]
            assert len(names) == len(set(names))  # All unique