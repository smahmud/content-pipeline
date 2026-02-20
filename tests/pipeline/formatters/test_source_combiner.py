"""
Unit tests for SourceCombiner.

Tests file loading, content merging, and deduplication.
"""

import json
import tempfile
from pathlib import Path

import pytest

from pipeline.formatters.source_combiner import (
    CombinedContent,
    SourceCombiner,
    SourceFile,
)


class TestSourceFileDataclass:
    """Tests for SourceFile dataclass."""
    
    def test_create_source_file(self):
        """Test basic SourceFile creation."""
        source = SourceFile(
            path=Path("/test/file.json"),
            file_type="enriched_json",
            content={"summary": "test"},
            metadata={"title": "Test"}
        )
        assert source.path == Path("/test/file.json")
        assert source.file_type == "enriched_json"
        assert source.content == {"summary": "test"}
        assert source.metadata == {"title": "Test"}
    
    def test_source_file_default_metadata(self):
        """Test SourceFile with default metadata."""
        source = SourceFile(
            path=Path("/test/file.txt"),
            file_type="txt",
            content="Hello world"
        )
        assert source.metadata == {}


class TestCombinedContentDataclass:
    """Tests for CombinedContent dataclass."""
    
    def test_create_combined_content(self):
        """Test basic CombinedContent creation."""
        combined = CombinedContent(
            enriched_content={"summary": "combined"},
            source_files=[],
            source_count=0
        )
        assert combined.enriched_content == {"summary": "combined"}
        assert combined.source_files == []
        assert combined.source_count == 0
        assert combined.warnings == []


class TestSourceCombinerFileLoading:
    """Tests for SourceCombiner file loading."""
    
    def test_load_json_file(self):
        """Test loading a JSON file."""
        combiner = SourceCombiner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a JSON file
            json_path = Path(tmpdir) / "test.json"
            json_content = {
                "enrichment_version": "v1",
                "summary": {"short": "Test summary"},
                "tags": ["tag1", "tag2"]
            }
            with open(json_path, "w") as f:
                json.dump(json_content, f)
            
            # Load sources
            sources = combiner.load_sources(Path(tmpdir))
            
            assert len(sources) == 1
            assert sources[0].file_type == "enriched_json"
            assert sources[0].content == json_content
            assert sources[0].metadata["is_enriched"] is True
    
    def test_load_text_file(self):
        """Test loading a TXT file."""
        combiner = SourceCombiner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a text file
            txt_path = Path(tmpdir) / "notes.txt"
            txt_content = "These are my notes about the topic."
            with open(txt_path, "w") as f:
                f.write(txt_content)
            
            # Load sources
            sources = combiner.load_sources(Path(tmpdir))
            
            assert len(sources) == 1
            assert sources[0].file_type == "txt"
            assert sources[0].content == txt_content
            assert sources[0].metadata["char_count"] == len(txt_content)
    
    def test_load_markdown_file(self):
        """Test loading a Markdown file."""
        combiner = SourceCombiner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a markdown file
            md_path = Path(tmpdir) / "readme.md"
            md_content = "# Title\n\nThis is markdown content."
            with open(md_path, "w") as f:
                f.write(md_content)
            
            # Load sources
            sources = combiner.load_sources(Path(tmpdir))
            
            assert len(sources) == 1
            assert sources[0].file_type == "md"
            assert sources[0].content == md_content
    
    def test_load_multiple_files(self):
        """Test loading multiple files of different types."""
        combiner = SourceCombiner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON file
            json_path = Path(tmpdir) / "enriched.json"
            with open(json_path, "w") as f:
                json.dump({"summary": "test"}, f)
            
            # Create text file
            txt_path = Path(tmpdir) / "notes.txt"
            with open(txt_path, "w") as f:
                f.write("Notes content")
            
            # Create markdown file
            md_path = Path(tmpdir) / "readme.md"
            with open(md_path, "w") as f:
                f.write("# Markdown")
            
            # Load sources
            sources = combiner.load_sources(Path(tmpdir))
            
            assert len(sources) == 3
            file_types = {s.file_type for s in sources}
            assert file_types == {"enriched_json", "txt", "md"}
    
    def test_load_nonexistent_folder(self):
        """Test loading from non-existent folder raises error."""
        combiner = SourceCombiner()
        
        with pytest.raises(FileNotFoundError, match="Source folder not found"):
            combiner.load_sources(Path("/nonexistent/folder"))
    
    def test_load_empty_folder(self):
        """Test loading from empty folder raises error."""
        combiner = SourceCombiner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No supported files found"):
                combiner.load_sources(Path(tmpdir))
    
    def test_load_folder_with_unsupported_files_only(self):
        """Test loading folder with only unsupported files raises error."""
        combiner = SourceCombiner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create unsupported file
            unsupported_path = Path(tmpdir) / "image.png"
            with open(unsupported_path, "wb") as f:
                f.write(b"fake image data")
            
            with pytest.raises(ValueError, match="No supported files found"):
                combiner.load_sources(Path(tmpdir))
    
    def test_skip_invalid_json(self):
        """Test that invalid JSON files are skipped with warning."""
        combiner = SourceCombiner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid JSON file
            invalid_path = Path(tmpdir) / "invalid.json"
            with open(invalid_path, "w") as f:
                f.write("not valid json {{{")
            
            # Create valid text file
            valid_path = Path(tmpdir) / "valid.txt"
            with open(valid_path, "w") as f:
                f.write("Valid content")
            
            # Should load the valid file and skip invalid
            sources = combiner.load_sources(Path(tmpdir))
            
            assert len(sources) == 1
            assert sources[0].file_type == "txt"


class TestSourceCombinerMerging:
    """Tests for SourceCombiner content merging."""
    
    def test_merge_single_json(self):
        """Test merging a single JSON source."""
        combiner = SourceCombiner()
        
        source = SourceFile(
            path=Path("/test/file.json"),
            file_type="enriched_json",
            content={
                "enrichment_version": "v1",
                "summary": {"short": "Test summary"},
                "tags": ["tag1", "tag2"],
                "topics": ["topic1"],
            },
            metadata={"title": "Test"}
        )
        
        result = combiner.combine([source])
        
        assert result.source_count == 1
        assert result.enriched_content["tags"] == ["tag1", "tag2"]
        assert result.enriched_content["topics"] == ["topic1"]
    
    def test_merge_multiple_json_files(self):
        """Test merging multiple JSON sources."""
        combiner = SourceCombiner()
        
        source1 = SourceFile(
            path=Path("/test/file1.json"),
            file_type="enriched_json",
            content={
                "summary": {"short": "Summary 1"},
                "tags": ["tag1", "tag2"],
                "topics": ["topic1"],
            },
            metadata={"title": "File 1"}
        )
        
        source2 = SourceFile(
            path=Path("/test/file2.json"),
            file_type="enriched_json",
            content={
                "summary": {"short": "Summary 2"},
                "tags": ["tag2", "tag3"],
                "topics": ["topic2"],
            },
            metadata={"title": "File 2"}
        )
        
        result = combiner.combine([source1, source2])
        
        assert result.source_count == 2
        # Tags should be deduplicated
        assert len(result.enriched_content["tags"]) == 3
        assert "tag1" in result.enriched_content["tags"]
        assert "tag2" in result.enriched_content["tags"]
        assert "tag3" in result.enriched_content["tags"]
    
    def test_merge_with_text_sources(self):
        """Test merging JSON with text sources."""
        combiner = SourceCombiner()
        
        json_source = SourceFile(
            path=Path("/test/enriched.json"),
            file_type="enriched_json",
            content={"summary": {"short": "JSON summary"}},
            metadata={"title": "Enriched"}
        )
        
        text_source = SourceFile(
            path=Path("/test/notes.txt"),
            file_type="txt",
            content="Additional notes content",
            metadata={"title": "Notes"}
        )
        
        result = combiner.combine([json_source, text_source])
        
        assert result.source_count == 2
        assert "additional_context" in result.enriched_content
        assert "Notes" in result.enriched_content["additional_context"]
    
    def test_merge_text_only(self):
        """Test merging only text sources (no JSON)."""
        combiner = SourceCombiner()
        
        text_source = SourceFile(
            path=Path("/test/notes.txt"),
            file_type="txt",
            content="Just text content",
            metadata={"title": "Notes"}
        )
        
        result = combiner.combine([text_source])
        
        assert result.source_count == 1
        assert len(result.warnings) > 0  # Should warn about no JSON
        assert "additional_context" in result.enriched_content
    
    def test_source_attribution(self):
        """Test that source attribution is preserved."""
        combiner = SourceCombiner()
        
        source = SourceFile(
            path=Path("/test/file.json"),
            file_type="enriched_json",
            content={"summary": "test"},
            metadata={"title": "Test"}
        )
        
        result = combiner.combine([source])
        
        combined_sources = result.enriched_content["metadata"]["combined_sources"]
        assert combined_sources["source_count"] == 1
        assert "file.json" in combined_sources["source_files"]


class TestSourceCombinerDeduplication:
    """Tests for SourceCombiner deduplication."""
    
    def test_deduplicate_strings(self):
        """Test deduplication of string lists."""
        combiner = SourceCombiner()
        
        items = ["Tag1", "tag1", "TAG1", "Tag2", "tag2"]
        result = combiner._deduplicate_list(items)
        
        # Should keep first occurrence, case-insensitive dedup
        assert len(result) == 2
    
    def test_deduplicate_preserves_order(self):
        """Test that deduplication preserves order."""
        combiner = SourceCombiner()
        
        items = ["first", "second", "first", "third", "second"]
        result = combiner._deduplicate_list(items)
        
        assert result == ["first", "second", "third"]
    
    def test_deduplicate_empty_list(self):
        """Test deduplication of empty list."""
        combiner = SourceCombiner()
        
        result = combiner._deduplicate_list([])
        assert result == []
    
    def test_deduplicate_dicts(self):
        """Test deduplication of dict lists."""
        combiner = SourceCombiner()
        
        items = [
            {"title": "Chapter 1", "time": "00:00"},
            {"title": "Chapter 1", "time": "00:00"},  # Duplicate
            {"title": "Chapter 2", "time": "01:00"},
        ]
        result = combiner._deduplicate_list(items)
        
        assert len(result) == 2


class TestSourceCombinerIntegration:
    """Integration tests for SourceCombiner."""
    
    def test_full_workflow(self):
        """Test complete load and combine workflow."""
        combiner = SourceCombiner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create enriched JSON
            json_path = Path(tmpdir) / "video1-enriched.json"
            with open(json_path, "w") as f:
                json.dump({
                    "enrichment_version": "v1",
                    "metadata": {"title": "Video 1"},
                    "summary": {"short": "Summary of video 1"},
                    "tags": ["python", "backend"],
                    "topics": ["programming"],
                }, f)
            
            # Create another enriched JSON
            json_path2 = Path(tmpdir) / "video2-enriched.json"
            with open(json_path2, "w") as f:
                json.dump({
                    "enrichment_version": "v1",
                    "metadata": {"title": "Video 2"},
                    "summary": {"short": "Summary of video 2"},
                    "tags": ["python", "api"],
                    "topics": ["web development"],
                }, f)
            
            # Create notes file
            notes_path = Path(tmpdir) / "my-notes.txt"
            with open(notes_path, "w") as f:
                f.write("My additional research notes")
            
            # Load and combine
            sources = combiner.load_sources(Path(tmpdir))
            result = combiner.combine(sources)
            
            # Verify
            assert result.source_count == 3
            assert len(result.enriched_content["tags"]) == 3  # python, backend, api
            assert len(result.enriched_content["topics"]) == 2
            assert "additional_context" in result.enriched_content
