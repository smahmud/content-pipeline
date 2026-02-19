"""
OutputWriter - Handles output file management for formatted content.

This module provides functionality for:
- Output path handling (--output flag)
- Default filename generation (append output type)
- Directory creation
- Overwrite confirmation
- Metadata embedding in Markdown frontmatter
- Sidecar metadata file generation

Implements Requirements: 16.1-16.7, 12.5, 12.6
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from pipeline.formatters.base import FormatResult
from pipeline.formatters.schemas.format_v1 import FormatV1


logger = logging.getLogger(__name__)


@dataclass
class WriteResult:
    """Result of a write operation.
    
    Attributes:
        success: Whether the write succeeded
        output_path: Path where content was written
        metadata_path: Path to sidecar metadata file (if generated)
        overwritten: Whether an existing file was overwritten
        error: Error message if write failed
    """
    success: bool
    output_path: str
    metadata_path: Optional[str] = None
    overwritten: bool = False
    error: Optional[str] = None


class OutputWriter:
    """Handles output file management for formatted content.
    
    The OutputWriter manages:
    - Path resolution and default filename generation
    - Directory creation
    - Overwrite confirmation
    - Metadata embedding (frontmatter for Markdown, inline for JSON)
    - Sidecar metadata file generation
    
    Example:
        >>> writer = OutputWriter()
        >>> result = writer.write(
        ...     format_result=format_result,
        ...     output_path="output/blog.md",
        ...     embed_metadata=True
        ... )
    """
    
    # Output types that produce Markdown files
    MARKDOWN_TYPES = {
        "blog", "tweet", "youtube", "linkedin", "newsletter",
        "chapters", "transcript-clean", "podcast-notes", "meeting-minutes",
        "slides", "notion", "obsidian", "quote-cards", "video-script",
        "tiktok-script",
    }
    
    # Output types that produce JSON files
    JSON_TYPES = {"seo"}
    
    def __init__(self, force_overwrite: bool = False):
        """Initialize the OutputWriter.
        
        Args:
            force_overwrite: If True, overwrite existing files without prompting
        """
        self._force_overwrite = force_overwrite
    
    def write(
        self,
        format_result: FormatResult,
        output_path: Optional[str] = None,
        input_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        embed_metadata: bool = True,
        generate_sidecar: bool = False,
        force: bool = False,
    ) -> WriteResult:
        """Write formatted content to a file.
        
        Args:
            format_result: FormatResult containing content and metadata
            output_path: Explicit output path (overrides default generation)
            input_path: Input file path (used for default filename generation)
            output_dir: Output directory (used with default filename)
            embed_metadata: Whether to embed metadata in the output file
            generate_sidecar: Whether to generate a sidecar .meta.json file
            force: Force overwrite without confirmation
            
        Returns:
            WriteResult with success status and paths
        """
        try:
            # Determine output path
            resolved_path = self._resolve_output_path(
                output_path=output_path,
                input_path=input_path,
                output_dir=output_dir,
                output_type=format_result.metadata.output_type if format_result.metadata else "output",
            )
            
            # Check if path is writable
            if not self._is_path_writable(resolved_path):
                return WriteResult(
                    success=False,
                    output_path=resolved_path,
                    error=f"Output path is not writable: {resolved_path}",
                )
            
            # Check for existing file
            overwritten = False
            if os.path.exists(resolved_path):
                if not (force or self._force_overwrite):
                    return WriteResult(
                        success=False,
                        output_path=resolved_path,
                        error=f"File already exists: {resolved_path}. Use --force to overwrite.",
                    )
                overwritten = True
            
            # Create directory if needed
            self._ensure_directory(resolved_path)
            
            # Prepare content with optional metadata embedding
            content = format_result.content
            if embed_metadata and format_result.metadata:
                content = self._embed_metadata(
                    content=content,
                    metadata=format_result.metadata,
                )
            
            # Write the file
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"Wrote formatted output to: {resolved_path}")
            
            # Generate sidecar metadata file if requested
            metadata_path = None
            if generate_sidecar and format_result.metadata:
                metadata_path = self._write_sidecar_metadata(
                    output_path=resolved_path,
                    metadata=format_result.metadata,
                )
            
            return WriteResult(
                success=True,
                output_path=resolved_path,
                metadata_path=metadata_path,
                overwritten=overwritten,
            )
            
        except Exception as e:
            logger.error(f"Failed to write output: {e}")
            return WriteResult(
                success=False,
                output_path=output_path or "",
                error=str(e),
            )
    
    def _resolve_output_path(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        output_dir: Optional[str],
        output_type: str,
    ) -> str:
        """Resolve the output file path.
        
        Priority:
        1. Explicit output_path if provided
        2. Generated from input_path + output_type in output_dir
        3. Generated from output_type in output_dir or current directory
        
        Args:
            output_path: Explicit output path
            input_path: Input file path for default generation
            output_dir: Output directory
            output_type: Output type for extension and naming
            
        Returns:
            Resolved output path
        """
        # Use explicit path if provided
        if output_path:
            return output_path
        
        # Generate default filename
        filename = self.generate_filename(
            input_path=input_path,
            output_type=output_type,
        )
        
        # Combine with output directory
        if output_dir:
            return os.path.join(output_dir, filename)
        
        return filename
    
    def generate_filename(
        self,
        input_path: Optional[str],
        output_type: str,
    ) -> str:
        """Generate a default output filename.
        
        The filename is generated by:
        1. Taking the input filename stem (without extension)
        2. Appending the output type
        3. Adding the appropriate extension (.md or .json)
        
        Args:
            input_path: Input file path (optional)
            output_type: Output type for naming
            
        Returns:
            Generated filename
        """
        # Determine extension based on output type
        extension = self._get_extension(output_type)
        
        if input_path:
            # Get stem from input path (remove all extensions)
            stem = Path(input_path).stem
            # Remove common suffixes like .enriched
            if stem.endswith(".enriched"):
                stem = stem[:-9]
            return f"{stem}_{output_type}{extension}"
        
        # Fallback to just output type
        return f"output_{output_type}{extension}"
    
    def _get_extension(self, output_type: str) -> str:
        """Get the file extension for an output type.
        
        Args:
            output_type: Output type
            
        Returns:
            File extension including the dot
        """
        if output_type in self.JSON_TYPES:
            return ".json"
        return ".md"
    
    def _is_path_writable(self, path: str) -> bool:
        """Check if a path is writable.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is writable
        """
        try:
            # Check if parent directory exists and is writable
            parent = Path(path).parent
            if parent.exists():
                return os.access(parent, os.W_OK)
            # If parent doesn't exist, check if we can create it
            return self._can_create_directory(str(parent))
        except Exception:
            return False
    
    def _can_create_directory(self, path: str) -> bool:
        """Check if a directory can be created.
        
        Args:
            path: Directory path to check
            
        Returns:
            True if the directory can be created
        """
        try:
            # Walk up to find an existing parent
            current = Path(path)
            while not current.exists():
                current = current.parent
                if current == current.parent:  # Reached root
                    return False
            return os.access(current, os.W_OK)
        except Exception:
            return False
    
    def _ensure_directory(self, file_path: str) -> None:
        """Ensure the directory for a file path exists.
        
        Args:
            file_path: File path whose parent directory should exist
        """
        parent = Path(file_path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {parent}")
    
    def _embed_metadata(self, content: str, metadata: FormatV1) -> str:
        """Embed metadata into the content.
        
        For Markdown files, embeds as YAML frontmatter.
        For JSON files, embeds as a top-level metadata field.
        
        Args:
            content: Original content
            metadata: FormatV1 metadata to embed
            
        Returns:
            Content with embedded metadata
        """
        output_type = metadata.output_type
        
        if output_type in self.JSON_TYPES:
            return self._embed_json_metadata(content, metadata)
        else:
            return self._embed_markdown_frontmatter(content, metadata)
    
    def _embed_markdown_frontmatter(self, content: str, metadata: FormatV1) -> str:
        """Embed metadata as YAML frontmatter in Markdown.
        
        Args:
            content: Markdown content
            metadata: FormatV1 metadata
            
        Returns:
            Markdown with frontmatter
        """
        # Convert metadata to dictionary
        meta_dict = self._metadata_to_dict(metadata)
        
        # Generate YAML frontmatter
        frontmatter = yaml.dump(
            meta_dict,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        
        # Check if content already has frontmatter
        if content.startswith("---\n"):
            # Find the end of existing frontmatter
            end_idx = content.find("\n---\n", 4)
            if end_idx != -1:
                # Replace existing frontmatter
                return f"---\n{frontmatter}---\n{content[end_idx + 5:]}"
        
        # Add new frontmatter
        return f"---\n{frontmatter}---\n\n{content}"
    
    def _embed_json_metadata(self, content: str, metadata: FormatV1) -> str:
        """Embed metadata in JSON content.
        
        Args:
            content: JSON content string
            metadata: FormatV1 metadata
            
        Returns:
            JSON string with embedded metadata
        """
        try:
            # Parse existing JSON
            data = json.loads(content)
            
            # Add metadata
            data["_metadata"] = self._metadata_to_dict(metadata)
            
            # Re-serialize
            return json.dumps(data, indent=2, ensure_ascii=False)
            
        except json.JSONDecodeError:
            # If content is not valid JSON, wrap it
            return json.dumps({
                "content": content,
                "_metadata": self._metadata_to_dict(metadata),
            }, indent=2, ensure_ascii=False)
    
    def _metadata_to_dict(self, metadata: FormatV1) -> Dict[str, Any]:
        """Convert FormatV1 metadata to a dictionary.
        
        Args:
            metadata: FormatV1 metadata object
            
        Returns:
            Dictionary representation
        """
        result = {
            "format_version": metadata.format_version,
            "output_type": metadata.output_type,
            "timestamp": metadata.timestamp.isoformat() if metadata.timestamp else None,
            "source_file": metadata.source_file,
        }
        
        # Add optional fields
        if metadata.platform:
            result["platform"] = metadata.platform
        
        if metadata.style_profile_used:
            result["style_profile_used"] = metadata.style_profile_used
        
        if metadata.tone:
            result["tone"] = metadata.tone
        
        if metadata.length:
            result["length"] = metadata.length
        
        # Add LLM metadata if present
        if metadata.llm_metadata:
            result["llm"] = {
                "provider": metadata.llm_metadata.provider,
                "model": metadata.llm_metadata.model,
                "cost_usd": metadata.llm_metadata.cost_usd,
                "tokens_used": metadata.llm_metadata.tokens_used,
                "enhanced": metadata.llm_metadata.enhanced,
            }
        
        # Add validation metadata
        if metadata.validation:
            result["validation"] = {
                "platform": metadata.validation.platform,
                "character_count": metadata.validation.character_count,
                "truncated": metadata.validation.truncated,
            }
            if metadata.validation.warnings:
                result["validation"]["warnings"] = metadata.validation.warnings
        
        return result
    
    def _write_sidecar_metadata(
        self,
        output_path: str,
        metadata: FormatV1,
    ) -> str:
        """Write a sidecar metadata file.
        
        Creates a .meta.json file alongside the output file.
        
        Args:
            output_path: Path to the output file
            metadata: FormatV1 metadata
            
        Returns:
            Path to the sidecar metadata file
        """
        # Generate sidecar path
        sidecar_path = f"{output_path}.meta.json"
        
        # Convert metadata to dictionary
        meta_dict = self._metadata_to_dict(metadata)
        
        # Write sidecar file
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Wrote sidecar metadata to: {sidecar_path}")
        
        return sidecar_path
    
    def write_bundle_outputs(
        self,
        results: Dict[str, FormatResult],
        output_dir: str,
        input_path: Optional[str] = None,
        embed_metadata: bool = True,
        generate_sidecar: bool = False,
        force: bool = False,
    ) -> Dict[str, WriteResult]:
        """Write multiple bundle outputs to a directory.
        
        Args:
            results: Dictionary mapping output_type to FormatResult
            output_dir: Directory to write outputs
            input_path: Input file path for filename generation
            embed_metadata: Whether to embed metadata
            generate_sidecar: Whether to generate sidecar files
            force: Force overwrite
            
        Returns:
            Dictionary mapping output_type to WriteResult
        """
        write_results = {}
        
        for output_type, format_result in results.items():
            write_result = self.write(
                format_result=format_result,
                input_path=input_path,
                output_dir=output_dir,
                embed_metadata=embed_metadata,
                generate_sidecar=generate_sidecar,
                force=force,
            )
            write_results[output_type] = write_result
        
        return write_results
    
    @property
    def force_overwrite(self) -> bool:
        """Get the force overwrite setting."""
        return self._force_overwrite
    
    @force_overwrite.setter
    def force_overwrite(self, value: bool) -> None:
        """Set the force overwrite setting."""
        self._force_overwrite = value
