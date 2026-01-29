"""
Output File Management

Handles output file path generation, validation, and writing for
enrichment results. Provides automatic filename generation, directory
creation, and overwrite confirmation.
"""

import os
import json
from pathlib import Path
from typing import Optional

from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1
from pipeline.enrichment.errors import OutputFileError


def generate_output_filename(input_path: str, suffix: str = "-enriched") -> str:
    """Generate output filename from input filename.
    
    Appends suffix to the input filename before the extension.
    
    Args:
        input_path: Input file path
        suffix: Suffix to append (default: "-enriched")
        
    Returns:
        Generated output filename
        
    Example:
        >>> generate_output_filename("transcript.json")
        "transcript-enriched.json"
    """
    path = Path(input_path)
    stem = path.stem
    extension = path.suffix
    
    return f"{stem}{suffix}{extension}"


def resolve_output_path(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Path:
    """Resolve output file path.
    
    Priority:
    1. If output_path is specified, use it
    2. If output_dir is specified, use input filename in that directory
    3. Otherwise, generate filename in same directory as input
    
    Args:
        input_path: Input file path
        output_path: Optional explicit output path
        output_dir: Optional output directory
        
    Returns:
        Resolved output path
    """
    input_path_obj = Path(input_path)
    
    if output_path:
        # Explicit output path specified
        return Path(output_path)
    
    elif output_dir:
        # Output directory specified - use input filename
        output_filename = generate_output_filename(input_path_obj.name)
        return Path(output_dir) / output_filename
    
    else:
        # Generate filename in same directory as input
        output_filename = generate_output_filename(input_path_obj.name)
        return input_path_obj.parent / output_filename


def validate_output_path(output_path: Path, create_dirs: bool = True) -> None:
    """Validate output path is writable.
    
    Args:
        output_path: Output file path to validate
        create_dirs: Whether to create parent directories if they don't exist
        
    Raises:
        OutputFileError: If path is not writable or directory cannot be created
    """
    # Check if parent directory exists
    parent_dir = output_path.parent
    
    if not parent_dir.exists():
        if create_dirs:
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise OutputFileError(
                    f"Cannot create output directory '{parent_dir}': {e}"
                )
        else:
            raise OutputFileError(
                f"Output directory does not exist: {parent_dir}"
            )
    
    # Check if parent directory is writable
    if not os.access(parent_dir, os.W_OK):
        raise OutputFileError(
            f"Output directory is not writable: {parent_dir}"
        )
    
    # Check if output file already exists and is writable
    if output_path.exists():
        if not os.access(output_path, os.W_OK):
            raise OutputFileError(
                f"Output file exists but is not writable: {output_path}"
            )


def check_overwrite(output_path: Path, force: bool = False) -> bool:
    """Check if output file should be overwritten.
    
    Args:
        output_path: Output file path
        force: If True, always overwrite without prompting
        
    Returns:
        True if should proceed with write, False otherwise
    """
    if not output_path.exists():
        return True
    
    if force:
        return True
    
    # Prompt user for confirmation
    response = input(
        f"Output file '{output_path}' already exists. Overwrite? (y/n): "
    ).strip().lower()
    
    return response in ('y', 'yes')


def write_enrichment_result(
    result: EnrichmentV1,
    output_path: Path,
    indent: int = 2,
    force: bool = False
) -> None:
    """Write enrichment result to file.
    
    Args:
        result: Enrichment result to write
        output_path: Output file path
        indent: JSON indentation (default: 2)
        force: If True, overwrite without prompting
        
    Raises:
        OutputFileError: If write fails
    """
    # Validate output path
    validate_output_path(output_path, create_dirs=True)
    
    # Check overwrite
    if not check_overwrite(output_path, force=force):
        raise OutputFileError(
            f"User cancelled overwrite of '{output_path}'"
        )
    
    try:
        # Convert to dict
        result_dict = result.model_dump()
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=indent, default=str)
    
    except Exception as e:
        raise OutputFileError(
            f"Failed to write output file '{output_path}': {e}"
        )


def write_batch_results(
    results: dict,
    output_dir: Path,
    force: bool = False
) -> dict:
    """Write batch enrichment results to files.
    
    Args:
        results: Dict mapping input path to enrichment result
        output_dir: Output directory for all files
        force: If True, overwrite without prompting
        
    Returns:
        Dict with write statistics (success_count, failed_count, failed_files)
        
    Raises:
        OutputFileError: If output directory cannot be created
    """
    # Validate output directory
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OutputFileError(
                f"Cannot create output directory '{output_dir}': {e}"
            )
    
    stats = {
        "success_count": 0,
        "failed_count": 0,
        "failed_files": []
    }
    
    for input_path, result in results.items():
        try:
            # Generate output filename
            output_filename = generate_output_filename(Path(input_path).name)
            output_path = output_dir / output_filename
            
            # Write result
            write_enrichment_result(result, output_path, force=force)
            stats["success_count"] += 1
        
        except Exception as e:
            stats["failed_count"] += 1
            stats["failed_files"].append({
                "input_path": input_path,
                "error": str(e)
            })
    
    return stats


class OutputManager:
    """Manager for output file operations.
    
    Provides a high-level interface for handling output files with
    automatic path resolution, validation, and error handling.
    
    Example:
        >>> manager = OutputManager()
        >>> output_path = manager.prepare_output(
        ...     input_path="transcript.json",
        ...     output_path=None,
        ...     output_dir="./enriched"
        ... )
        >>> manager.write_result(result, output_path)
    """
    
    def __init__(self, force_overwrite: bool = False):
        """Initialize output manager.
        
        Args:
            force_overwrite: If True, always overwrite without prompting
        """
        self.force_overwrite = force_overwrite
    
    def prepare_output(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Path:
        """Prepare output path and validate it.
        
        Args:
            input_path: Input file path
            output_path: Optional explicit output path
            output_dir: Optional output directory
            
        Returns:
            Validated output path
            
        Raises:
            OutputFileError: If path is invalid or not writable
        """
        # Resolve output path
        resolved_path = resolve_output_path(input_path, output_path, output_dir)
        
        # Validate path
        validate_output_path(resolved_path, create_dirs=True)
        
        return resolved_path
    
    def write_result(
        self,
        result: EnrichmentV1,
        output_path: Path,
        indent: int = 2
    ) -> None:
        """Write enrichment result to file.
        
        Args:
            result: Enrichment result to write
            output_path: Output file path
            indent: JSON indentation
            
        Raises:
            OutputFileError: If write fails
        """
        write_enrichment_result(
            result,
            output_path,
            indent=indent,
            force=self.force_overwrite
        )
