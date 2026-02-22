"""
Source combiner for multi-source input.

Combines multiple enriched files, PDFs, and text documents
into unified content for formatting.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class SourceFile:
    """Represents a single source file.
    
    Attributes:
        path: Path to the source file
        file_type: Type of file (enriched_json, pdf, txt, md)
        content: Parsed content (dict for JSON, str for text)
        metadata: Additional metadata about the source
    """
    path: Path
    file_type: Literal["enriched_json", "pdf", "txt", "md"]
    content: dict | str
    metadata: dict = field(default_factory=dict)


@dataclass
class CombinedContent:
    """Result of combining multiple sources.
    
    Attributes:
        enriched_content: Merged EnrichmentV1-compatible structure
        source_files: List of source files that were combined
        source_count: Number of sources combined
        warnings: List of warnings during combination
    """
    enriched_content: dict
    source_files: list[SourceFile]
    source_count: int
    warnings: list[str] = field(default_factory=list)


class SourceCombiner:
    """Combines multiple source files into unified content.
    
    Supports:
    - Enriched JSON files (EnrichmentV1 schema)
    - PDF files (text extraction)
    - TXT files (plain text)
    - MD files (markdown)
    """
    
    SUPPORTED_EXTENSIONS = {".json", ".pdf", ".txt", ".md"}
    
    def __init__(self) -> None:
        """Initialize the source combiner."""
        self._pdf_available = self._check_pdf_support()
    
    def _check_pdf_support(self) -> bool:
        """Check if PDF extraction is available."""
        try:
            import PyPDF2  # noqa: F401
            return True
        except ImportError:
            return False
    
    def load_sources(self, folder_path: Path) -> list[SourceFile]:
        """Load all supported files from a folder.
        
        Args:
            folder_path: Path to folder containing source files
            
        Returns:
            List of loaded SourceFile objects
            
        Raises:
            FileNotFoundError: If folder doesn't exist
            ValueError: If folder is empty or has no supported files
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Source folder not found: {folder_path}")
        
        if not folder.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        sources: list[SourceFile] = []
        skipped: list[str] = []
        
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    source = self._load_file(file_path)
                    if source:
                        sources.append(source)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
                    skipped.append(file_path.name)
        
        if not sources:
            if skipped:
                raise ValueError(
                    f"No valid source files found in {folder_path}. "
                    f"Skipped files: {', '.join(skipped)}"
                )
            raise ValueError(f"No supported files found in {folder_path}")
        
        logger.info(f"Loaded {len(sources)} source files from {folder_path}")
        if skipped:
            logger.warning(f"Skipped {len(skipped)} files: {', '.join(skipped)}")
        
        return sources
    
    def _load_file(self, file_path: Path) -> Optional[SourceFile]:
        """Load a single file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SourceFile object or None if loading fails
        """
        ext = file_path.suffix.lower()
        
        if ext == ".json":
            return self._load_json(file_path)
        elif ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext in {".txt", ".md"}:
            return self._load_text(file_path)
        
        return None
    
    def _load_json(self, file_path: Path) -> Optional[SourceFile]:
        """Load and validate an enriched JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            SourceFile with parsed JSON content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        
        # Check if it's an enriched JSON (has enrichment_version or summary/tags)
        is_enriched = (
            "enrichment_version" in content or
            "summary" in content or
            "tags" in content or
            "chapters" in content
        )
        
        return SourceFile(
            path=file_path,
            file_type="enriched_json",
            content=content,
            metadata={
                "is_enriched": is_enriched,
                "title": content.get("metadata", {}).get("title", file_path.stem),
            }
        )
    
    def _load_text(self, file_path: Path) -> Optional[SourceFile]:
        """Load a text or markdown file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            SourceFile with text content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        file_type: Literal["txt", "md"] = "md" if file_path.suffix.lower() == ".md" else "txt"
        
        return SourceFile(
            path=file_path,
            file_type=file_type,
            content=content,
            metadata={
                "title": file_path.stem,
                "char_count": len(content),
                "line_count": content.count("\n") + 1,
            }
        )
    
    def _load_pdf(self, file_path: Path) -> Optional[SourceFile]:
        """Load and extract text from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            SourceFile with extracted text content
        """
        if not self._pdf_available:
            logger.warning(f"PDF support not available. Install PyPDF2 to read {file_path.name}")
            return None
        
        try:
            import PyPDF2
            
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                content = "\n\n".join(text_parts)
            
            return SourceFile(
                path=file_path,
                file_type="pdf",
                content=content,
                metadata={
                    "title": file_path.stem,
                    "page_count": len(reader.pages),
                    "char_count": len(content),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to extract text from PDF {file_path.name}: {e}")
            return None
    
    def combine(self, sources: list[SourceFile]) -> CombinedContent:
        """Merge multiple sources into unified content.
        
        Args:
            sources: List of SourceFile objects to combine
            
        Returns:
            CombinedContent with merged enriched content
        """
        warnings: list[str] = []
        
        # Separate enriched JSON from text sources
        json_sources = [s for s in sources if s.file_type == "enriched_json"]
        text_sources = [s for s in sources if s.file_type in {"txt", "md", "pdf"}]
        
        # Start with merged JSON content or empty structure
        if json_sources:
            merged = self._merge_enriched_json(json_sources)
        else:
            merged = self._create_empty_enrichment()
            warnings.append("No enriched JSON files found, created empty structure")
        
        # Add text content as additional context
        if text_sources:
            additional_text = self._combine_text_sources(text_sources)
            merged["additional_context"] = additional_text
            merged["metadata"]["text_sources"] = [s.path.name for s in text_sources]
        
        # Add source attribution
        merged["metadata"]["combined_sources"] = {
            "source_count": len(sources),
            "source_files": [s.path.name for s in sources],
            "json_sources": len(json_sources),
            "text_sources": len(text_sources),
        }
        
        return CombinedContent(
            enriched_content=merged,
            source_files=sources,
            source_count=len(sources),
            warnings=warnings,
        )
    
    def _merge_enriched_json(self, json_sources: list[SourceFile]) -> dict:
        """Merge multiple enriched JSON files.
        
        Args:
            json_sources: List of JSON source files
            
        Returns:
            Merged enrichment dictionary
        """
        merged: dict[str, Any] = {
            "enrichment_version": "v1",
            "metadata": {
                "title": "Combined Content",
            },
            "summary": {},
            "tags": [],
            "chapters": [],
            "highlights": [],
            "key_points": [],
            "topics": [],
        }
        
        # Carry forward required metadata fields from first source
        first_meta = json_sources[0].content.get("metadata", {}) if json_sources else {}
        for field in ("provider", "model", "timestamp", "cost_usd", "tokens_used", "enrichment_types"):
            if field in first_meta:
                merged["metadata"][field] = first_meta[field]
        
        all_tags: list[str] = []
        all_categories: list[str] = []
        all_keywords: list[str] = []
        all_entities: list[str] = []
        all_topics: list[str] = []
        all_key_points: list[str] = []
        summaries: list[str] = []
        
        for source in json_sources:
            content = source.content
            if not isinstance(content, dict):
                continue
            
            # Collect summaries
            summary = content.get("summary", {})
            if isinstance(summary, dict):
                if summary.get("long"):
                    summaries.append(summary["long"])
                elif summary.get("medium"):
                    summaries.append(summary["medium"])
                elif summary.get("short"):
                    summaries.append(summary["short"])
            elif isinstance(summary, str):
                summaries.append(summary)
            
            # Collect tags
            tags = content.get("tags", [])
            if isinstance(tags, dict):
                # Handle EnrichmentV1 schema format (categories, keywords, entities)
                all_categories.extend(tags.get("categories", []))
                all_keywords.extend(tags.get("keywords", []))
                all_entities.extend(tags.get("entities", []))
                # Also handle legacy format (primary, secondary)
                all_tags.extend(tags.get("primary", []))
                all_tags.extend(tags.get("secondary", []))
            elif isinstance(tags, list):
                all_tags.extend(tags)
            
            # Collect topics
            topics = content.get("topics", [])
            if isinstance(topics, list):
                all_topics.extend(topics)
            
            # Collect key points
            key_points = content.get("key_points", [])
            if isinstance(key_points, list):
                all_key_points.extend(key_points)
            
            # Collect chapters
            chapters = content.get("chapters", [])
            if isinstance(chapters, list):
                merged["chapters"].extend(chapters)
            
            # Collect highlights
            highlights = content.get("highlights", [])
            if isinstance(highlights, list):
                merged["highlights"].extend(highlights)
        
        # Deduplicate and set merged values
        # Rebuild tags in EnrichmentV1 schema format if we have structured data
        if all_categories or all_keywords or all_entities:
            merged["tags"] = {
                "categories": self._deduplicate_list(all_categories),
                "keywords": self._deduplicate_list(all_keywords),
                "entities": self._deduplicate_list(all_entities),
            }
        else:
            merged["tags"] = self._deduplicate_list(all_tags)
        merged["topics"] = self._deduplicate_list(all_topics)
        merged["key_points"] = self._deduplicate_list(all_key_points)
        
        # Combine summaries
        if summaries:
            merged["summary"] = {
                "combined": "\n\n".join(summaries),
                "source_count": len(summaries),
            }
        
        # Set title from first source if available
        if json_sources:
            first_title = json_sources[0].metadata.get("title")
            if first_title:
                merged["metadata"]["title"] = f"Combined: {first_title}"
        
        return merged
    
    def _combine_text_sources(self, text_sources: list[SourceFile]) -> str:
        """Combine text sources into a single string.
        
        Args:
            text_sources: List of text source files
            
        Returns:
            Combined text content
        """
        parts = []
        for source in text_sources:
            title = source.metadata.get("title", source.path.name)
            content = source.content if isinstance(source.content, str) else str(source.content)
            parts.append(f"## {title}\n\n{content}")
        
        return "\n\n---\n\n".join(parts)
    
    def _create_empty_enrichment(self) -> dict:
        """Create an empty enrichment structure.
        
        Returns:
            Empty EnrichmentV1-compatible dictionary
        """
        return {
            "enrichment_version": "v1",
            "metadata": {
                "title": "Combined Content",
            },
            "summary": {},
            "tags": [],
            "chapters": [],
            "highlights": [],
            "key_points": [],
            "topics": [],
        }
    
    def _deduplicate_list(self, items: list) -> list:
        """Remove duplicates from a list while preserving order.
        
        Args:
            items: List with potential duplicates
            
        Returns:
            Deduplicated list
        """
        seen: set = set()
        result: list = []
        
        for item in items:
            # Normalize strings for comparison
            if isinstance(item, str):
                normalized = item.lower().strip()
                if normalized not in seen:
                    seen.add(normalized)
                    result.append(item)
            elif isinstance(item, dict):
                # For dicts, use a string representation
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            else:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
        
        return result
