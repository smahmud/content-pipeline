"""
Clean transcript generator.

Generates cleaned and formatted transcripts with speaker labels,
paragraphs, and optional timestamps.

Supports platforms: All (general transcript format)
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("transcript-clean")
class TranscriptCleanGenerator(BaseGenerator):
    """Generator for clean transcript output format.
    
    Creates cleaned transcripts with:
    - Speaker labels (if available)
    - Paragraph breaks
    - Optional timestamps
    - Cleaned text (filler words removed)
    
    Required enrichments: transcript
    Optional enrichments: chapters
    """
    
    @property
    def output_type(self) -> str:
        return "transcript-clean"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["medium", "wordpress", "ghost", "substack"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["transcript"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for transcript template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        transcript = enriched_content.get("transcript", {})
        chapters = enriched_content.get("chapters", [])
        
        # Extract transcript text and segments
        transcript_text = ""
        segments = []
        
        if isinstance(transcript, dict):
            transcript_text = transcript.get("text", "")
            segments = transcript.get("segments", [])
        elif isinstance(transcript, str):
            transcript_text = transcript
        
        # Process segments for speaker labels
        processed_segments = []
        for segment in segments:
            if isinstance(segment, dict):
                processed_segments.append({
                    "speaker": segment.get("speaker", ""),
                    "text": segment.get("text", ""),
                    "start_time": segment.get("start_time", 0),
                    "end_time": segment.get("end_time", 0),
                })
        
        # Get title from metadata
        title = self._get_field(enriched_content, "metadata.title", "")
        
        # Check if we have speaker information
        has_speakers = any(s.get("speaker") for s in processed_segments)
        
        return {
            "title": title,
            "transcript_text": transcript_text,
            "segments": processed_segments,
            "has_speakers": has_speakers,
            "chapters": chapters,
            "platform": request.platform,
            "metadata": enriched_content.get("metadata", {}),
        }
