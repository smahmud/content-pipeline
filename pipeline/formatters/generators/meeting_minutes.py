"""
Meeting minutes generator.

Generates structured meeting minutes with attendees, agenda,
discussion points, decisions, and action items.

Supports platforms: General document formats
"""

from pipeline.formatters.base import FormatRequest
from pipeline.formatters.generators.base_generator import BaseGenerator, GeneratorConfig
from pipeline.formatters.generators.factory import GeneratorFactory


@GeneratorFactory.register("meeting-minutes")
class MeetingMinutesGenerator(BaseGenerator):
    """Generator for meeting minutes output format.
    
    Creates meeting minutes with:
    - Meeting title and date
    - Attendees (from speakers if available)
    - Agenda/topics discussed
    - Key discussion points
    - Decisions made
    - Action items
    
    Required enrichments: summary
    Optional enrichments: chapters, highlights, transcript
    """
    
    @property
    def output_type(self) -> str:
        return "meeting-minutes"
    
    @property
    def supported_platforms(self) -> list[str]:
        return ["medium", "wordpress", "ghost", "substack"]
    
    @property
    def required_enrichments(self) -> list[str]:
        return ["summary"]
    
    def _build_template_context(
        self,
        enriched_content: dict,
        request: FormatRequest,
    ) -> dict:
        """Build context for meeting minutes template.
        
        Args:
            enriched_content: The enriched content data
            request: The format request with options
            
        Returns:
            Template context dictionary
        """
        summary = enriched_content.get("summary", {})
        chapters = enriched_content.get("chapters", [])
        highlights = enriched_content.get("highlights", [])
        transcript = enriched_content.get("transcript", {})
        
        # Get title from metadata
        title = self._get_field(enriched_content, "metadata.title", "Meeting Notes")
        
        # Extract attendees from transcript speakers
        attendees = []
        if isinstance(transcript, dict):
            segments = transcript.get("segments", [])
            speakers = set()
            for segment in segments:
                if isinstance(segment, dict) and segment.get("speaker"):
                    speakers.add(segment["speaker"])
            attendees = list(speakers)
        
        # Build agenda from chapters
        agenda = []
        for chapter in chapters:
            if isinstance(chapter, dict):
                agenda.append({
                    "topic": chapter.get("title", ""),
                    "summary": chapter.get("summary", ""),
                })
        
        # Build discussion points from highlights
        discussion_points = []
        if highlights:
            discussion_points = [
                h.get("text", h) if isinstance(h, dict) else h 
                for h in highlights[:10]
            ]
        
        # Get date from metadata
        date = self._get_field(enriched_content, "metadata.date", "")
        if not date:
            date = self._get_field(enriched_content, "metadata.timestamp", "")
        
        return {
            "title": title,
            "date": date,
            "attendees": attendees,
            "agenda": agenda,
            "discussion_points": discussion_points,
            "summary": summary,
            "short_summary": summary.get("short", ""),
            "medium_summary": summary.get("medium", ""),
            "chapters": chapters,
            "highlights": highlights,
            "platform": request.platform,
            "metadata": enriched_content.get("metadata", {}),
        }
