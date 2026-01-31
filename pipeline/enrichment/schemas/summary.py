"""
Summary Enrichment Schema

Schema for summary enrichment results. Provides three length variants
(short, medium, long) to accommodate different use cases.
"""

from pydantic import BaseModel, Field


class SummaryEnrichment(BaseModel):
    """Summary enrichment with multiple length variants.
    
    Provides three different summary lengths to accommodate different
    use cases:
    - Short: 1-2 sentences for quick overview
    - Medium: Paragraph-length for moderate detail
    - Long: Multi-paragraph for comprehensive understanding
    
    Attributes:
        short: 1-2 sentence summary (max 500 characters)
        medium: Paragraph-length summary (max 2000 characters)
        long: Detailed multi-paragraph summary (max 5000 characters)
    """
    short: str = Field(
        ...,
        description="1-2 sentence summary",
        max_length=500
    )
    medium: str = Field(
        ...,
        description="Paragraph-length summary",
        max_length=2000
    )
    long: str = Field(
        ...,
        description="Detailed multi-paragraph summary",
        max_length=5000
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "short": "A brief overview of the main topic discussed in the content.",
                "medium": "This content covers several key points including the main topic, "
                         "supporting arguments, and practical examples. The discussion provides "
                         "valuable insights into the subject matter.",
                "long": "This comprehensive content explores the main topic in depth, beginning "
                       "with foundational concepts and building toward more advanced ideas. "
                       "The discussion includes multiple perspectives, practical examples, and "
                       "actionable insights. Key themes include the importance of understanding "
                       "core principles, applying knowledge in real-world scenarios, and "
                       "considering various approaches to problem-solving."
            }
        }
