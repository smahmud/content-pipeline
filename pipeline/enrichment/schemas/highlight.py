"""
Highlight Enrichment Schema

Schema for highlight identification results. Identifies key moments with
timestamps, quotes, importance levels, and optional context.
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import re


class ImportanceLevel(str, Enum):
    """Importance level for highlights.
    
    Categorizes highlights by their significance:
    - HIGH: Critical insights, key takeaways, major revelations
    - MEDIUM: Important points, supporting arguments, notable examples
    - LOW: Interesting details, minor points, supplementary information
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HighlightEnrichment(BaseModel):
    """Highlight identification with timestamp, quote, and importance.
    
    Identifies key moments within content that deserve special attention:
    - Timestamp: When the highlight occurs (HH:MM:SS format)
    - Quote: The actual content being highlighted
    - Importance: Significance level (HIGH, MEDIUM, LOW)
    - Context: Optional additional context or explanation
    
    Attributes:
        timestamp: Timestamp of highlight (HH:MM:SS format)
        quote: Key quote or moment (max 1000 characters)
        importance: Importance level (HIGH, MEDIUM, LOW)
        context: Optional additional context (max 500 characters)
    """
    timestamp: str = Field(
        ...,
        description="Timestamp of highlight (HH:MM:SS format)"
    )
    quote: str = Field(
        ...,
        description="Key quote or moment",
        min_length=1,
        max_length=1000
    )
    importance: ImportanceLevel = Field(
        ...,
        description="Importance level"
    )
    context: Optional[str] = Field(
        None,
        description="Additional context",
        max_length=500
    )
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Validate timestamp is in HH:MM:SS format."""
        pattern = r'^\d{2}:\d{2}:\d{2}$'
        if not re.match(pattern, v):
            raise ValueError(
                f"Timestamp must be in HH:MM:SS format, got: {v}"
            )
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "00:12:45",
                "quote": "The key insight here is that neural networks learn hierarchical "
                        "representations of data, with each layer capturing increasingly "
                        "abstract features.",
                "importance": "high",
                "context": "This explains why deep learning is so effective for complex tasks "
                          "like image recognition and natural language processing."
            }
        }
