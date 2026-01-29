"""
Chapter Enrichment Schema

Schema for chapter detection results. Identifies content segments with
titles, timestamp ranges, and descriptions.
"""

from pydantic import BaseModel, Field, field_validator
import re


class ChapterEnrichment(BaseModel):
    """Chapter detection with title, timestamps, and description.
    
    Identifies logical segments within content, providing structure
    for navigation and understanding:
    - Title: Descriptive chapter name
    - Timestamps: Start and end times in HH:MM:SS format
    - Description: Brief summary of chapter content
    
    Attributes:
        title: Chapter title
        start_time: Start timestamp (HH:MM:SS format)
        end_time: End timestamp (HH:MM:SS format)
        description: Brief chapter description (max 500 characters)
    """
    title: str = Field(
        ...,
        description="Chapter title",
        min_length=1,
        max_length=200
    )
    start_time: str = Field(
        ...,
        description="Start timestamp (HH:MM:SS format)"
    )
    end_time: str = Field(
        ...,
        description="End timestamp (HH:MM:SS format)"
    )
    description: str = Field(
        ...,
        description="Brief chapter description",
        max_length=500
    )
    
    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Validate timestamp is in HH:MM:SS format."""
        pattern = r'^\d{2}:\d{2}:\d{2}$'
        if not re.match(pattern, v):
            raise ValueError(
                f"Timestamp must be in HH:MM:SS format, got: {v}"
            )
        return v
    
    def model_post_init(self, __context):
        """Validate that end_time is after start_time."""
        def time_to_seconds(time_str: str) -> int:
            """Convert HH:MM:SS to total seconds."""
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
        
        start_seconds = time_to_seconds(self.start_time)
        end_seconds = time_to_seconds(self.end_time)
        
        if end_seconds <= start_seconds:
            raise ValueError(
                f"end_time ({self.end_time}) must be after start_time ({self.start_time})"
            )
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Introduction to Machine Learning",
                "start_time": "00:00:00",
                "end_time": "00:15:30",
                "description": "Overview of machine learning concepts, including supervised "
                              "and unsupervised learning, with practical examples."
            }
        }
