"""
Tag Enrichment Schema

Schema for tag extraction results. Provides structured metadata including
categories, keywords, and named entities.
"""

from typing import List
from pydantic import BaseModel, Field


class TagEnrichment(BaseModel):
    """Tag extraction with categories, keywords, and entities.
    
    Extracts structured metadata from content to enable categorization,
    search, and discovery:
    - Categories: High-level content classifications
    - Keywords: Important terms and phrases
    - Entities: Named entities (people, places, organizations)
    
    Attributes:
        categories: High-level content categories
        keywords: Important keywords and phrases
        entities: Named entities (people, places, organizations)
    """
    categories: List[str] = Field(
        ...,
        description="High-level content categories",
        min_items=1
    )
    keywords: List[str] = Field(
        ...,
        description="Important keywords and phrases",
        min_items=1
    )
    entities: List[str] = Field(
        ...,
        description="Named entities (people, places, organizations)",
        min_items=0  # Entities may not always be present
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "categories": ["Technology", "Artificial Intelligence", "Software Development"],
                "keywords": [
                    "machine learning",
                    "neural networks",
                    "deep learning",
                    "natural language processing",
                    "model training"
                ],
                "entities": [
                    "OpenAI",
                    "GPT-4",
                    "Python",
                    "TensorFlow",
                    "Stanford University"
                ]
            }
        }
