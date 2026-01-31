"""
Mock LLM Responses for Testing

This module provides realistic mock responses for all LLM providers and
enrichment types. These mocks are used in unit and integration tests to
avoid making actual API calls while still testing the full enrichment pipeline.

The mocks include:
- Valid responses for all enrichment types (summary, tags, chapters, highlights)
- Error responses (rate limits, authentication, malformed JSON)
- Edge cases (empty content, very long content, special characters)
"""

from datetime import datetime
from typing import Dict, Any, List


# =============================================================================
# VALID ENRICHMENT RESPONSES
# =============================================================================

MOCK_SUMMARY_RESPONSE = {
    "short": "A comprehensive discussion about machine learning fundamentals and their practical applications in modern AI systems.",
    "medium": "This content explores the foundations of machine learning, covering supervised and unsupervised learning approaches. "
             "The discussion includes practical examples of neural networks, deep learning architectures, and real-world applications "
             "in computer vision and natural language processing. Key concepts include model training, optimization techniques, "
             "and the importance of data quality in achieving accurate predictions.",
    "long": "This comprehensive exploration of machine learning begins with fundamental concepts and progressively builds toward "
           "advanced topics in artificial intelligence. The content covers the distinction between supervised learning, where models "
           "learn from labeled data, and unsupervised learning, which discovers patterns in unlabeled datasets. Detailed explanations "
           "of neural network architectures demonstrate how layers of interconnected nodes process information hierarchically. "
           "The discussion emphasizes practical applications, including image recognition systems that can identify objects with "
           "human-level accuracy, and natural language processing models that understand and generate human language. Throughout, "
           "the importance of proper data preparation, model selection, and hyperparameter tuning is highlighted as critical to "
           "achieving optimal performance. The content concludes with insights into current challenges and future directions in "
           "machine learning research, including interpretability, fairness, and efficiency."
}

MOCK_TAG_RESPONSE = {
    "categories": [
        "Technology",
        "Artificial Intelligence",
        "Machine Learning",
        "Software Development"
    ],
    "keywords": [
        "neural networks",
        "deep learning",
        "supervised learning",
        "unsupervised learning",
        "model training",
        "optimization",
        "computer vision",
        "natural language processing",
        "data quality",
        "hyperparameter tuning"
    ],
    "entities": [
        "TensorFlow",
        "PyTorch",
        "Python",
        "Stanford University",
        "ImageNet",
        "GPT",
        "BERT"
    ]
}

MOCK_CHAPTERS_RESPONSE = [
    {
        "title": "Introduction to Machine Learning",
        "start_time": "00:00:00",
        "end_time": "00:08:30",
        "description": "Overview of machine learning concepts, historical context, and the distinction between AI, machine learning, and deep learning."
    },
    {
        "title": "Supervised Learning Fundamentals",
        "start_time": "00:08:30",
        "end_time": "00:18:45",
        "description": "Detailed explanation of supervised learning, including classification and regression tasks, with practical examples."
    },
    {
        "title": "Neural Networks and Deep Learning",
        "start_time": "00:18:45",
        "end_time": "00:32:15",
        "description": "Architecture of neural networks, backpropagation, activation functions, and the power of deep learning."
    },
    {
        "title": "Practical Applications and Case Studies",
        "start_time": "00:32:15",
        "end_time": "00:45:00",
        "description": "Real-world applications in computer vision, NLP, and other domains, with discussion of challenges and best practices."
    }
]

MOCK_HIGHLIGHTS_RESPONSE = [
    {
        "timestamp": "00:05:12",
        "quote": "Machine learning is fundamentally about teaching computers to learn from experience, rather than being explicitly programmed for every task.",
        "importance": "high",
        "context": "This defines the core principle that distinguishes machine learning from traditional programming approaches."
    },
    {
        "timestamp": "00:12:45",
        "quote": "The key insight is that neural networks learn hierarchical representations, with each layer capturing increasingly abstract features.",
        "importance": "high",
        "context": "Explains why deep learning is so effective for complex tasks like image recognition and language understanding."
    },
    {
        "timestamp": "00:25:30",
        "quote": "Data quality is often more important than model complexity. Garbage in, garbage out applies especially to machine learning.",
        "importance": "medium",
        "context": "Emphasizes the critical importance of data preparation and cleaning in the ML pipeline."
    },
    {
        "timestamp": "00:38:20",
        "quote": "Transfer learning allows us to leverage pre-trained models, dramatically reducing the data and compute requirements for new tasks.",
        "importance": "medium",
        "context": "Introduces a practical technique that has made deep learning accessible for many applications."
    }
]

# Complete enrichment response with all types
MOCK_COMPLETE_ENRICHMENT = {
    "enrichment_version": "v1",
    "metadata": {
        "provider": "openai",
        "model": "gpt-4-turbo",
        "timestamp": "2026-01-29T10:30:00Z",
        "cost_usd": 0.31,
        "tokens_used": 8300,
        "enrichment_types": ["summary", "tags", "chapters", "highlights"],
        "cache_hit": False
    },
    "summary": MOCK_SUMMARY_RESPONSE,
    "tags": MOCK_TAG_RESPONSE,
    "chapters": MOCK_CHAPTERS_RESPONSE,
    "highlights": MOCK_HIGHLIGHTS_RESPONSE
}


# =============================================================================
# PROVIDER-SPECIFIC RESPONSE FORMATS
# =============================================================================

def get_openai_response(content: str) -> Dict[str, Any]:
    """Mock OpenAI API response format."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 2500,
            "completion_tokens": 800,
            "total_tokens": 3300
        }
    }


def get_claude_response(content: str) -> Dict[str, Any]:
    """Mock Claude API response format."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": content
        }],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 2500,
            "output_tokens": 800
        }
    }


def get_bedrock_response(content: str) -> Dict[str, Any]:
    """Mock AWS Bedrock response format."""
    return {
        "body": {
            "completion": content,
            "stop_reason": "end_turn",
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 2500,
                "outputTokenCount": 800,
                "invocationLatency": 1234,
                "firstByteLatency": 456
            }
        }
    }


def get_ollama_response(content: str) -> Dict[str, Any]:
    """Mock Ollama API response format."""
    return {
        "model": "llama2",
        "created_at": "2026-01-29T10:30:00Z",
        "response": content,
        "done": True,
        "context": [1, 2, 3],  # Token context
        "total_duration": 5000000000,
        "load_duration": 1000000000,
        "prompt_eval_count": 2500,
        "eval_count": 800
    }


# =============================================================================
# ERROR RESPONSES
# =============================================================================

MOCK_RATE_LIMIT_ERROR = {
    "error": {
        "message": "Rate limit exceeded. Please try again later.",
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded"
    }
}

MOCK_AUTH_ERROR = {
    "error": {
        "message": "Invalid API key provided.",
        "type": "invalid_request_error",
        "code": "invalid_api_key"
    }
}

MOCK_TIMEOUT_ERROR = {
    "error": {
        "message": "Request timed out. Please try again.",
        "type": "timeout_error",
        "code": "timeout"
    }
}

MOCK_MALFORMED_JSON_RESPONSE = """
{
    "summary": {
        "short": "This is a test",
        "medium": "This is a longer test"
        // Missing closing brace and long field
"""

MOCK_INVALID_SCHEMA_RESPONSE = {
    "summary": {
        "short": "Valid short summary",
        # Missing required 'medium' and 'long' fields
    }
}


# =============================================================================
# EDGE CASE RESPONSES
# =============================================================================

MOCK_EMPTY_CONTENT_RESPONSE = {
    "summary": {
        "short": "No content available.",
        "medium": "The provided content appears to be empty or contains no meaningful information.",
        "long": "The provided content appears to be empty or contains no meaningful information that can be summarized."
    },
    "tags": {
        "categories": ["Unknown"],
        "keywords": [],
        "entities": []
    },
    "chapters": [],
    "highlights": []
}

MOCK_VERY_LONG_CONTENT_RESPONSE = {
    "summary": {
        "short": "An extensive multi-hour discussion covering numerous topics in great detail.",
        "medium": "This comprehensive content spans multiple hours and covers a wide range of topics including technical concepts, "
                 "practical applications, historical context, and future implications. The discussion is thorough and detailed, "
                 "requiring significant time investment to fully absorb all the information presented.",
        "long": "This extensive content represents a comprehensive exploration of the subject matter, spanning multiple hours of "
               "detailed discussion. The material is organized into numerous segments, each addressing specific aspects of the "
               "broader topic. Throughout the content, speakers provide in-depth analysis, practical examples, and theoretical "
               "frameworks that build upon one another to create a complete understanding. The length and depth of coverage "
               "make this suitable for serious students of the subject who are willing to invest significant time in mastering "
               "the material. Key themes recur throughout, with each iteration adding new layers of understanding and nuance."
    },
    "tags": {
        "categories": ["Education", "In-Depth Analysis", "Comprehensive Study"],
        "keywords": ["detailed", "comprehensive", "extensive", "thorough", "multi-part"],
        "entities": []
    }
}

MOCK_SPECIAL_CHARACTERS_RESPONSE = {
    "summary": {
        "short": "Content with special characters: émojis 🎉, symbols ©®™, and unicode ñ.",
        "medium": "This content demonstrates handling of various special characters including accented letters (é, ñ, ü), "
                 "symbols (©, ®, ™, €, £), emojis (🎉, 🚀, 💡), and other unicode characters. Proper encoding ensures "
                 "these are preserved correctly.",
        "long": "This comprehensive test of special character handling includes a wide variety of unicode characters that "
               "commonly appear in real-world content. Accented letters from various languages (café, niño, über) ensure "
               "international content is handled correctly. Mathematical symbols (∑, ∫, √, π) and currency symbols (€, £, ¥) "
               "test numeric and financial content. Emojis (🎉, 🚀, 💡, ❤️) represent modern communication styles. "
               "Proper handling of these characters is essential for global content processing."
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_mock_enrichment_by_type(enrichment_type: str) -> Dict[str, Any]:
    """Get mock response for a specific enrichment type.
    
    Args:
        enrichment_type: One of 'summary', 'tags', 'chapters', 'highlights'
        
    Returns:
        Mock response data for the specified type
    """
    type_map = {
        "summary": MOCK_SUMMARY_RESPONSE,
        "tags": MOCK_TAG_RESPONSE,
        "chapters": MOCK_CHAPTERS_RESPONSE,
        "highlights": MOCK_HIGHLIGHTS_RESPONSE
    }
    return type_map.get(enrichment_type, {})


def get_mock_enrichment_multiple_types(types: List[str]) -> Dict[str, Any]:
    """Get mock enrichment with multiple types.
    
    Args:
        types: List of enrichment types to include
        
    Returns:
        Complete enrichment response with specified types
    """
    result = {
        "enrichment_version": "v1",
        "metadata": {
            "provider": "openai",
            "model": "gpt-4-turbo",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cost_usd": 0.15 * len(types),
            "tokens_used": 2000 * len(types),
            "enrichment_types": types,
            "cache_hit": False
        }
    }
    
    for enrichment_type in types:
        if enrichment_type == "summary":
            result["summary"] = MOCK_SUMMARY_RESPONSE
        elif enrichment_type == "tags":
            result["tags"] = MOCK_TAG_RESPONSE
        elif enrichment_type == "chapters":
            result["chapters"] = MOCK_CHAPTERS_RESPONSE
        elif enrichment_type == "highlights":
            result["highlights"] = MOCK_HIGHLIGHTS_RESPONSE
    
    return result


def get_mock_error_response(error_type: str) -> Dict[str, Any]:
    """Get mock error response.
    
    Args:
        error_type: One of 'rate_limit', 'auth', 'timeout', 'malformed_json', 'invalid_schema'
        
    Returns:
        Mock error response
    """
    error_map = {
        "rate_limit": MOCK_RATE_LIMIT_ERROR,
        "auth": MOCK_AUTH_ERROR,
        "timeout": MOCK_TIMEOUT_ERROR,
        "malformed_json": MOCK_MALFORMED_JSON_RESPONSE,
        "invalid_schema": MOCK_INVALID_SCHEMA_RESPONSE
    }
    return error_map.get(error_type, {})
