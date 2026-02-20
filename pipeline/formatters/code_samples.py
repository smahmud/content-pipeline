"""
Code sample generator for technical content.

Generates relevant code samples based on content topics
for embedding in blogs, tutorials, and technical documentation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeSample:
    """A generated code sample.
    
    Attributes:
        code: The actual code content
        language: Programming language identifier
        description: Brief explanation of what the code does
        position_hint: Where to insert in content
    """
    code: str
    language: str
    description: str
    position_hint: str


@dataclass
class CodeSamplesResult:
    """Collection of code samples for content.
    
    Attributes:
        samples: List of generated code samples
        is_technical: Whether the content was detected as technical
        detected_topics: Topics that influenced code generation
        detected_languages: Languages detected from content
    """
    samples: list[CodeSample]
    is_technical: bool
    detected_topics: list[str] = field(default_factory=list)
    detected_languages: list[str] = field(default_factory=list)


# Keywords that indicate technical content
TECHNICAL_INDICATORS = [
    # Programming general
    "programming", "code", "coding", "developer", "software",
    "function", "class", "method", "variable", "algorithm",
    
    # Web/API
    "api", "rest", "graphql", "endpoint", "http", "request",
    "frontend", "backend", "fullstack", "web development",
    
    # CLI/DevOps
    "cli", "command", "command-line", "terminal", "shell",
    "bash", "script", "automation", "devops", "docker",
    "kubernetes", "ci/cd", "deployment",
    
    # Data
    "database", "sql", "query", "data", "json", "xml",
    
    # Languages
    "python", "javascript", "typescript", "java", "go",
    "rust", "c++", "ruby", "php", "swift", "kotlin",
    
    # Frameworks
    "react", "vue", "angular", "django", "flask", "fastapi",
    "express", "node", "spring", "rails",
]

# Map topics to programming languages
LANGUAGE_MAPPINGS = {
    # CLI/Shell
    "cli": "bash",
    "command-line": "bash",
    "command": "bash",
    "terminal": "bash",
    "shell": "bash",
    "bash": "bash",
    "devops": "bash",
    "docker": "bash",
    "kubernetes": "yaml",
    
    # Web Frontend
    "web": "typescript",
    "frontend": "typescript",
    "react": "typescript",
    "vue": "typescript",
    "angular": "typescript",
    "javascript": "javascript",
    "typescript": "typescript",
    
    # Backend
    "backend": "python",
    "api": "python",
    "rest": "python",
    "fastapi": "python",
    "django": "python",
    "flask": "python",
    "python": "python",
    
    # Data
    "database": "sql",
    "sql": "sql",
    "query": "sql",
    
    # Other languages
    "java": "java",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "ruby": "ruby",
    "rails": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "node": "javascript",
    "express": "javascript",
}

# Output types that support code samples
SUPPORTED_OUTPUT_TYPES = {"blog", "tutorial", "newsletter", "slides"}

# Output types that do NOT support code samples
UNSUPPORTED_OUTPUT_TYPES = {
    "tweet", "linkedin", "youtube", "chapters", "transcript-clean",
    "meeting-minutes", "podcast-notes", "notion", "obsidian",
    "quote-cards", "video-script", "tiktok-script", "ai-video-script", "seo"
}


class CodeSampleGenerator:
    """Generates code samples for technical content.
    
    Analyzes content to determine if it's technical, detects
    appropriate programming languages, and generates relevant
    code samples.
    """
    
    def __init__(self, llm_enhancer=None):
        """Initialize the code sample generator.
        
        Args:
            llm_enhancer: Optional LLM enhancer for advanced code generation
        """
        self.llm_enhancer = llm_enhancer
    
    def is_supported(self, output_type: str) -> bool:
        """Check if output type supports code samples.
        
        Args:
            output_type: The format output type
            
        Returns:
            True if code samples are supported
        """
        return output_type in SUPPORTED_OUTPUT_TYPES
    
    def is_technical_content(self, enriched_content: dict) -> bool:
        """Determine if content is technical.
        
        Analyzes topics, tags, and summary for technical indicators.
        
        Args:
            enriched_content: The enriched content dictionary
            
        Returns:
            True if content appears to be technical
        """
        # Gather all text to analyze
        text_parts = []
        
        # Add topics
        topics = enriched_content.get("topics", [])
        if isinstance(topics, list):
            text_parts.extend([t.lower() for t in topics if isinstance(t, str)])
        
        # Add tags
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            text_parts.extend([t.lower() for t in tags.get("primary", []) if isinstance(t, str)])
            text_parts.extend([t.lower() for t in tags.get("secondary", []) if isinstance(t, str)])
        elif isinstance(tags, list):
            text_parts.extend([t.lower() for t in tags if isinstance(t, str)])
        
        # Add summary
        summary = enriched_content.get("summary", {})
        if isinstance(summary, dict):
            for key in ["short", "medium", "long"]:
                if key in summary and isinstance(summary[key], str):
                    text_parts.append(summary[key].lower())
        elif isinstance(summary, str):
            text_parts.append(summary.lower())
        
        # Add title
        metadata = enriched_content.get("metadata", {})
        title = metadata.get("title", "")
        if title:
            text_parts.append(title.lower())
        
        # Check for technical indicators
        combined_text = " ".join(text_parts)
        
        for indicator in TECHNICAL_INDICATORS:
            if indicator.lower() in combined_text:
                return True
        
        return False
    
    def detect_languages(self, enriched_content: dict) -> list[str]:
        """Detect appropriate programming languages for content.
        
        Args:
            enriched_content: The enriched content dictionary
            
        Returns:
            List of detected programming languages
        """
        detected: set[str] = set()
        
        # Gather all text to analyze
        text_parts = []
        
        # Add topics
        topics = enriched_content.get("topics", [])
        if isinstance(topics, list):
            text_parts.extend([t.lower() for t in topics if isinstance(t, str)])
        
        # Add tags
        tags = enriched_content.get("tags", [])
        if isinstance(tags, dict):
            text_parts.extend([t.lower() for t in tags.get("primary", []) if isinstance(t, str)])
        elif isinstance(tags, list):
            text_parts.extend([t.lower() for t in tags if isinstance(t, str)])
        
        # Add title
        metadata = enriched_content.get("metadata", {})
        title = metadata.get("title", "")
        if title:
            text_parts.append(title.lower())
        
        combined_text = " ".join(text_parts)
        
        # Check for language mappings
        for keyword, language in LANGUAGE_MAPPINGS.items():
            if keyword in combined_text:
                detected.add(language)
        
        # Default to Python if no specific language detected but content is technical
        if not detected and self.is_technical_content(enriched_content):
            detected.add("python")
        
        return list(detected)
    
    def generate(
        self,
        enriched_content: dict,
        output_type: str,
    ) -> CodeSamplesResult:
        """Generate code samples for content.
        
        Args:
            enriched_content: The enriched content dictionary
            output_type: Target output format type
            
        Returns:
            CodeSamplesResult with generated samples
        """
        # Check if output type supports code
        if not self.is_supported(output_type):
            logger.warning(f"Code samples not supported for output type: {output_type}")
            return CodeSamplesResult(
                samples=[],
                is_technical=False,
            )
        
        # Check if content is technical
        is_technical = self.is_technical_content(enriched_content)
        
        if not is_technical:
            logger.warning("Content is non-technical, skipping code generation")
            return CodeSamplesResult(
                samples=[],
                is_technical=False,
            )
        
        # Detect languages
        languages = self.detect_languages(enriched_content)
        
        # Extract topics for context
        topics = enriched_content.get("topics", [])
        if isinstance(topics, list):
            topics = [t for t in topics if isinstance(t, str)][:5]
        else:
            topics = []
        
        # Generate samples
        samples = self._generate_samples(
            languages=languages,
            topics=topics,
            enriched_content=enriched_content,
        )
        
        return CodeSamplesResult(
            samples=samples,
            is_technical=True,
            detected_topics=topics,
            detected_languages=languages,
        )
    
    def _generate_samples(
        self,
        languages: list[str],
        topics: list[str],
        enriched_content: dict,
    ) -> list[CodeSample]:
        """Generate code samples based on detected languages and topics.
        
        Args:
            languages: Detected programming languages
            topics: Content topics
            enriched_content: The enriched content
            
        Returns:
            List of CodeSample objects
        """
        samples: list[CodeSample] = []
        topic_str = ", ".join(topics[:3]) if topics else "the topic"
        
        # Generate a sample for each detected language (max 3)
        for i, language in enumerate(languages[:3]):
            sample = self._generate_sample_for_language(
                language=language,
                topic_str=topic_str,
                position=i + 1,
            )
            if sample:
                samples.append(sample)
        
        return samples
    
    def _generate_sample_for_language(
        self,
        language: str,
        topic_str: str,
        position: int,
    ) -> Optional[CodeSample]:
        """Generate a code sample for a specific language.
        
        Args:
            language: Programming language
            topic_str: Topic description
            position: Position in content
            
        Returns:
            CodeSample or None
        """
        # Template-based generation (LLM enhancement would replace this)
        templates = {
            "python": self._python_template,
            "javascript": self._javascript_template,
            "typescript": self._typescript_template,
            "bash": self._bash_template,
            "sql": self._sql_template,
            "go": self._go_template,
            "java": self._java_template,
            "yaml": self._yaml_template,
        }
        
        template_func = templates.get(language, self._generic_template)
        return template_func(topic_str, position)
    
    def _python_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate Python code sample."""
        code = f'''# Example: Working with {topic_str}
def process_data(data: dict) -> dict:
    """
    Process and transform data related to {topic_str}.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Processed data dictionary
    """
    # Validate input
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Process the data
    result = {{
        "processed": True,
        "items": len(data),
    }}
    
    return result


# Usage example
if __name__ == "__main__":
    sample_data = {{"key": "value"}}
    result = process_data(sample_data)
    print(result)'''
        
        return CodeSample(
            code=code,
            language="python",
            description=f"Python example demonstrating {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def _javascript_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate JavaScript code sample."""
        code = f'''// Example: Working with {topic_str}

/**
 * Process and transform data
 * @param {{Object}} data - Input data object
 * @returns {{Object}} Processed data
 */
function processData(data) {{
    // Validate input
    if (!data || Object.keys(data).length === 0) {{
        throw new Error('Data cannot be empty');
    }}
    
    // Process the data
    return {{
        processed: true,
        items: Object.keys(data).length,
    }};
}}

// Usage example
const sampleData = {{ key: 'value' }};
const result = processData(sampleData);
console.log(result);'''
        
        return CodeSample(
            code=code,
            language="javascript",
            description=f"JavaScript example demonstrating {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def _typescript_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate TypeScript code sample."""
        code = f'''// Example: Working with {topic_str}

interface DataInput {{
    [key: string]: unknown;
}}

interface ProcessedResult {{
    processed: boolean;
    items: number;
}}

/**
 * Process and transform data
 */
function processData(data: DataInput): ProcessedResult {{
    // Validate input
    if (!data || Object.keys(data).length === 0) {{
        throw new Error('Data cannot be empty');
    }}
    
    // Process the data
    return {{
        processed: true,
        items: Object.keys(data).length,
    }};
}}

// Usage example
const sampleData: DataInput = {{ key: 'value' }};
const result = processData(sampleData);
console.log(result);'''
        
        return CodeSample(
            code=code,
            language="typescript",
            description=f"TypeScript example demonstrating {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def _bash_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate Bash code sample."""
        code = f'''#!/bin/bash
# Example: Working with {topic_str}

# Function to process data
process_data() {{
    local input="$1"
    
    # Validate input
    if [ -z "$input" ]; then
        echo "Error: Input cannot be empty" >&2
        return 1
    fi
    
    # Process the data
    echo "Processing: $input"
    echo "Done!"
}}

# Usage example
process_data "sample_data"'''
        
        return CodeSample(
            code=code,
            language="bash",
            description=f"Bash script example for {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def _sql_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate SQL code sample."""
        code = f'''-- Example: Working with {topic_str}

-- Create a sample table
CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO items (name) VALUES ('Sample Item');

-- Query the data
SELECT 
    id,
    name,
    created_at
FROM items
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY created_at DESC;'''
        
        return CodeSample(
            code=code,
            language="sql",
            description=f"SQL example for {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def _go_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate Go code sample."""
        code = f'''// Example: Working with {topic_str}
package main

import (
    "fmt"
    "errors"
)

// ProcessData processes and transforms data
func ProcessData(data map[string]interface{{}}) (map[string]interface{{}}, error) {{
    // Validate input
    if len(data) == 0 {{
        return nil, errors.New("data cannot be empty")
    }}
    
    // Process the data
    result := map[string]interface{{}}{{
        "processed": true,
        "items":     len(data),
    }}
    
    return result, nil
}}

func main() {{
    sampleData := map[string]interface{{}}{{"key": "value"}}
    result, err := ProcessData(sampleData)
    if err != nil {{
        fmt.Println("Error:", err)
        return
    }}
    fmt.Println(result)
}}'''
        
        return CodeSample(
            code=code,
            language="go",
            description=f"Go example demonstrating {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def _java_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate Java code sample."""
        code = f'''// Example: Working with {topic_str}
import java.util.HashMap;
import java.util.Map;

public class DataProcessor {{
    
    /**
     * Process and transform data
     * @param data Input data map
     * @return Processed data map
     */
    public static Map<String, Object> processData(Map<String, Object> data) {{
        // Validate input
        if (data == null || data.isEmpty()) {{
            throw new IllegalArgumentException("Data cannot be empty");
        }}
        
        // Process the data
        Map<String, Object> result = new HashMap<>();
        result.put("processed", true);
        result.put("items", data.size());
        
        return result;
    }}
    
    public static void main(String[] args) {{
        Map<String, Object> sampleData = new HashMap<>();
        sampleData.put("key", "value");
        
        Map<String, Object> result = processData(sampleData);
        System.out.println(result);
    }}
}}'''
        
        return CodeSample(
            code=code,
            language="java",
            description=f"Java example demonstrating {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def _yaml_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate YAML code sample."""
        code = f'''# Example: Configuration for {topic_str}
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  labels:
    app: example
data:
  # Application settings
  APP_NAME: "example-app"
  LOG_LEVEL: "info"
  
---
# Deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: example-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: example'''
        
        return CodeSample(
            code=code,
            language="yaml",
            description=f"YAML configuration example for {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def _generic_template(self, topic_str: str, position: int) -> CodeSample:
        """Generate generic code sample."""
        code = f'''// Example: Working with {topic_str}
// This is a placeholder code sample
// Replace with actual implementation

function example() {{
    // Your code here
    console.log("Processing {topic_str}");
}}

example();'''
        
        return CodeSample(
            code=code,
            language="javascript",
            description=f"Example code for {topic_str}",
            position_hint=f"code example {position}",
        )
    
    def format_for_markdown(self, sample: CodeSample) -> str:
        """Format a code sample for Markdown embedding.
        
        Args:
            sample: CodeSample to format
            
        Returns:
            Markdown-formatted code block
        """
        return f'''```{sample.language}
{sample.code}
```

*{sample.description}*'''
