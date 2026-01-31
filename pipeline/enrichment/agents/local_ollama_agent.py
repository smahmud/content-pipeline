"""
Local Ollama LLM Agent

Local deployment adapter for Ollama models (llama2, mistral, codellama, etc.).
Handles HTTP API calls to local Ollama service, service health checks,
and model availability detection. Returns zero cost for all operations.

File naming follows pattern: local_{provider}_agent.py
Provider ID: local-ollama
"""

import json
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

from pipeline.enrichment.agents.base import BaseLLMAgent, LLMRequest, LLMResponse
from pipeline.enrichment.errors import LLMProviderError


@dataclass
class LocalOllamaAgentConfig:
    """Configuration for Local Ollama agent.
    
    Attributes:
        base_url: Base URL for Ollama API (default: http://localhost:11434)
        default_model: Default model to use if not specified in request
        max_tokens: Default maximum tokens for responses
        temperature: Default sampling temperature
        timeout: Request timeout in seconds (local models may be slower)
    """
    base_url: str = "http://localhost:11434"
    default_model: str = "llama2"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 120  # Local models may be slower


class LocalOllamaAgent(BaseLLMAgent):
    """Local Ollama LLM provider adapter.
    
    This agent integrates with a local Ollama service to provide LLM capabilities
    without any API costs. It supports various open-source models like llama2,
    mistral, codellama, and others that can be run locally.
    
    Deployment: Local (runs on user's machine)
    Provider: Ollama
    Access Method: Local HTTP server
    
    Key features:
    - Zero cost (local processing)
    - Privacy (no data sent to external services)
    - Service health checks
    - Model availability detection
    """
    
    # Approximate context window sizes for common models (in tokens)
    # These are estimates since Ollama models vary
    CONTEXT_WINDOWS = {
        "llama2": 4096,
        "llama2:7b": 4096,
        "llama2:13b": 4096,
        "llama2:70b": 4096,
        "mistral": 8192,
        "mistral:7b": 8192,
        "codellama": 4096,
        "codellama:7b": 4096,
        "codellama:13b": 4096,
        "phi": 2048,
        "neural-chat": 4096,
    }
    
    def __init__(self, config: LocalOllamaAgentConfig):
        """Initialize Local Ollama agent.
        
        Args:
            config: Configuration for the Local Ollama agent
        """
        self.config = config
    
    def _make_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to Ollama API.
        
        Args:
            endpoint: API endpoint (e.g., "/api/generate")
            data: Request payload (for POST requests)
            
        Returns:
            Response JSON as dictionary
            
        Raises:
            LLMProviderError: If the request fails
        """
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            if data:
                response = requests.post(
                    url,
                    json=data,
                    timeout=self.config.timeout
                )
            else:
                response = requests.get(
                    url,
                    timeout=self.config.timeout
                )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise LLMProviderError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise LLMProviderError(
                f"Ollama request timed out after {self.config.timeout}s"
            )
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(f"Ollama API request failed: {str(e)}")
    
    def _count_tokens_approximate(self, text: str) -> int:
        """Approximate token count for text.
        
        Since Ollama doesn't provide a tokenizer API, we use a rough
        approximation: 1 token ≈ 4 characters.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        return len(text) // 4
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from Ollama.
        
        Args:
            request: Standardized LLM request
            
        Returns:
            Standardized LLM response with content and metadata
            
        Raises:
            LLMProviderError: If the API call fails
        """
        model = request.model or self.config.default_model
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": request.prompt,
            "stream": False,  # We want the complete response
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            }
        }
        
        # Add any additional metadata
        if request.metadata:
            payload["options"].update(request.metadata)
        
        # Make API call
        response_data = self._make_request("/api/generate", payload)
        
        # Extract response content
        content = response_data.get("response", "")
        
        # Calculate approximate token usage
        input_tokens = self._count_tokens_approximate(request.prompt)
        output_tokens = self._count_tokens_approximate(content)
        total_tokens = input_tokens + output_tokens
        
        return LLMResponse(
            content=content,
            model_used=model,
            tokens_used=total_tokens,
            cost_usd=0.0,  # Local models are free
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "eval_count": response_data.get("eval_count", 0),
                "eval_duration": response_data.get("eval_duration", 0),
                "load_duration": response_data.get("load_duration", 0),
            }
        )
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for the request.
        
        Args:
            request: The LLM request to estimate cost for
            
        Returns:
            Always returns 0.0 (local models are free)
        """
        return 0.0
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return Ollama provider capabilities.
        
        Returns:
            Dictionary with provider capabilities
        """
        # Try to get list of available models
        available_models = []
        try:
            response = self._make_request("/api/tags")
            available_models = [
                model["name"] for model in response.get("models", [])
            ]
        except LLMProviderError:
            # If we can't connect, return empty list
            pass
        
        return {
            "provider": "local-ollama",
            "supported_models": available_models or list(self.CONTEXT_WINDOWS.keys()),
            "max_tokens": max(self.CONTEXT_WINDOWS.values()),
            "supports_streaming": True,
            "supports_functions": False,
            "supports_vision": False,
            "cost_per_token": 0.0,
        }
    
    def validate_requirements(self) -> bool:
        """Check if Ollama is available.
        
        Returns:
            True if Ollama service is running and accessible, False otherwise
        """
        try:
            # Ping the Ollama service
            self._make_request("/api/tags")
            return True
        except LLMProviderError:
            return False
    
    def get_context_window(self, model: str) -> int:
        """Return maximum context window size for the model.
        
        Args:
            model: The model identifier
            
        Returns:
            Maximum context window size in tokens (approximate)
        """
        # Try exact match first
        if model in self.CONTEXT_WINDOWS:
            return self.CONTEXT_WINDOWS[model]
        
        # Try base model name (e.g., "llama2:7b" -> "llama2")
        base_model = model.split(":")[0]
        if base_model in self.CONTEXT_WINDOWS:
            return self.CONTEXT_WINDOWS[base_model]
        
        # Default to 4096 for unknown models
        return 4096
