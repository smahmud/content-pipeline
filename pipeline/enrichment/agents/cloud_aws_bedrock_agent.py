"""
Cloud AWS Bedrock LLM Agent

Cloud-based integration with AWS Bedrock service to access multiple LLM models
(Claude, Titan, etc.) through AWS's managed AI service.
Handles AWS credential management and Bedrock-specific API formatting.

File naming follows pattern: cloud_{provider}_{service}_agent.py
Provider ID: cloud-aws-bedrock
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from pipeline.enrichment.agents.base import BaseLLMAgent, LLMRequest, LLMResponse
from pipeline.enrichment.errors import (
    LLMProviderError,
    AuthenticationError,
    RateLimitError,
)


@dataclass
class CloudAWSBedrockAgentConfig:
    """Configuration for Cloud AWS Bedrock agent.
    
    Attributes:
        region: AWS region (default: us-east-1)
        default_model: Default model ID (default: anthropic.claude-v2)
        access_key_id: AWS access key ID (optional, uses default credentials if not provided)
        secret_access_key: AWS secret access key (optional)
        session_token: AWS session token for temporary credentials (optional)
        max_tokens: Maximum tokens for completion (default: 4096)
        temperature: Sampling temperature (default: 0.3)
        timeout: Request timeout in seconds (default: 60)
    """
    region: str = "us-east-1"
    default_model: str = "anthropic.claude-3-haiku-20240307-v1:0"  # Updated to Claude 3 Haiku
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 60


class CloudAWSBedrockAgent(BaseLLMAgent):
    """Cloud AWS Bedrock LLM agent implementation.
    
    Supports multiple LLM models (Claude, Titan, etc.) through AWS Bedrock service.
    Handles AWS credential management and converts Bedrock response formats.
    
    Deployment: Cloud (requires AWS credentials and internet connection)
    Provider: AWS
    Service: Bedrock (gateway to multiple models)
    Access Method: AWS API Gateway
    """
    
    # Model context windows (in tokens)
    CONTEXT_WINDOWS = {
        "anthropic.claude-v2": 100000,
        "anthropic.claude-v2:1": 100000,
        "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
        "anthropic.claude-3-haiku-20240307-v1:0": 200000,
        "anthropic.claude-3-opus-20240229-v1:0": 200000,
        "anthropic.claude-instant-v1": 100000,
        "amazon.titan-text-express-v1": 8000,
        "amazon.titan-text-lite-v1": 4000,
    }
    
    # Pricing per 1K tokens (input, output) in USD
    PRICING = {
        "anthropic.claude-v2": {"input_per_1k": 0.008, "output_per_1k": 0.024},
        "anthropic.claude-v2:1": {"input_per_1k": 0.008, "output_per_1k": 0.024},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input_per_1k": 0.003, "output_per_1k": 0.015},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
        "anthropic.claude-3-opus-20240229-v1:0": {"input_per_1k": 0.015, "output_per_1k": 0.075},
        "anthropic.claude-instant-v1": {"input_per_1k": 0.0008, "output_per_1k": 0.0024},
        "amazon.titan-text-express-v1": {"input_per_1k": 0.0008, "output_per_1k": 0.0016},
        "amazon.titan-text-lite-v1": {"input_per_1k": 0.0003, "output_per_1k": 0.0004},
    }
    
    def __init__(self, config: CloudAWSBedrockAgentConfig):
        """Initialize Cloud AWS Bedrock agent.
        
        Args:
            config: Cloud AWS Bedrock agent configuration
            
        Raises:
            ImportError: If boto3 is not installed
            AuthenticationError: If AWS credentials are invalid
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for AWS Bedrock agent. "
                "Install it with: pip install boto3"
            )
        
        self.config = config
        self.client = self._create_client()
    
    def _create_client(self):
        """Create boto3 Bedrock client with credentials.
        
        Returns:
            boto3 Bedrock runtime client
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        try:
            # Build session kwargs
            session_kwargs = {}
            if self.config.access_key_id:
                session_kwargs['aws_access_key_id'] = self.config.access_key_id
            if self.config.secret_access_key:
                session_kwargs['aws_secret_access_key'] = self.config.secret_access_key
            if self.config.session_token:
                session_kwargs['aws_session_token'] = self.config.session_token
            
            # Create session
            if session_kwargs:
                session = boto3.Session(**session_kwargs)
            else:
                # Use default credentials (environment, IAM role, etc.)
                session = boto3.Session()
            
            # Create Bedrock runtime client
            client = session.client(
                'bedrock-runtime',
                region_name=self.config.region
            )
            
            return client
        
        except Exception as e:
            raise AuthenticationError(
                f"Failed to create AWS Bedrock client: {e}"
            )
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from Bedrock model.
        
        Args:
            request: Standardized LLM request
            
        Returns:
            Standardized LLM response
            
        Raises:
            LLMProviderError: If API call fails
            RateLimitError: If rate limit is exceeded
        """
        model = request.model or self.config.default_model
        
        # Build request body based on model family
        if model.startswith("anthropic.claude"):
            body = self._build_claude_request(request, model)
        elif model.startswith("amazon.titan"):
            body = self._build_titan_request(request, model)
        else:
            raise LLMProviderError(f"Unsupported Bedrock model: {model}")
        
        try:
            # Invoke model
            response = self.client.invoke_model(
                modelId=model,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract content based on model family
            if model.startswith("anthropic.claude"):
                content, tokens_used = self._parse_claude_response(response_body)
            elif model.startswith("amazon.titan"):
                content, tokens_used = self._parse_titan_response(response_body)
            else:
                raise LLMProviderError(f"Unknown model family: {model}")
            
            # Calculate cost
            cost = self._calculate_cost(model, request.prompt, tokens_used)
            
            return LLMResponse(
                content=content,
                model_used=model,
                tokens_used=tokens_used,
                cost_usd=cost,
                metadata={
                    "provider": "bedrock",
                    "region": self.config.region,
                }
            )
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'ThrottlingException':
                raise RateLimitError(f"Bedrock rate limit exceeded: {error_message}")
            elif error_code in ['AccessDeniedException', 'UnauthorizedException']:
                raise AuthenticationError(f"Bedrock authentication failed: {error_message}")
            else:
                raise LLMProviderError(f"Bedrock API error: {error_message}")
        
        except Exception as e:
            raise LLMProviderError(f"Unexpected Bedrock error: {e}")
    
    def _build_claude_request(self, request: LLMRequest, model: str) -> Dict[str, Any]:
        """Build request body for Claude models.
        
        Args:
            request: LLM request
            model: Model ID
            
        Returns:
            Request body dict
        """
        # Claude 3 uses messages API, older versions use prompt
        if "claude-3" in model:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ]
            }
        else:
            # Claude v2 and earlier
            return {
                "prompt": f"\n\nHuman: {request.prompt}\n\nAssistant:",
                "max_tokens_to_sample": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature,
                "stop_sequences": ["\n\nHuman:"]
            }
    
    def _build_titan_request(self, request: LLMRequest, model: str) -> Dict[str, Any]:
        """Build request body for Titan models.
        
        Args:
            request: LLM request
            model: Model ID
            
        Returns:
            Request body dict
        """
        return {
            "inputText": request.prompt,
            "textGenerationConfig": {
                "maxTokenCount": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature,
                "topP": 0.9,
            }
        }
    
    def _parse_claude_response(self, response_body: Dict[str, Any]) -> tuple[str, int]:
        """Parse Claude model response.
        
        Args:
            response_body: Response body dict
            
        Returns:
            Tuple of (content, tokens_used)
        """
        # Claude 3 format
        if "content" in response_body:
            content = response_body["content"][0]["text"]
            tokens_used = response_body.get("usage", {}).get("input_tokens", 0) + \
                         response_body.get("usage", {}).get("output_tokens", 0)
        # Claude v2 format
        else:
            content = response_body.get("completion", "")
            # Estimate tokens if not provided
            tokens_used = len(content.split()) * 1.3  # Rough estimate
        
        return content, int(tokens_used)
    
    def _parse_titan_response(self, response_body: Dict[str, Any]) -> tuple[str, int]:
        """Parse Titan model response.
        
        Args:
            response_body: Response body dict
            
        Returns:
            Tuple of (content, tokens_used)
        """
        results = response_body.get("results", [])
        if not results:
            return "", 0
        
        content = results[0].get("outputText", "")
        tokens_used = response_body.get("inputTextTokenCount", 0) + \
                     results[0].get("tokenCount", 0)
        
        return content, tokens_used
    
    def _calculate_cost(self, model: str, prompt: str, tokens_used: int) -> float:
        """Calculate cost for the request.
        
        Args:
            model: Model ID
            prompt: Input prompt
            tokens_used: Total tokens used
            
        Returns:
            Cost in USD
        """
        pricing = self.PRICING.get(model)
        if not pricing:
            return 0.0
        
        # Estimate input tokens (rough approximation)
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = max(0, tokens_used - input_tokens)
        
        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
        
        return input_cost + output_cost
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for a request.
        
        Args:
            request: LLM request
            
        Returns:
            Estimated cost in USD
        """
        model = request.model or self.config.default_model
        pricing = self.PRICING.get(model)
        
        if not pricing:
            return 0.0
        
        # Estimate input tokens
        input_tokens = len(request.prompt.split()) * 1.3
        expected_output_tokens = request.max_tokens or self.config.max_tokens
        
        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (expected_output_tokens / 1000) * pricing["output_per_1k"]
        
        return input_cost + output_cost
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Bedrock agent capabilities.
        
        Returns:
            Capabilities dict
        """
        return {
            "provider": "cloud-aws-bedrock",
            "supported_models": list(self.CONTEXT_WINDOWS.keys()),
            "supports_streaming": False,
            "supports_function_calling": False,
            "max_context_window": max(self.CONTEXT_WINDOWS.values()),
        }
    
    def validate_requirements(self) -> bool:
        """Check if Bedrock is available and credentials are valid.
        
        Returns:
            True if Bedrock is available
        """
        if not BOTO3_AVAILABLE:
            return False
        
        try:
            # Try to list available models as a credential check
            self.client.list_foundation_models()
            return True
        except Exception:
            return False
    
    def get_context_window(self, model: str) -> int:
        """Get context window size for model.
        
        Args:
            model: Model ID
            
        Returns:
            Context window size in tokens
        """
        return self.CONTEXT_WINDOWS.get(model, 100000)
