"""
Batch Processor

Handles batch enrichment of multiple transcript files with progress tracking,
cost estimation, and error handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import glob
import json
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from pipeline.enrichment.orchestrator import EnrichmentOrchestrator, EnrichmentRequest
from pipeline.enrichment.cost_estimator import CostEstimator
from pipeline.enrichment.errors import EnrichmentError, CostLimitExceededError


@dataclass
class BatchResult:
    """Result for a single file in batch processing.
    
    Attributes:
        input_path: Input file path
        output_path: Output file path
        success: Whether enrichment succeeded
        cost: Cost in USD (0.0 if failed)
        tokens: Tokens used (0 if failed)
        duration: Processing time in seconds
        error: Error message if failed
    """
    input_path: str
    output_path: str
    success: bool
    cost: float = 0.0
    tokens: int = 0
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class BatchReport:
    """Summary report for batch processing.
    
    Attributes:
        total_files: Total number of files processed
        successful: Number of successful enrichments
        failed: Number of failed enrichments
        total_cost: Total cost across all files
        total_tokens: Total tokens used
        total_duration: Total processing time in seconds
        results: List of individual file results
    """
    total_files: int
    successful: int
    failed: int
    total_cost: float
    total_tokens: int
    total_duration: float
    results: List[BatchResult]


class BatchProcessor:
    """Processes multiple transcript files in batch.
    
    This class handles:
    - Glob pattern matching for file discovery
    - Pre-flight cost estimation across all files
    - Progress tracking with tqdm
    - Individual file error handling
    - Summary report generation
    
    Example:
        >>> processor = BatchProcessor(orchestrator)
        >>> report = processor.process_batch(
        ...     pattern="transcripts/*.json",
        ...     enrichment_types=["summary", "tag"],
        ...     provider="openai"
        ... )
    """
    
    def __init__(self, orchestrator: EnrichmentOrchestrator):
        """Initialize batch processor.
        
        Args:
            orchestrator: Enrichment orchestrator to use
        """
        self.orchestrator = orchestrator
    
    def process_batch(
        self,
        pattern: str,
        enrichment_types: List[str],
        provider: str,
        output_dir: Optional[str] = None,
        model: Optional[str] = None,
        max_cost: Optional[float] = None,
        use_cache: bool = True,
        custom_prompts_dir: Optional[str] = None
    ) -> BatchReport:
        """Process multiple files in batch.
        
        Args:
            pattern: Glob pattern for input files
            enrichment_types: List of enrichment types
            provider: LLM provider to use
            output_dir: Optional output directory (default: same as input)
            model: Optional specific model
            max_cost: Optional total cost limit for entire batch
            use_cache: Whether to use caching
            custom_prompts_dir: Optional custom prompts directory
            
        Returns:
            Batch processing report
            
        Raises:
            CostLimitExceededError: If total estimated cost exceeds max_cost
        """
        # Find matching files
        input_files = self._find_files(pattern)
        
        if not input_files:
            raise ValueError(f"No files found matching pattern: {pattern}")
        
        # Estimate total cost if max_cost is specified
        if max_cost:
            total_estimate = self._estimate_batch_cost(
                input_files, enrichment_types, provider, model
            )
            
            if total_estimate > max_cost:
                raise CostLimitExceededError(
                    f"Estimated batch cost ${total_estimate:.4f} exceeds "
                    f"limit ${max_cost:.4f}"
                )
        
        # Process files
        results = []
        total_cost = 0.0
        total_tokens = 0
        start_time = time.time()
        
        # Use tqdm if available
        if TQDM_AVAILABLE:
            iterator = tqdm(input_files, desc="Processing files", unit="file")
        else:
            iterator = input_files
        
        for input_path in iterator:
            result = self._process_single_file(
                input_path=input_path,
                enrichment_types=enrichment_types,
                provider=provider,
                output_dir=output_dir,
                model=model,
                use_cache=use_cache,
                custom_prompts_dir=custom_prompts_dir
            )
            
            results.append(result)
            total_cost += result.cost
            total_tokens += result.tokens
        
        total_duration = time.time() - start_time
        
        # Generate report
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchReport(
            total_files=len(results),
            successful=successful,
            failed=failed,
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_duration=total_duration,
            results=results
        )
    
    def _find_files(self, pattern: str) -> List[str]:
        """Find files matching glob pattern.
        
        Args:
            pattern: Glob pattern
            
        Returns:
            List of file paths
        """
        return sorted(glob.glob(pattern))
    
    def _estimate_batch_cost(
        self,
        input_files: List[str],
        enrichment_types: List[str],
        provider: str,
        model: Optional[str]
    ) -> float:
        """Estimate total cost for batch processing.
        
        Args:
            input_files: List of input file paths
            enrichment_types: List of enrichment types
            provider: LLM provider
            model: Optional specific model
            
        Returns:
            Total estimated cost in USD
        """
        total_estimate = 0.0
        
        # Create provider for cost estimation
        provider_instance = self.orchestrator.provider_factory.create_provider(provider)
        cost_estimator = CostEstimator(provider_instance)
        
        for input_path in input_files:
            try:
                # Load transcript
                with open(input_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                
                # Estimate cost
                estimate = cost_estimator.estimate(
                    transcript_text=transcript_data.get("text", ""),
                    enrichment_types=enrichment_types,
                    model=model
                )
                
                total_estimate += estimate.total_cost
            
            except Exception:
                # If we can't estimate for a file, skip it
                # (will be handled during actual processing)
                continue
        
        return total_estimate
    
    def _process_single_file(
        self,
        input_path: str,
        enrichment_types: List[str],
        provider: str,
        output_dir: Optional[str],
        model: Optional[str],
        use_cache: bool,
        custom_prompts_dir: Optional[str]
    ) -> BatchResult:
        """Process a single file.
        
        Args:
            input_path: Input file path
            enrichment_types: List of enrichment types
            provider: LLM provider
            output_dir: Optional output directory
            model: Optional specific model
            use_cache: Whether to use caching
            custom_prompts_dir: Optional custom prompts directory
            
        Returns:
            Batch result for this file
        """
        start_time = time.time()
        
        try:
            # Load transcript
            with open(input_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Determine output path
            output_path = self._generate_output_path(input_path, output_dir)
            
            # Create enrichment request
            request = EnrichmentRequest(
                transcript_text=transcript_data.get("text", ""),
                language=transcript_data.get("metadata", {}).get("language", "en"),
                duration=transcript_data.get("metadata", {}).get("duration", 0.0),
                enrichment_types=enrichment_types,
                provider=provider,
                model=model,
                use_cache=use_cache,
                custom_prompts_dir=custom_prompts_dir
            )
            
            # Execute enrichment
            result = self.orchestrator.enrich(request)
            
            # Save result
            self._save_result(result, output_path)
            
            duration = time.time() - start_time
            
            return BatchResult(
                input_path=input_path,
                output_path=output_path,
                success=True,
                cost=result.metadata.cost_usd,
                tokens=result.metadata.tokens_used,
                duration=duration
            )
        
        except Exception as e:
            duration = time.time() - start_time
            
            return BatchResult(
                input_path=input_path,
                output_path="",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def _generate_output_path(
        self,
        input_path: str,
        output_dir: Optional[str]
    ) -> str:
        """Generate output path for enriched file.
        
        Args:
            input_path: Input file path
            output_dir: Optional output directory
            
        Returns:
            Output file path
        """
        input_file = Path(input_path)
        
        if output_dir:
            # Save to output directory with same filename
            output_path = Path(output_dir) / f"{input_file.stem}-enriched{input_file.suffix}"
        else:
            # Save next to input file
            output_path = input_file.parent / f"{input_file.stem}-enriched{input_file.suffix}"
        
        return str(output_path)
    
    def _save_result(self, enrichment, output_path: str):
        """Save enrichment result to file.
        
        Args:
            enrichment: EnrichmentV1 result
            output_path: Output file path
        """
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save (use mode='json' for proper datetime serialization)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enrichment.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
    
    def format_report(self, report: BatchReport) -> str:
        """Format batch report for display.
        
        Args:
            report: Batch report
            
        Returns:
            Formatted string
        """
        lines = [
            "="*60,
            "Batch Processing Report",
            "="*60,
            f"\nTotal files: {report.total_files}",
            f"Successful: {report.successful}",
            f"Failed: {report.failed}",
            f"\nTotal cost: ${report.total_cost:.4f}",
            f"Total tokens: {report.total_tokens:,}",
            f"Total duration: {report.total_duration:.1f}s",
            ""
        ]
        
        if report.failed > 0:
            lines.append("\nFailed files:")
            for result in report.results:
                if not result.success:
                    lines.append(f"  - {result.input_path}")
                    lines.append(f"    Error: {result.error}")
        
        lines.append("="*60)
        
        return "\n".join(lines)
