"""
Enrich Subcommand Module

This module implements the enrich subcommand for the Content Pipeline CLI.
New in v0.7.0 to support:
- LLM-powered enrichment (summaries, tags, chapters, highlights)
- Multiple LLM providers (OpenAI, AWS Bedrock, Claude, Ollama)
- Cost estimation and control
- Intelligent caching
- Batch processing

v0.8.6 additions:
- Separate output files per enrichment type (default)
- --combine flag for backward compatibility (single file)
- --output-dir for specifying output directory
"""

import os
import sys
import logging
import json
import click
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

from pipeline.llm.factory import LLMProviderFactory
from pipeline.llm.config import OpenAIConfig, OllamaConfig, BedrockConfig
from pipeline.enrichment.orchestrator import EnrichmentOrchestrator, EnrichmentRequest, DryRunReport
from pipeline.enrichment.prompts.loader import PromptLoader
from pipeline.enrichment.schemas import IndividualEnrichment, extract_individual_enrichment
from pipeline.enrichment.errors import (
    EnrichmentError,
    CostLimitExceededError,
    ConfigurationError
)


@click.command(help="Enrich transcripts with LLM-powered analysis")
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to transcript JSON file"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Path for enriched output (default: <input>-enriched.json)"
)
@click.option(
    "--provider", "-p",
    type=click.Choice([
        # New provider IDs (preferred)
        "cloud-openai", "cloud-anthropic", "cloud-aws-bedrock", "local-ollama",
        # Legacy names (deprecated but supported)
        "openai", "claude", "bedrock", "ollama",
        # Auto-selection
        "auto"
    ]),
    default="auto",
    help="LLM provider to use (default: auto). Legacy names (openai, claude, bedrock, ollama) are deprecated."
)
@click.option(
    "--model", "-m",
    type=str,
    help="Specific model to use (overrides provider default)"
)
@click.option(
    "--summarize",
    is_flag=True,
    help="Generate summary (short, medium, long)"
)
@click.option(
    "--tag",
    is_flag=True,
    help="Extract tags (categories, keywords, entities)"
)
@click.option(
    "--chapterize",
    is_flag=True,
    help="Detect chapters with timestamps"
)
@click.option(
    "--highlight",
    is_flag=True,
    help="Identify key highlights"
)
@click.option(
    "--all",
    "all_types",
    is_flag=True,
    help="[DEPRECATED] Enable all enrichment types. Use separate flags (--summarize --tag --chapterize --highlight) instead for better reliability."
)
@click.option(
    "--max-cost",
    type=float,
    help="Maximum cost limit in USD"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Estimate costs without executing"
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Bypass cache and generate fresh results"
)
@click.option(
    "--custom-prompts",
    type=click.Path(exists=True),
    help="Directory with custom prompt templates"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
@click.option(
    "--combine",
    is_flag=True,
    help="Combine all enrichments into single output file (legacy behavior)"
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Directory for output files (default: same as input)"
)
def enrich(
    input,
    output,
    provider,
    model,
    summarize,
    tag,
    chapterize,
    highlight,
    all_types,
    max_cost,
    dry_run,
    no_cache,
    custom_prompts,
    log_level,
    combine,
    output_dir
):
    """
    Enrich transcripts with LLM-powered semantic analysis.
    
    Generates summaries, tags, chapters, and highlights using
    multiple LLM providers with cost control and caching.
    
    By default (v0.8.6+), each enrichment type is saved to a separate file.
    Use --combine to save all enrichments to a single file (legacy behavior).
    
    Examples:
        # Generate summary and tags (separate files by default)
        content-pipeline enrich --input transcript.json --summarize --tag
        # Creates: transcript-summary.json, transcript-tags.json
        
        # Combine into single file (legacy behavior)
        content-pipeline enrich --input transcript.json --summarize --tag --combine
        # Creates: transcript-enriched.json
        
        # Specify output directory
        content-pipeline enrich --input transcript.json --summarize --output-dir ./enriched/
        
        # Estimate costs without executing
        content-pipeline enrich --input transcript.json --summarize --tag --dry-run
        
        # Set cost limit
        content-pipeline enrich --input transcript.json --summarize --max-cost 1.00
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Content Pipeline v0.8.6 - Enrich Command")
    logger.debug(f"CLI arguments: input={input}, provider={provider}, model={model}")
    
    try:
        # Validate --output with multiple types without --combine
        enrichment_types = _determine_enrichment_types(
            summarize, tag, chapterize, highlight, all_types
        )
        
        if output and not combine and len(enrichment_types) > 1:
            click.echo("Error: --output cannot be used with multiple enrichment types.", err=True)
            click.echo("Use --output-dir instead, or add --combine for single file output.", err=True)
            sys.exit(1)
        
        # Step 1: Load transcript
        logger.info(f"Loading transcript from {input}")
        transcript_data = _load_transcript(input)
        
        # Step 2: Determine enrichment types (already done above for validation)
        if not enrichment_types:
            click.echo("Error: No enrichment types specified. Use --summarize, --tag, --chapterize, or --highlight.")
            click.echo("Run 'content-pipeline enrich --help' for usage information.")
            sys.exit(1)
        
        # Warn if using deprecated --all flag
        if all_types:
            click.echo("\n⚠️  WARNING: The --all flag is DEPRECATED and may be removed in a future release.", err=True)
            click.echo("⚠️  Reason: Unreliable with some providers (AWS Bedrock, OpenAI).", err=True)
            click.echo("⚠️  Recommendation: Use separate commands for each enrichment type:", err=True)
            click.echo("     content-pipeline enrich --input transcript.json --summarize", err=True)
            click.echo("     content-pipeline enrich --input transcript.json --tag", err=True)
            click.echo("     content-pipeline enrich --input transcript.json --chapterize", err=True)
            click.echo("     content-pipeline enrich --input transcript.json --highlight", err=True)
            click.echo("⚠️  Separate commands provide better reliability, cost control, and debugging.\n", err=True)
        
        logger.info(f"Enrichment types: {', '.join(enrichment_types)}")
        logger.info(f"Output mode: {'combined' if combine else 'separate files'}")
        
        # Step 3: Create provider factory
        logger.info(f"Initializing LLM provider: {provider}")
        provider_factory = _create_provider_factory()
        
        # Step 4: Create orchestrator
        prompt_loader = PromptLoader(custom_prompts_dir=custom_prompts)
        orchestrator = EnrichmentOrchestrator(
            provider_factory=provider_factory,
            prompt_loader=prompt_loader
        )
        
        # Step 5: Execute based on mode
        if combine or len(enrichment_types) == 1:
            # Combined mode or single type: existing behavior
            _execute_combined_mode(
                orchestrator=orchestrator,
                transcript_data=transcript_data,
                enrichment_types=enrichment_types,
                provider=provider,
                model=model,
                max_cost=max_cost,
                dry_run=dry_run,
                no_cache=no_cache,
                custom_prompts=custom_prompts,
                input_path=input,
                output_path=output,
                output_dir=output_dir,
                logger=logger
            )
        else:
            # Separate mode: new default behavior
            _execute_separate_mode(
                orchestrator=orchestrator,
                transcript_data=transcript_data,
                enrichment_types=enrichment_types,
                provider=provider,
                model=model,
                max_cost=max_cost,
                dry_run=dry_run,
                no_cache=no_cache,
                custom_prompts=custom_prompts,
                input_path=input,
                output_dir=output_dir,
                logger=logger
            )
        
        logger.info("Enrichment completed successfully")
        
    except CostLimitExceededError as e:
        click.echo(f"\n❌ Cost Limit Exceeded: {e}", err=True)
        sys.exit(1)
    
    except ConfigurationError as e:
        click.echo(f"\n❌ Configuration Error: {e}", err=True)
        click.echo("\nSetup instructions:")
        click.echo("  - cloud-openai: Set OPENAI_API_KEY environment variable")
        click.echo("  - cloud-aws-bedrock: Configure AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        click.echo("  - cloud-anthropic: Set ANTHROPIC_API_KEY environment variable")
        click.echo("  - local-ollama: Start local service with 'ollama serve'")
        sys.exit(1)
    
    except EnrichmentError as e:
        click.echo(f"\n❌ Enrichment Error: {e}", err=True)
        logger.exception("Enrichment failed")
        sys.exit(1)
    
    except Exception as e:
        click.echo(f"\n❌ Unexpected Error: {e}", err=True)
        logger.exception("Unexpected error during enrichment")
        sys.exit(1)


def _load_transcript(path: str) -> dict:
    """Load transcript from JSON file.
    
    Args:
        path: Path to transcript file
        
    Returns:
        Transcript data dict with 'text' field extracted from segments
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If transcript is in segments, combine into single text
    if 'transcript' in data and isinstance(data['transcript'], list):
        # Combine all segment texts
        text = ' '.join(segment.get('text', '').strip() for segment in data['transcript'])
        data['text'] = text
    
    return data


def _determine_enrichment_types(
    summarize: bool,
    tag: bool,
    chapterize: bool,
    highlight: bool,
    all_types: bool
) -> List[str]:
    """Determine which enrichment types to generate.
    
    Args:
        summarize: Generate summary
        tag: Extract tags
        chapterize: Detect chapters
        highlight: Identify highlights
        all_types: Enable all types
        
    Returns:
        List of enrichment type names
    """
    if all_types:
        return ["summary", "tag", "chapter", "highlight"]
    
    types = []
    if summarize:
        types.append("summary")
    if tag:
        types.append("tag")
    if chapterize:
        types.append("chapter")
    if highlight:
        types.append("highlight")
    
    return types


def _create_provider_factory() -> LLMProviderFactory:
    """Create provider factory with default configurations.
    
    Returns:
        Configured provider factory
    """
    # Create configs from environment
    openai_config = OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY", "")
    )
    
    ollama_config = OllamaConfig()
    
    bedrock_config = BedrockConfig(
        region=os.getenv("AWS_REGION", "us-east-1"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        session_token=os.getenv("AWS_SESSION_TOKEN")
    )
    
    # Create LLMConfig with all provider configs
    from pipeline.llm.config import LLMConfig
    llm_config = LLMConfig(
        openai=openai_config,
        ollama=ollama_config,
        bedrock=bedrock_config
    )
    
    return LLMProviderFactory(config=llm_config)


def _generate_output_path(input_path: str) -> str:
    """Generate output path from input path.
    
    Args:
        input_path: Input file path
        
    Returns:
        Output file path
    """
    path = Path(input_path)
    return str(path.parent / f"{path.stem}-enriched{path.suffix}")


def _save_enrichment(enrichment, output_path: str):
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


def _display_dry_run_report(report: DryRunReport):
    """Display dry-run cost estimate report.
    
    Args:
        report: Dry-run report
    """
    click.echo("\n" + "="*60)
    click.echo("DRY RUN - Cost Estimate")
    click.echo("="*60)
    click.echo(f"\nProvider: {report.provider}")
    click.echo(f"Model: {report.model}")
    click.echo(f"Enrichment types: {', '.join(report.enrichment_types)}")
    click.echo(f"\nEstimated cost: ${report.estimate.total_cost:.4f}")
    click.echo(f"  Input tokens: {report.estimate.input_tokens:,}")
    click.echo(f"  Output tokens: {report.estimate.output_tokens:,}")
    click.echo(f"  Input cost: ${report.estimate.input_cost:.4f}")
    click.echo(f"  Output cost: ${report.estimate.output_cost:.4f}")
    
    click.echo("\nBreakdown by enrichment type:")
    for etype, cost in report.estimate.breakdown.items():
        click.echo(f"  - {etype}: ${cost:.4f}")
    
    click.echo(f"\nCache: {'Would use cache' if report.would_use_cache else 'Cache disabled'}")
    if report.cache_hit:
        click.echo("  ✓ Result is cached (no API calls needed)")
    
    click.echo("\n" + "="*60)
    click.echo("No API calls were made (dry-run mode)")
    click.echo("="*60 + "\n")


def _display_success_summary(enrichment, output_path: str):
    """Display success summary.
    
    Args:
        enrichment: EnrichmentV1 result
        output_path: Output file path
    """
    metadata = enrichment.metadata
    
    click.echo("\n" + "="*60)
    click.echo("Enrichment Completed Successfully")
    click.echo("="*60)
    click.echo(f"\nProvider: {metadata.provider}")
    click.echo(f"Model: {metadata.model}")
    click.echo(f"Cost: ${metadata.cost_usd:.4f}")
    click.echo(f"Tokens used: {metadata.tokens_used:,}")
    
    click.echo(f"\nOutput saved to: {output_path}")
    click.echo("="*60 + "\n")


# ============================================================================
# v0.8.6 - Separate Output Files Support
# ============================================================================

def _generate_output_path_for_type(
    input_path: str,
    enrichment_type: str,
    output_dir: Optional[str] = None
) -> str:
    """Generate output path for individual enrichment type.
    
    Args:
        input_path: Input file path
        enrichment_type: Type of enrichment (summary, tag, chapter, highlight)
        output_dir: Optional output directory
        
    Returns:
        Output file path with type-specific suffix
        
    Examples:
        input: transcript.json, type: summary -> transcript-summary.json
        input: transcript.json, type: tag -> transcript-tags.json
    """
    path = Path(input_path)
    
    # Map enrichment type to file suffix (pluralize where appropriate)
    suffix_map = {
        "summary": "summary",
        "tag": "tags",
        "chapter": "chapters",
        "highlight": "highlights"
    }
    suffix = suffix_map.get(enrichment_type, enrichment_type)
    
    filename = f"{path.stem}-{suffix}.json"
    
    if output_dir:
        return str(Path(output_dir) / filename)
    return str(path.parent / filename)


def _execute_combined_mode(
    orchestrator: EnrichmentOrchestrator,
    transcript_data: dict,
    enrichment_types: List[str],
    provider: str,
    model: Optional[str],
    max_cost: Optional[float],
    dry_run: bool,
    no_cache: bool,
    custom_prompts: Optional[str],
    input_path: str,
    output_path: Optional[str],
    output_dir: Optional[str],
    logger: logging.Logger
):
    """Execute enrichment in combined mode (single output file).
    
    This is the legacy behavior and is used when:
    - --combine flag is specified
    - Only one enrichment type is requested
    """
    # Create enrichment request
    request = EnrichmentRequest(
        transcript_text=transcript_data.get("text", ""),
        language=transcript_data.get("metadata", {}).get("language", "en"),
        duration=transcript_data.get("metadata", {}).get("duration", 0.0),
        enrichment_types=enrichment_types,
        provider=provider,
        model=model,
        max_cost=max_cost,
        dry_run=dry_run,
        use_cache=not no_cache,
        custom_prompts_dir=custom_prompts
    )
    
    # Execute enrichment
    logger.info("Starting enrichment (combined mode)...")
    result = orchestrator.enrich(request)
    
    # Handle result
    if isinstance(result, DryRunReport):
        _display_dry_run_report(result)
    else:
        # Determine output path
        if output_path:
            final_output_path = output_path
        elif len(enrichment_types) == 1:
            # Single type: use type-specific filename
            final_output_path = _generate_output_path_for_type(
                input_path, enrichment_types[0], output_dir
            )
        elif output_dir:
            # Multiple types with --combine and --output-dir
            final_output_path = str(Path(output_dir) / f"{Path(input_path).stem}-enriched.json")
        else:
            # Multiple types with --combine: use -enriched suffix
            final_output_path = _generate_output_path(input_path)
        
        # Save result
        logger.info(f"Saving enriched result to {final_output_path}")
        _save_enrichment(result, final_output_path)
        
        # Display summary
        _display_success_summary(result, final_output_path)


def _execute_separate_mode(
    orchestrator: EnrichmentOrchestrator,
    transcript_data: dict,
    enrichment_types: List[str],
    provider: str,
    model: Optional[str],
    max_cost: Optional[float],
    dry_run: bool,
    no_cache: bool,
    custom_prompts: Optional[str],
    input_path: str,
    output_dir: Optional[str],
    logger: logging.Logger
):
    """Execute enrichment in separate mode (one file per type).
    
    This is the new default behavior (v0.8.6+). Each enrichment type
    is processed independently and saved to its own file.
    """
    results: Dict[str, Tuple[str, Any]] = {}
    
    for etype in enrichment_types:
        logger.info(f"Processing enrichment type: {etype}")
        
        try:
            # Create request for single type
            request = EnrichmentRequest(
                transcript_text=transcript_data.get("text", ""),
                language=transcript_data.get("metadata", {}).get("language", "en"),
                duration=transcript_data.get("metadata", {}).get("duration", 0.0),
                enrichment_types=[etype],
                provider=provider,
                model=model,
                max_cost=max_cost,
                dry_run=dry_run,
                use_cache=not no_cache,
                custom_prompts_dir=custom_prompts
            )
            
            # Execute enrichment
            result = orchestrator.enrich(request)
            
            if isinstance(result, DryRunReport):
                results[etype] = ("dry_run", result)
            else:
                # Generate output path for this type
                output_path = _generate_output_path_for_type(input_path, etype, output_dir)
                
                # Extract individual enrichment and save
                individual = extract_individual_enrichment(result, etype)
                _save_individual_enrichment(individual, output_path)
                
                results[etype] = ("success", {
                    "output_path": output_path,
                    "metadata": result.metadata
                })
                logger.info(f"Saved {etype} to {output_path}")
                
        except Exception as e:
            logger.error(f"Failed to process {etype}: {e}")
            results[etype] = ("failed", str(e))
    
    # Display summary
    if dry_run:
        _display_separate_dry_run_summary(results)
    else:
        _display_separate_summary(results)
    
    # Determine exit code
    failures = sum(1 for status, _ in results.values() if status == "failed")
    if failures == len(results):
        click.echo("\n❌ All enrichments failed", err=True)
        sys.exit(1)
    elif failures > 0:
        click.echo(f"\n⚠️  {failures} enrichment(s) failed", err=True)


def _save_individual_enrichment(enrichment: IndividualEnrichment, output_path: str):
    """Save individual enrichment result to file.
    
    Args:
        enrichment: IndividualEnrichment result
        output_path: Output file path
    """
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enrichment.model_dump(mode='json'), f, indent=2, ensure_ascii=False)


def _display_separate_summary(results: Dict[str, Tuple[str, Any]]):
    """Display summary for separate enrichment mode.
    
    Args:
        results: Dict mapping enrichment type to (status, data) tuple
    """
    click.echo("\n" + "="*60)
    click.echo("Enrichment Results (Separate Files)")
    click.echo("="*60)
    
    total_cost = 0.0
    total_tokens = 0
    
    for etype, (status, data) in results.items():
        if status == "success":
            click.echo(f"  ✓ {etype}: {data['output_path']}")
            total_cost += data['metadata'].cost_usd
            total_tokens += data['metadata'].tokens_used
        else:
            click.echo(f"  ✗ {etype}: FAILED - {data}")
    
    click.echo(f"\nTotal cost: ${total_cost:.4f}")
    click.echo(f"Total tokens: {total_tokens:,}")
    click.echo("="*60 + "\n")


def _display_separate_dry_run_summary(results: Dict[str, Tuple[str, Any]]):
    """Display dry-run summary for separate enrichment mode.
    
    Args:
        results: Dict mapping enrichment type to (status, data) tuple
    """
    click.echo("\n" + "="*60)
    click.echo("DRY RUN - Cost Estimate (Separate Files)")
    click.echo("="*60)
    
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for etype, (status, data) in results.items():
        if status == "dry_run":
            report: DryRunReport = data
            click.echo(f"\n{etype}:")
            click.echo(f"  Provider: {report.provider}")
            click.echo(f"  Model: {report.model}")
            click.echo(f"  Estimated cost: ${report.estimate.total_cost:.4f}")
            click.echo(f"  Input tokens: {report.estimate.input_tokens:,}")
            click.echo(f"  Output tokens: {report.estimate.output_tokens:,}")
            if report.cache_hit:
                click.echo("  ✓ Result is cached (no API calls needed)")
            
            total_cost += report.estimate.total_cost
            total_input_tokens += report.estimate.input_tokens
            total_output_tokens += report.estimate.output_tokens
        else:
            click.echo(f"\n{etype}: FAILED - {data}")
    
    click.echo(f"\n" + "-"*40)
    click.echo(f"Total estimated cost: ${total_cost:.4f}")
    click.echo(f"Total input tokens: {total_input_tokens:,}")
    click.echo(f"Total output tokens: {total_output_tokens:,}")
    click.echo("\n" + "="*60)
    click.echo("No API calls were made (dry-run mode)")
    click.echo("="*60 + "\n")
