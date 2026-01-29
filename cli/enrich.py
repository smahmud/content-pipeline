"""
Enrich Subcommand Module

This module implements the enrich subcommand for the Content Pipeline CLI.
New in v0.7.0 to support:
- LLM-powered enrichment (summaries, tags, chapters, highlights)
- Multiple LLM providers (OpenAI, AWS Bedrock, Claude, Ollama)
- Cost estimation and control
- Intelligent caching
- Batch processing
"""

import os
import sys
import logging
import json
import click
from pathlib import Path
from typing import Optional, List

from pipeline.enrichment.agents.factory import AgentFactory, AutoSelectionConfig
from pipeline.enrichment.agents.openai_agent import OpenAIAgentConfig
from pipeline.enrichment.agents.ollama_agent import OllamaAgentConfig
from pipeline.enrichment.agents.bedrock_agent import BedrockAgentConfig
from pipeline.enrichment.orchestrator import EnrichmentOrchestrator, EnrichmentRequest, DryRunReport
from pipeline.enrichment.prompts.loader import PromptLoader
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
    type=click.Choice(["openai", "bedrock", "claude", "ollama", "auto"]),
    default="auto",
    help="LLM provider to use (default: auto)"
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
    help="Enable all enrichment types"
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
    log_level
):
    """
    Enrich transcripts with LLM-powered semantic analysis.
    
    Generates summaries, tags, chapters, and highlights using
    multiple LLM providers with cost control and caching.
    
    Examples:
        # Generate all enrichments with auto provider selection
        content-pipeline enrich --input transcript.json --all
        
        # Generate only summary and tags with OpenAI
        content-pipeline enrich --input transcript.json --provider openai --summarize --tag
        
        # Estimate costs without executing
        content-pipeline enrich --input transcript.json --all --dry-run
        
        # Set cost limit
        content-pipeline enrich --input transcript.json --all --max-cost 1.00
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Content Pipeline v0.7.0 - Enrich Command")
    logger.debug(f"CLI arguments: input={input}, provider={provider}, model={model}")
    
    try:
        # Step 1: Load transcript
        logger.info(f"Loading transcript from {input}")
        transcript_data = _load_transcript(input)
        
        # Step 2: Determine enrichment types
        enrichment_types = _determine_enrichment_types(
            summarize, tag, chapterize, highlight, all_types
        )
        
        if not enrichment_types:
            click.echo("Error: No enrichment types specified. Use --all or specify individual types.")
            click.echo("Run 'content-pipeline enrich --help' for usage information.")
            sys.exit(1)
        
        logger.info(f"Enrichment types: {', '.join(enrichment_types)}")
        
        # Step 3: Create agent factory
        logger.info(f"Initializing LLM provider: {provider}")
        agent_factory = _create_agent_factory()
        
        # Step 4: Create orchestrator
        prompt_loader = PromptLoader(custom_prompts_dir=custom_prompts)
        orchestrator = EnrichmentOrchestrator(
            agent_factory=agent_factory,
            prompt_loader=prompt_loader
        )
        
        # Step 5: Create enrichment request
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
        
        # Step 6: Execute enrichment
        logger.info("Starting enrichment...")
        result = orchestrator.enrich(request)
        
        # Step 7: Handle result
        if isinstance(result, DryRunReport):
            _display_dry_run_report(result)
        else:
            # Determine output path
            output_path = output or _generate_output_path(input)
            
            # Save result
            logger.info(f"Saving enriched result to {output_path}")
            _save_enrichment(result, output_path)
            
            # Display summary
            _display_success_summary(result, output_path)
        
        logger.info("Enrichment completed successfully")
        
    except CostLimitExceededError as e:
        click.echo(f"\n❌ Cost Limit Exceeded: {e}", err=True)
        sys.exit(1)
    
    except ConfigurationError as e:
        click.echo(f"\n❌ Configuration Error: {e}", err=True)
        click.echo("\nSetup instructions:")
        click.echo("  - OpenAI: Set OPENAI_API_KEY environment variable")
        click.echo("  - Bedrock: Configure AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        click.echo("  - Claude: Set ANTHROPIC_API_KEY environment variable")
        click.echo("  - Ollama: Start local service with 'ollama serve'")
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
        Transcript data dict
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def _create_agent_factory() -> AgentFactory:
    """Create agent factory with default configurations.
    
    Returns:
        Configured agent factory
    """
    # Create configs from environment
    openai_config = OpenAIAgentConfig(
        api_key=os.getenv("OPENAI_API_KEY", "")
    )
    
    ollama_config = OllamaAgentConfig()
    
    bedrock_config = BedrockAgentConfig(
        region=os.getenv("AWS_REGION", "us-east-1"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        session_token=os.getenv("AWS_SESSION_TOKEN")
    )
    
    auto_selection = AutoSelectionConfig(
        priority_order=["openai", "claude", "bedrock", "ollama"],
        fallback_enabled=True
    )
    
    return AgentFactory(
        openai_config=openai_config,
        ollama_config=ollama_config,
        bedrock_config=bedrock_config,
        auto_selection=auto_selection
    )


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
    
    # Convert to dict and save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enrichment.dict(), f, indent=2, ensure_ascii=False)


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
    click.echo("✓ Enrichment Completed Successfully")
    click.echo("="*60)
    click.echo(f"\nProvider: {metadata.provider}")
    click.echo(f"Model: {metadata.model}")
    click.echo(f"Cost: ${metadata.cost_usd:.4f}")
    click.echo(f"Tokens used: {metadata.tokens_used:,}")
    
    click.echo(f"\nOutput saved to: {output_path}")
    click.echo("="*60 + "\n")
