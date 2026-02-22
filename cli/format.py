"""
Format Subcommand Module

This module implements the format subcommand for the Content Pipeline CLI.
Transforms enriched content into various output formats for publishing.

Features:
- 16 output types (blog, tweet, youtube, linkedin, etc.)
- LLM enhancement with style profiles
- Platform validation and truncation
- Bundle generation (named bundles like blog-launch, video-launch)
- Batch processing with glob patterns
- Cost estimation and control
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

from pipeline.formatters.base import (
    VALID_OUTPUT_TYPES,
    VALID_PLATFORMS,
    VALID_TONES,
    VALID_LENGTHS,
)
from pipeline.formatters.bundles.loader import BundleNotFoundError
from pipeline.formatters.composer import FormatComposer
from pipeline.formatters.errors import (
    FormatError,
    InputValidationError,
    CostLimitExceededError,
)
from pipeline.formatters.writer import OutputWriter


logger = logging.getLogger(__name__)


@click.command(help="Format enriched content for publishing")
@click.option(
    "--input", "-i",
    "input_path",
    required=False,
    type=click.Path(exists=True),
    help="Path to enriched JSON file"
)
@click.option(
    "--output", "-o",
    "output_path",
    type=click.Path(),
    help="Output file path (default: <input>_<type>.md)"
)
@click.option(
    "--type", "-t",
    "output_type",
    type=click.Choice(VALID_OUTPUT_TYPES, case_sensitive=False),
    help="Output format type (e.g., blog, tweet, linkedin)"
)
@click.option(
    "--platform", "-p",
    type=click.Choice(VALID_PLATFORMS, case_sensitive=False),
    help="Target platform for validation (e.g., twitter, medium)"
)
@click.option(
    "--style-profile",
    type=click.Path(exists=True),
    help="Path to style profile Markdown file"
)
@click.option(
    "--tone",
    type=click.Choice(VALID_TONES, case_sensitive=False),
    help="Tone override (professional, casual, technical, friendly)"
)
@click.option(
    "--length",
    type=click.Choice(VALID_LENGTHS, case_sensitive=False),
    help="Length override (short, medium, long)"
)
@click.option(
    "--llm-enhance/--no-llm",
    default=True,
    help="Enable/disable LLM enhancement (default: enabled)"
)
@click.option(
    "--provider",
    type=click.Choice(["auto", "cloud-aws-bedrock", "cloud-openai", "cloud-anthropic", "local-ollama"], case_sensitive=False),
    default="auto",
    help="LLM provider (default: auto)"
)
@click.option(
    "--model",
    type=str,
    help="Specific LLM model to use"
)
@click.option(
    "--max-cost",
    type=float,
    help="Maximum cost limit in USD"
)
@click.option(
    "--url",
    type=str,
    help="URL to include in promotional content (e.g., link to blog/linkedin post for tweet promotion)"
)
@click.option(
    "--batch",
    type=str,
    help="Glob pattern for batch processing (e.g., '*.enriched.json')"
)
@click.option(
    "--bundle",
    type=str,
    help="Named bundle to generate (e.g., blog-launch, video-launch)"
)
@click.option(
    "--bundles-config",
    type=click.Path(exists=True),
    help="Path to custom bundles YAML configuration"
)
@click.option(
    "--list-bundles",
    is_flag=True,
    help="List available bundles and exit"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Estimate costs without execution"
)
@click.option(
    "--sources",
    type=click.Path(exists=True, file_okay=False),
    help="Path to folder containing multiple source files to combine"
)
@click.option(
    "--image-prompts",
    is_flag=True,
    help="Generate AI image prompts alongside formatted output"
)
@click.option(
    "--include-code",
    is_flag=True,
    help="Generate code samples for technical content"
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Output directory for batch/bundle processing"
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files without prompting"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level"
)
def format(
    input_path: Optional[str],
    output_path: Optional[str],
    output_type: Optional[str],
    platform: Optional[str],
    style_profile: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    llm_enhance: bool,
    provider: str,
    model: Optional[str],
    max_cost: Optional[float],
    url: Optional[str],
    batch: Optional[str],
    bundle: Optional[str],
    bundles_config: Optional[str],
    list_bundles: bool,
    dry_run: bool,
    sources: Optional[str],
    image_prompts: bool,
    include_code: bool,
    output_dir: Optional[str],
    force: bool,
    log_level: str,
):
    """
    Format enriched content into various output types for publishing.
    
    Transforms EnrichmentV1 JSON files into formatted content like blog posts,
    tweets, YouTube descriptions, LinkedIn posts, and more.
    
    Examples:
        # Generate a blog post
        content-pipeline format --input transcript-enriched.json --type blog
        
        # Generate with LLM enhancement and style profile
        content-pipeline format --input enriched.json --type linkedin --style-profile profiles/linkedin.md
        
        # Generate a bundle of outputs
        content-pipeline format --input enriched.json --bundle blog-launch --output-dir ./outputs
        
        # Combine multiple sources into a blog post
        content-pipeline format --sources ./enriched-files/ --type blog
        
        # Generate with image prompts
        content-pipeline format --input enriched.json --type blog --image-prompts
        
        # Generate with code samples for technical content
        content-pipeline format --input enriched.json --type blog --include-code
        
        # Generate AI video script
        content-pipeline format --input enriched.json --type ai-video-script --platform youtube
        
        # Batch process multiple files
        content-pipeline format --batch "*.enriched.json" --type tweet --output-dir ./tweets
        
        # List available bundles
        content-pipeline format --list-bundles
        
        # Dry run to estimate costs
        content-pipeline format --input enriched.json --type blog --llm-enhance --dry-run
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Content Pipeline - Format Command")
    
    try:
        # Initialize bundle loader first (needed for --list-bundles)
        bundle_loader = _create_bundle_loader(bundles_config)
        
        # Handle --list-bundles early (doesn't need LLM)
        if list_bundles:
            composer = FormatComposer(bundle_loader=bundle_loader)
            _display_bundle_list(composer)
            return
        
        # Initialize LLM enhancer if LLM enhancement is enabled
        llm_enhancer_instance = None
        if llm_enhance:
            from pipeline.formatters.llm.enhancer import LLMEnhancer
            from pipeline.llm import LLMProviderFactory, LLMConfig
            
            # Create provider factory with configuration
            llm_config = LLMConfig.load_from_yaml('.content-pipeline/config.yaml')
            provider_factory = LLMProviderFactory(llm_config)
            llm_enhancer_instance = LLMEnhancer(
                provider_factory=provider_factory,
                default_provider=provider if provider != "auto" else "auto"
            )
        
        # Initialize composer with optional custom bundles config and LLM enhancer
        composer = FormatComposer(
            bundle_loader=bundle_loader,
            llm_enhancer=llm_enhancer_instance
        )
        
        # Initialize output writer
        writer = OutputWriter(force_overwrite=force)
        
        # Validate required options based on mode
        if sources:
            # Sources mode - combine multiple files
            if not output_type:
                click.echo("Error: --type is required for sources mode", err=True)
                sys.exit(1)
            _run_sources_mode(
                composer=composer,
                writer=writer,
                sources_folder=sources,
                output_path=output_path,
                output_dir=output_dir,
                output_type=output_type,
                platform=platform,
                style_profile=style_profile,
                tone=tone,
                length=length,
                llm_enhance=llm_enhance,
                provider=provider,
                model=model,
                max_cost=max_cost,
                dry_run=dry_run,
                force=force,
                url=url,
                image_prompts=image_prompts,
                include_code=include_code,
            )
        elif batch:
            # Batch mode requires --type
            if not output_type:
                click.echo("Error: --type is required for batch processing", err=True)
                sys.exit(1)
            _run_batch_mode(
                composer=composer,
                writer=writer,
                batch_pattern=batch,
                output_type=output_type,
                output_dir=output_dir,
                platform=platform,
                style_profile=style_profile,
                tone=tone,
                length=length,
                llm_enhance=llm_enhance,
                provider=provider,
                model=model,
                max_cost=max_cost,
                dry_run=dry_run,
            )
        elif bundle:
            # Bundle mode requires --input
            if not input_path:
                click.echo("Error: --input is required for bundle generation", err=True)
                sys.exit(1)
            _run_bundle_mode(
                composer=composer,
                writer=writer,
                input_path=input_path,
                bundle_name=bundle,
                output_dir=output_dir,
                style_profile=style_profile,
                tone=tone,
                length=length,
                llm_enhance=llm_enhance,
                provider=provider,
                model=model,
                max_cost=max_cost,
                dry_run=dry_run,
                force=force,
            )
        else:
            # Single format mode requires --input and --type
            if not input_path:
                click.echo("Error: --input is required", err=True)
                click.echo("Run 'content-pipeline format --help' for usage", err=True)
                sys.exit(1)
            if not output_type:
                click.echo("Error: --type is required for single format generation", err=True)
                click.echo("Run 'content-pipeline format --help' for usage", err=True)
                sys.exit(1)
            _run_single_mode(
                composer=composer,
                writer=writer,
                input_path=input_path,
                output_path=output_path,
                output_dir=output_dir,
                output_type=output_type,
                platform=platform,
                style_profile=style_profile,
                tone=tone,
                length=length,
                llm_enhance=llm_enhance,
                provider=provider,
                model=model,
                max_cost=max_cost,
                dry_run=dry_run,
                force=force,
                url=url,
                image_prompts=image_prompts,
                include_code=include_code,
            )
        
    except BundleNotFoundError as e:
        click.echo(f"\n❌ Bundle Error: {e}", err=True)
        click.echo("\nUse --list-bundles to see available bundles", err=True)
        sys.exit(1)
    
    except CostLimitExceededError as e:
        click.echo(f"\n❌ Cost Limit Exceeded: {e}", err=True)
        click.echo("\nUse --dry-run to estimate costs before execution", err=True)
        sys.exit(1)
    
    except InputValidationError as e:
        click.echo(f"\n❌ Input Validation Error: {e}", err=True)
        sys.exit(1)
    
    except FormatError as e:
        click.echo(f"\n❌ Format Error: {e}", err=True)
        logger.exception("Format operation failed")
        sys.exit(1)
    
    except FileNotFoundError as e:
        click.echo(f"\n❌ File Not Found: {e}", err=True)
        sys.exit(1)
    
    except json.JSONDecodeError as e:
        click.echo(f"\n❌ Invalid JSON: {e}", err=True)
        sys.exit(1)
    
    except Exception as e:
        click.echo(f"\n❌ Unexpected Error: {e}", err=True)
        logger.exception("Unexpected error during format operation")
        sys.exit(1)


def _create_bundle_loader(bundles_config: Optional[str]):
    """Create a BundleLoader with optional custom config."""
    from pipeline.formatters.bundles.loader import BundleLoader
    return BundleLoader(custom_bundles_path=bundles_config)


def _load_enriched_content(input_path: str) -> dict:
    """Load enriched content from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_style_profile(style_profile_path: Optional[str]) -> Optional[dict]:
    """Load and parse a style profile if specified."""
    if not style_profile_path:
        return None
    
    from pipeline.formatters.style_profile import StyleProfileLoader
    loader = StyleProfileLoader()
    profile = loader.load(style_profile_path)
    return profile.__dict__ if profile else None


def _display_bundle_list(composer: FormatComposer) -> None:
    """Display available bundles."""
    click.echo("\n" + composer.format_bundle_list())



def _run_single_mode(
    composer: FormatComposer,
    writer: OutputWriter,
    input_path: str,
    output_path: Optional[str],
    output_dir: Optional[str],
    output_type: str,
    platform: Optional[str],
    style_profile: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    llm_enhance: bool,
    provider: str,
    model: Optional[str],
    max_cost: Optional[float],
    dry_run: bool,
    force: bool,
    url: Optional[str] = None,
    image_prompts: bool = False,
    include_code: bool = False,
) -> None:
    """Run single format generation mode."""
    from pipeline.formatters.base import FormatRequest
    
    logger.info(f"Generating {output_type} from {input_path}")
    
    # Load input
    enriched_content = _load_enriched_content(input_path)
    
    # Load style profile if specified
    style_profile_data = _load_style_profile(style_profile)
    
    # Show progress for LLM enhancement
    if llm_enhance and not dry_run:
        click.echo(f"Generating {output_type}...")
        if style_profile:
            click.echo(f"  Using style profile: {style_profile}")
    
    # Create request
    request = FormatRequest(
        enriched_content=enriched_content,
        output_type=output_type,
        platform=platform,
        style_profile=style_profile_data,
        tone=tone,
        length=length,
        llm_enhance=llm_enhance,
        provider=provider,
        model=model,
        max_cost=max_cost,
        dry_run=dry_run,
        url=url,
    )
    
    # Handle dry run
    if dry_run:
        _display_dry_run(composer, request)
        return
    
    # Format content
    start_time = time.time()
    result = composer.format_single(request)
    elapsed = time.time() - start_time
    
    if not result.success:
        click.echo(f"\n❌ Format generation failed: {result.error}", err=True)
        sys.exit(1)
    
    # Write output
    write_result = writer.write(
        format_result=result,
        output_path=output_path,
        input_path=input_path,
        output_dir=output_dir,
        embed_metadata=True,
        force=force,
    )
    
    if not write_result.success:
        click.echo(f"\n❌ Failed to write output: {write_result.error}", err=True)
        sys.exit(1)
    
    # Generate image prompts if requested
    if image_prompts:
        _process_image_prompts(
            composer=composer,
            enriched_content=enriched_content,
            output_type=output_type,
            platform=platform,
            output_path=write_result.output_path,
            force=force,
        )
    
    # Generate code samples if requested
    if include_code:
        _process_code_samples(
            composer=composer,
            enriched_content=enriched_content,
            output_type=output_type,
        )
    
    # Display success summary
    _display_single_success(result, write_result.output_path, elapsed)


def _run_sources_mode(
    composer: FormatComposer,
    writer: OutputWriter,
    sources_folder: str,
    output_path: Optional[str],
    output_dir: Optional[str],
    output_type: str,
    platform: Optional[str],
    style_profile: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    llm_enhance: bool,
    provider: str,
    model: Optional[str],
    max_cost: Optional[float],
    dry_run: bool,
    force: bool,
    url: Optional[str] = None,
    image_prompts: bool = False,
    include_code: bool = False,
) -> None:
    """Run multi-source format generation mode."""
    from pathlib import Path
    
    sources_path = Path(sources_folder)
    logger.info(f"Generating {output_type} from sources in {sources_path}")
    
    click.echo(f"Loading sources from {sources_path}...")
    
    # Load style profile if specified
    style_profile_data = _load_style_profile(style_profile)
    
    # Format from sources
    start_time = time.time()
    result = composer.format_from_sources(
        sources_folder=sources_path,
        output_type=output_type,
        platform=platform,
        style_profile=style_profile_data,
        tone=tone,
        length=length,
        llm_enhance=llm_enhance,
        provider=provider,
        model=model,
        max_cost=max_cost,
        dry_run=dry_run,
        url=url,
    )
    elapsed = time.time() - start_time
    
    if not result.success:
        click.echo(f"\n❌ Format generation failed: {result.error}", err=True)
        sys.exit(1)
    
    # Determine output path for sources mode
    if not output_path:
        output_path = str(
            sources_path / f"combined_{output_type.replace('-', '_')}.md"
        )
    
    # Write output
    write_result = writer.write(
        format_result=result,
        output_path=output_path,
        input_path=str(sources_path),
        output_dir=output_dir,
        embed_metadata=True,
        force=force,
    )
    
    if not write_result.success:
        click.echo(f"\n❌ Failed to write output: {write_result.error}", err=True)
        sys.exit(1)
    
    # Generate image prompts if requested
    if image_prompts:
        _process_image_prompts(
            composer=composer,
            enriched_content=result.metadata.__dict__ if hasattr(result, 'metadata') else {},
            output_type=output_type,
            platform=platform,
            output_path=write_result.output_path,
            force=force,
        )
    
    # Display success
    source_count = len(result.warnings)  # Approximate from warnings
    click.echo(f"\n✅ Generated {output_type} from sources")
    click.echo(f"  Output: {write_result.output_path}")
    click.echo(f"  Time: {elapsed:.1f}s")
    if result.warnings:
        click.echo(f"  Warnings: {len(result.warnings)}")


def _process_image_prompts(
    composer: FormatComposer,
    enriched_content: dict,
    output_type: str,
    platform: Optional[str],
    output_path: str,
    force: bool,
) -> None:
    """Generate and write image prompts alongside main output."""
    prompts_result = composer.generate_image_prompts(
        enriched_content=enriched_content,
        output_type=output_type,
        platform=platform,
    )
    
    if prompts_result is None or not prompts_result.prompts:
        click.echo("  ℹ No image prompts generated (unsupported output type or no content)")
        return
    
    # Write image prompts to separate file
    from pipeline.formatters.image_prompts import ImagePromptGenerator
    prompts_path = ImagePromptGenerator().get_output_filename(output_path)
    
    import json
    from dataclasses import asdict
    prompts_data = asdict(prompts_result)
    
    from pathlib import Path
    Path(prompts_path).parent.mkdir(parents=True, exist_ok=True)
    
    if Path(prompts_path).exists() and not force:
        click.echo(f"  ⚠ Image prompts file exists: {prompts_path} (use --force to overwrite)")
        return
    
    with open(prompts_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=2)
    
    click.echo(f"  ✓ Image prompts: {prompts_path} ({len(prompts_result.prompts)} prompts)")


def _process_code_samples(
    composer: FormatComposer,
    enriched_content: dict,
    output_type: str,
) -> None:
    """Generate code samples and display info."""
    samples_result = composer.generate_code_samples(
        enriched_content=enriched_content,
        output_type=output_type,
    )
    
    if samples_result is None or not samples_result.samples:
        click.echo("  ℹ No code samples generated (non-technical content or unsupported type)")
        return
    
    click.echo(
        f"  ✓ Code samples: {len(samples_result.samples)} samples "
        f"({', '.join(s.language for s in samples_result.samples)})"
    )


def _run_bundle_mode(
    composer: FormatComposer,
    writer: OutputWriter,
    input_path: str,
    bundle_name: str,
    output_dir: Optional[str],
    style_profile: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    llm_enhance: bool,
    provider: str,
    model: Optional[str],
    max_cost: Optional[float],
    dry_run: bool,
    force: bool,
) -> None:
    """Run bundle generation mode."""
    logger.info(f"Generating bundle '{bundle_name}' from {input_path}")
    
    # Load input
    enriched_content = _load_enriched_content(input_path)
    
    # Get bundle info for progress display
    bundle = composer.bundle_loader.load_bundle(bundle_name)
    total_outputs = len(bundle.outputs)
    
    click.echo(f"\nGenerating bundle: {bundle_name}")
    click.echo(f"  Description: {bundle.description}")
    click.echo(f"  Outputs: {', '.join(bundle.outputs)}")
    click.echo("")
    
    # Handle dry run
    if dry_run:
        _display_bundle_dry_run(composer, enriched_content, bundle_name, llm_enhance, provider, model)
        return
    
    # Progress callback for bundle generation
    current_output = [0]  # Use list for closure
    
    def progress_callback(output_type: str, status: str):
        current_output[0] += 1
        click.echo(f"  [{current_output[0]}/{total_outputs}] {output_type}: {status}")
    
    # Generate bundle
    result = composer.format_bundle(
        bundle_name=bundle_name,
        enriched_content=enriched_content,
        output_dir=output_dir or ".",
        llm_enhance=llm_enhance,
        style_profile=style_profile,
        tone=tone,
        length=length,
        provider=provider,
        model=model,
        max_cost=max_cost,
        dry_run=False,
    )
    
    # Write outputs
    if result.results:
        write_results = writer.write_bundle_outputs(
            results=result.results,
            output_dir=output_dir or ".",
            input_path=input_path,
            embed_metadata=True,
            force=force,
        )
        
        # Show progress for each output
        for output_type in bundle.outputs:
            if output_type in result.successful:
                write_result = write_results.get(output_type)
                if write_result and write_result.success:
                    click.echo(f"  ✓ {output_type} -> {write_result.output_path}")
                else:
                    click.echo(f"  ✓ {output_type} (generated)")
            elif any(ot == output_type for ot, _ in result.failed):
                error = next((err for ot, err in result.failed if ot == output_type), "Unknown error")
                click.echo(f"  ✗ {output_type}: {error}")
    
    # Display summary
    _display_bundle_success(result, bundle_name)


def _run_batch_mode(
    composer: FormatComposer,
    writer: OutputWriter,
    batch_pattern: str,
    output_type: str,
    output_dir: Optional[str],
    platform: Optional[str],
    style_profile: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    llm_enhance: bool,
    provider: str,
    model: Optional[str],
    max_cost: Optional[float],
    dry_run: bool,
) -> None:
    """Run batch processing mode."""
    import glob
    
    logger.info(f"Batch processing: {batch_pattern} -> {output_type}")
    
    # Find matching files
    input_files = sorted(glob.glob(batch_pattern, recursive=True))
    
    if not input_files:
        click.echo(f"No files matching pattern: {batch_pattern}", err=True)
        sys.exit(1)
    
    total_files = len(input_files)
    click.echo(f"\nBatch processing {total_files} file(s)")
    click.echo(f"  Pattern: {batch_pattern}")
    click.echo(f"  Output type: {output_type}")
    click.echo("")
    
    # Handle dry run
    if dry_run:
        _display_batch_dry_run(composer, input_files, output_type, llm_enhance, provider, model, max_cost)
        return
    
    # Progress callback
    def progress_callback(current: int, total: int, file_path: str):
        click.echo(f"  [{current}/{total}] Processing: {Path(file_path).name}")
    
    # Run batch processing
    result = composer.format_batch(
        input_pattern=batch_pattern,
        output_type=output_type,
        output_dir=output_dir,
        llm_enhance=llm_enhance,
        style_profile=style_profile,
        tone=tone,
        length=length,
        platform=platform,
        provider=provider,
        model=model,
        max_cost=max_cost,
        progress_callback=progress_callback,
    )
    
    # Write outputs
    if result.results:
        for file_path, format_result in result.results.items():
            write_result = writer.write(
                format_result=format_result,
                input_path=file_path,
                output_dir=output_dir,
                embed_metadata=True,
                force=True,  # Batch mode always overwrites
            )
            if write_result.success:
                click.echo(f"    ✓ -> {write_result.output_path}")
    
    # Display summary
    _display_batch_success(result)


def _display_dry_run(composer: FormatComposer, request) -> None:
    """Display dry run cost estimate for single format."""
    from pipeline.formatters.base import FormatRequest
    
    estimate = composer.estimate_cost(request)
    
    click.echo("\n" + "="*60)
    click.echo("DRY RUN - Cost Estimate")
    click.echo("="*60)
    click.echo(f"\nOutput type: {request.output_type}")
    click.echo(f"LLM enhancement: {'enabled' if request.llm_enhance else 'disabled'}")
    
    if request.llm_enhance:
        click.echo(f"\nProvider: {estimate.provider}")
        click.echo(f"Model: {estimate.model}")
        click.echo(f"Estimated tokens: {estimate.estimated_tokens:,}")
        click.echo(f"Estimated cost: ${estimate.estimated_cost_usd:.4f}")
        
        if request.max_cost:
            click.echo(f"\nMax cost limit: ${request.max_cost:.4f}")
            if estimate.within_budget:
                click.echo("✓ Within budget")
            else:
                click.echo("✗ EXCEEDS BUDGET")
    else:
        click.echo("\nNo LLM costs (template-only generation)")
    
    click.echo("\n" + "="*60)
    click.echo("No API calls were made (dry-run mode)")
    click.echo("="*60 + "\n")


def _display_bundle_dry_run(
    composer: FormatComposer,
    enriched_content: dict,
    bundle_name: str,
    llm_enhance: bool,
    provider: str,
    model: Optional[str],
) -> None:
    """Display dry run cost estimate for bundle."""
    from pipeline.formatters.base import FormatRequest
    
    bundle = composer.bundle_loader.load_bundle(bundle_name)
    
    click.echo("\n" + "="*60)
    click.echo("DRY RUN - Bundle Cost Estimate")
    click.echo("="*60)
    click.echo(f"\nBundle: {bundle_name}")
    click.echo(f"Outputs: {', '.join(bundle.outputs)}")
    click.echo(f"LLM enhancement: {'enabled' if llm_enhance else 'disabled'}")
    
    if llm_enhance:
        total_cost = 0.0
        click.echo("\nPer-output estimates:")
        
        for output_type in bundle.outputs:
            request = FormatRequest(
                enriched_content=enriched_content,
                output_type=output_type,
                llm_enhance=True,
                provider=provider,
                model=model,
            )
            estimate = composer.estimate_cost(request)
            total_cost += estimate.estimated_cost_usd
            click.echo(f"  {output_type}: ${estimate.estimated_cost_usd:.4f}")
        
        click.echo(f"\nTotal estimated cost: ${total_cost:.4f}")
    else:
        click.echo("\nNo LLM costs (template-only generation)")
    
    click.echo("\n" + "="*60)
    click.echo("No API calls were made (dry-run mode)")
    click.echo("="*60 + "\n")


def _display_batch_dry_run(
    composer: FormatComposer,
    input_files: list,
    output_type: str,
    llm_enhance: bool,
    provider: str,
    model: Optional[str],
    max_cost: Optional[float],
) -> None:
    """Display dry run cost estimate for batch."""
    from pipeline.formatters.base import FormatRequest
    
    click.echo("\n" + "="*60)
    click.echo("DRY RUN - Batch Cost Estimate")
    click.echo("="*60)
    click.echo(f"\nFiles: {len(input_files)}")
    click.echo(f"Output type: {output_type}")
    click.echo(f"LLM enhancement: {'enabled' if llm_enhance else 'disabled'}")
    
    if llm_enhance:
        total_cost = 0.0
        
        for file_path in input_files[:5]:  # Show first 5 files
            try:
                enriched_content = _load_enriched_content(file_path)
                request = FormatRequest(
                    enriched_content=enriched_content,
                    output_type=output_type,
                    llm_enhance=True,
                    provider=provider,
                    model=model,
                )
                estimate = composer.estimate_cost(request)
                total_cost += estimate.estimated_cost_usd
                click.echo(f"  {Path(file_path).name}: ${estimate.estimated_cost_usd:.4f}")
            except Exception as e:
                click.echo(f"  {Path(file_path).name}: (error: {e})")
        
        if len(input_files) > 5:
            # Estimate remaining files based on average
            avg_cost = total_cost / min(5, len(input_files))
            remaining = len(input_files) - 5
            estimated_remaining = avg_cost * remaining
            total_cost += estimated_remaining
            click.echo(f"  ... and {remaining} more files (estimated: ${estimated_remaining:.4f})")
        
        click.echo(f"\nTotal estimated cost: ${total_cost:.4f}")
        
        if max_cost:
            click.echo(f"Max cost limit: ${max_cost:.4f}")
            if total_cost <= max_cost:
                click.echo("✓ Within budget")
            else:
                click.echo("✗ EXCEEDS BUDGET")
    else:
        click.echo("\nNo LLM costs (template-only generation)")
    
    click.echo("\n" + "="*60)
    click.echo("No API calls were made (dry-run mode)")
    click.echo("="*60 + "\n")


def _display_single_success(result, output_path: str, elapsed: float) -> None:
    """Display success summary for single format."""
    click.echo("\n" + "="*60)
    click.echo("Format Generation Complete")
    click.echo("="*60)
    click.echo(f"\nOutput: {output_path}")
    click.echo(f"Type: {result.metadata.output_type}")
    click.echo(f"Time: {elapsed:.2f}s")
    
    if result.metadata.llm_metadata and result.metadata.llm_metadata.enhanced:
        click.echo(f"\nLLM Enhancement:")
        click.echo(f"  Provider: {result.metadata.llm_metadata.provider}")
        click.echo(f"  Model: {result.metadata.llm_metadata.model}")
        click.echo(f"  Cost: ${result.metadata.llm_metadata.cost_usd:.4f}")
        click.echo(f"  Tokens: {result.metadata.llm_metadata.tokens_used:,}")
    
    if result.warnings:
        click.echo(f"\nWarnings:")
        for warning in result.warnings:
            click.echo(f"  ⚠️  {warning}")
    
    click.echo("="*60 + "\n")


def _display_bundle_success(result, bundle_name: str) -> None:
    """Display success summary for bundle generation."""
    click.echo("\n" + "="*60)
    click.echo("Bundle Generation Complete")
    click.echo("="*60)
    click.echo(f"\nBundle: {bundle_name}")
    click.echo(f"Successful: {len(result.successful)}")
    click.echo(f"Failed: {len(result.failed)}")
    click.echo(f"Total cost: ${result.total_cost:.4f}")
    click.echo(f"Total time: {result.total_time:.2f}s")
    
    if result.manifest_path:
        click.echo(f"\nManifest: {result.manifest_path}")
    
    if result.failed:
        click.echo(f"\nFailed outputs:")
        for output_type, error in result.failed:
            click.echo(f"  ✗ {output_type}: {error}")
    
    click.echo("="*60 + "\n")


def _display_batch_success(result) -> None:
    """Display success summary for batch processing."""
    click.echo("\n" + "="*60)
    click.echo("Batch Processing Complete")
    click.echo("="*60)
    click.echo(f"\nTotal files: {len(result.successful) + len(result.failed)}")
    click.echo(f"Successful: {len(result.successful)}")
    click.echo(f"Failed: {len(result.failed)}")
    click.echo(f"Total cost: ${result.total_cost:.4f}")
    click.echo(f"Total time: {result.total_time:.2f}s")
    
    if result.failed:
        click.echo(f"\nFailed files:")
        for file_path, error in result.failed:
            click.echo(f"  ✗ {Path(file_path).name}: {error}")
    
    click.echo("="*60 + "\n")
