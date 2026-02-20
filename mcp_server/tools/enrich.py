"""Enrich tool — wraps the enrich CLI command."""

import json
from typing import Optional


async def enrich(
    input_path: str,
    provider: str = "auto",
    summarize: bool = True,
    tag: bool = True,
    chapterize: bool = True,
    highlight: bool = True,
    output_path: Optional[str] = None,
) -> dict:
    """Generate semantic enrichment from transcript.

    Args:
        input_path: Path to transcript JSON file.
        provider: LLM provider (auto, cloud-aws-bedrock, cloud-openai, cloud-anthropic, local-ollama).
        summarize: Generate summary enrichment.
        tag: Generate tag enrichment.
        chapterize: Generate chapter enrichment.
        highlight: Generate highlight enrichment.
        output_path: Output enriched JSON file path.

    Returns:
        Dict with success status and enrichment details.
    """
    try:
        from pipeline.enrichment.orchestrator import EnrichmentOrchestrator
        from pipeline.llm import LLMProviderFactory, LLMConfig

        llm_config = LLMConfig.load_from_yaml(".content-pipeline/config.yaml")
        provider_factory = LLMProviderFactory(llm_config)

        enrichment_types = []
        if summarize:
            enrichment_types.append("summary")
        if tag:
            enrichment_types.append("tag")
        if chapterize:
            enrichment_types.append("chapter")
        if highlight:
            enrichment_types.append("highlight")

        orchestrator = EnrichmentOrchestrator(
            provider_factory=provider_factory,
            default_provider=provider if provider != "auto" else None,
        )

        result = orchestrator.enrich(
            input_path=input_path,
            enrichment_types=enrichment_types,
            output_path=output_path,
        )

        return {
            "success": True,
            "output_path": result.get("output_path", output_path),
            "enrichment_types": enrichment_types,
            "provider": provider,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
