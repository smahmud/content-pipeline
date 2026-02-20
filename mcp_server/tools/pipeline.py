"""Pipeline tool — convenience tool that chains extract → transcribe → enrich → format."""

from typing import List, Optional

from mcp_server.tools.extract import extract
from mcp_server.tools.transcribe import transcribe
from mcp_server.tools.enrich import enrich
from mcp_server.tools.format import format_content


async def run_pipeline(
    input_url: str,
    output_types: Optional[List[str]] = None,
    platform: Optional[str] = None,
) -> dict:
    """Run full pipeline: extract → transcribe → enrich → format.

    Args:
        input_url: YouTube URL or local file path.
        output_types: List of output format types (default: ["blog"]).
        platform: Target platform for format validation.

    Returns:
        Dict with results from each pipeline stage.
    """
    if output_types is None:
        output_types = ["blog"]

    results = {"stages": {}}

    # 1. Extract
    extract_result = await extract(source=input_url)
    results["stages"]["extract"] = extract_result
    if not extract_result.get("success"):
        results["success"] = False
        results["error"] = f"Extract failed: {extract_result.get('error')}"
        return results

    audio_path = extract_result["output_path"]

    # 2. Transcribe
    transcribe_result = await transcribe(input_path=audio_path)
    results["stages"]["transcribe"] = transcribe_result
    if not transcribe_result.get("success"):
        results["success"] = False
        results["error"] = f"Transcribe failed: {transcribe_result.get('error')}"
        return results

    transcript_path = transcribe_result.get("output_path", audio_path.replace(".mp3", ".json"))

    # 3. Enrich
    enrich_result = await enrich(input_path=transcript_path)
    results["stages"]["enrich"] = enrich_result
    if not enrich_result.get("success"):
        results["success"] = False
        results["error"] = f"Enrich failed: {enrich_result.get('error')}"
        return results

    enriched_path = enrich_result.get("output_path", transcript_path.replace(".json", "-enriched.json"))

    # 4. Format (for each output type)
    format_results = {}
    for otype in output_types:
        fmt_result = await format_content(
            input_path=enriched_path,
            output_type=otype,
            platform=platform,
        )
        format_results[otype] = fmt_result

    results["stages"]["format"] = format_results
    results["success"] = True
    results["output_types"] = output_types
    return results
