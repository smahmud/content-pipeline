"""
Full Pipeline Integration Test

End-to-end integration test covering the complete content pipeline workflow:
1. Extract audio from YouTube video
2. Transcribe audio to text
3. Enrich transcript with LLM-powered analysis

This test uses REAL external services and will incur costs:
- YouTube API (free)
- Whisper transcription (local or API)
- Claude API (costs money)

Run with: pytest tests/integration/test_full_pipeline.py -v -s --external --slow

Requirements:
- ANTHROPIC_API_KEY environment variable set
- ffmpeg installed (for audio extraction)
- whisper installed (for local transcription) OR OPENAI_API_KEY (for API transcription)
"""

import os
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Import pipeline components
from pipeline.extractors.youtube.extractor import YouTubeExtractor
from pipeline.transcription.factory import TranscriptionProviderFactory
from pipeline.transcribers.normalize import normalize_transcript_v1
from pipeline.transcribers.persistence import LocalFilePersistence
from pipeline.config.schema import TranscriptionConfig, EngineType
from pipeline.enrichment.orchestrator import EnrichmentOrchestrator, EnrichmentRequest
from pipeline.llm.factory import LLMProviderFactory
from pipeline.llm.config import LLMConfig
from pipeline.enrichment.schemas.enrichment_v1 import EnrichmentV1


# Test video: Short educational tutorial (2-3 minutes)
# Using a public domain educational video about Python basics
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=kqtD5dpn9C8"  # Python in 100 Seconds
TEST_VIDEO_TITLE = "Python in 100 Seconds"


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for test files."""
    temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def check_api_keys():
    """Check that required API keys are available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set - skipping real API test")
    return api_key


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.slow
class TestFullPipeline:
    """Integration tests for complete pipeline workflow."""
    
    def test_full_pipeline_youtube_to_enrichment(self, temp_workspace, check_api_keys):
        """
        Test complete pipeline: YouTube → Audio → Transcript → Enrichment
        
        This test:
        1. Extracts audio from a short YouTube video
        2. Transcribes the audio using local-whisper or openai-whisper
        3. Enriches the transcript with Claude API
        4. Validates all outputs conform to schemas
        5. Verifies files are created at each stage
        
        Expected cost: ~$0.01-0.05 USD (using Claude Haiku)
        Expected duration: 2-5 minutes
        """
        print("\n" + "="*70)
        print("FULL PIPELINE INTEGRATION TEST")
        print("="*70)
        print(f"Test video: {TEST_VIDEO_TITLE}")
        print(f"Workspace: {temp_workspace}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*70 + "\n")
        
        # Define file paths
        audio_path = temp_workspace / "test_audio.mp3"
        metadata_path = temp_workspace / "test_audio.json"
        transcript_path = temp_workspace / "test_transcript.json"
        enrichment_path = temp_workspace / "test_enrichment.json"
        
        # ===================================================================
        # STAGE 1: EXTRACT AUDIO FROM YOUTUBE
        # ===================================================================
        print("STAGE 1: Extracting audio from YouTube...")
        print(f"  URL: {TEST_VIDEO_URL}")
        
        extractor = YouTubeExtractor()
        
        # Extract audio
        result_audio = extractor.extract_audio(TEST_VIDEO_URL, str(audio_path))
        assert result_audio.endswith(".mp3"), "Audio extraction should return .mp3 file"
        assert audio_path.exists(), "Audio file should be created"
        assert audio_path.stat().st_size > 0, "Audio file should not be empty"
        
        print(f"  ✓ Audio extracted: {audio_path.name} ({audio_path.stat().st_size:,} bytes)")
        
        # Extract metadata
        metadata = extractor.extract_metadata(TEST_VIDEO_URL)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        assert metadata_path.exists(), "Metadata file should be created"
        assert metadata["source_type"] == "streaming"
        assert metadata["source_url"] == TEST_VIDEO_URL
        assert isinstance(metadata["title"], str)
        assert isinstance(metadata["duration"], int)
        
        print(f"  ✓ Metadata extracted: {metadata['title']}")
        print(f"  ✓ Duration: {metadata['duration']} seconds")
        print(f"  ✓ Author: {metadata.get('author', 'Unknown')}")
        
        # ===================================================================
        # STAGE 2: TRANSCRIBE AUDIO TO TEXT
        # ===================================================================
        print("\nSTAGE 2: Transcribing audio to text...")
        
        # Create transcription config
        config = TranscriptionConfig(
            engine="local-whisper",  # Try local first
            output_dir=str(temp_workspace),
            language="en"
        )
        
        # Create engine factory and select engine
        factory = TranscriptionProviderFactory()
        
        # Try to use local-whisper, fall back to openai-whisper if not available
        try:
            print("  Attempting local-whisper...")
            adapter = factory.create_engine("local-whisper", config)
            engine_used = "local-whisper"
        except Exception as e:
            print(f"  Local whisper not available: {e}")
            print("  Falling back to openai-whisper...")
            
            # Check for OpenAI API key
            if not os.getenv("OPENAI_API_KEY"):
                pytest.skip("Neither local-whisper nor OPENAI_API_KEY available")
            
            config.engine = "openai-whisper"
            adapter = factory.create_engine("openai-whisper", config)
            engine_used = "openai-whisper"
        
        print(f"  ✓ Using engine: {engine_used}")
        
        # Transcribe audio
        print("  Transcribing... (this may take 1-2 minutes)")
        raw_transcript = adapter.transcribe(str(audio_path), language="en")
        
        # Normalize to TranscriptV1 schema
        transcript = normalize_transcript_v1(raw_transcript, adapter)
        
        # Save transcript
        persistence = LocalFilePersistence()
        persistence.persist(transcript, str(transcript_path))
        
        assert transcript_path.exists(), "Transcript file should be created"
        
        # Validate transcript structure
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
        
        assert transcript_data["transcript_version"] == "v1"
        assert "text" in transcript_data
        assert "metadata" in transcript_data
        assert len(transcript_data["text"]) > 0, "Transcript should not be empty"
        
        word_count = len(transcript_data["text"].split())
        print(f"  ✓ Transcript generated: {word_count} words")
        print(f"  ✓ Engine: {transcript_data['metadata']['engine']}")
        print(f"  ✓ Language: {transcript_data['metadata']['language']}")
        print(f"  ✓ Preview: {transcript_data['text'][:100]}...")
        
        # ===================================================================
        # STAGE 3: ENRICH TRANSCRIPT WITH LLM
        # ===================================================================
        print("\nSTAGE 3: Enriching transcript with Claude API...")
        
        # Create LLM provider factory
        llm_config = LLMConfig.load_from_config()
        provider_factory = LLMProviderFactory(llm_config)
        
        # Create orchestrator
        orchestrator = EnrichmentOrchestrator(provider_factory=provider_factory)
        
        # Create enrichment request (only summary to minimize cost)
        enrichment_request = EnrichmentRequest(
            transcript_text=transcript_data["text"],
            language=transcript_data["metadata"]["language"],
            duration=transcript_data["metadata"]["duration"],
            enrichment_types=["summary"],  # Only summary to keep costs low
            provider="claude",
            model="claude-3-haiku-20240307",
            max_cost=0.10,  # Safety limit: $0.10 USD
            dry_run=False,
            use_cache=True
        )
        
        # Estimate cost first
        print("  Estimating cost...")
        dry_run_request = EnrichmentRequest(
            transcript_text=transcript_data["text"],
            language=transcript_data["metadata"]["language"],
            duration=transcript_data["metadata"]["duration"],
            enrichment_types=["summary"],
            provider="claude",
            model="claude-3-haiku-20240307",
            dry_run=True
        )
        
        dry_run_result = orchestrator.enrich(dry_run_request)
        print(f"  ✓ Estimated cost: ${dry_run_result.estimate.total_cost:.4f} USD")
        print(f"  ✓ Estimated tokens: {dry_run_result.estimate.input_tokens + dry_run_result.estimate.output_tokens:,}")
        
        # Execute enrichment
        print("  Generating enrichment... (this will cost money)")
        enrichment = orchestrator.enrich(enrichment_request)
        
        assert isinstance(enrichment, EnrichmentV1), "Should return EnrichmentV1 object"
        
        # Save enrichment
        with open(enrichment_path, "w") as f:
            json.dump(enrichment.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
        
        assert enrichment_path.exists(), "Enrichment file should be created"
        
        # Validate enrichment structure
        assert enrichment.enrichment_version == "v1"
        assert enrichment.metadata.provider == "claude"
        assert enrichment.metadata.model == "claude-3-haiku-20240307"
        assert enrichment.metadata.cost_usd > 0, "Should have non-zero cost"
        assert enrichment.metadata.tokens_used > 0, "Should have used tokens"
        assert enrichment.summary is not None, "Should have summary"
        
        print(f"  ✓ Enrichment generated")
        print(f"  ✓ Provider: {enrichment.metadata.provider}")
        print(f"  ✓ Model: {enrichment.metadata.model}")
        print(f"  ✓ Actual cost: ${enrichment.metadata.cost_usd:.4f} USD")
        print(f"  ✓ Tokens used: {enrichment.metadata.tokens_used:,}")
        
        # Display summary
        if enrichment.summary:
            print(f"\n  Summary (short): {enrichment.summary.get('short', 'N/A')}")
        
        # ===================================================================
        # STAGE 4: VALIDATE ALL OUTPUTS
        # ===================================================================
        print("\nSTAGE 4: Validating all outputs...")
        
        # Check all files exist
        assert audio_path.exists(), "Audio file should exist"
        assert metadata_path.exists(), "Metadata file should exist"
        assert transcript_path.exists(), "Transcript file should exist"
        assert enrichment_path.exists(), "Enrichment file should exist"
        
        print(f"  ✓ All files created successfully")
        print(f"  ✓ Audio: {audio_path.name}")
        print(f"  ✓ Metadata: {metadata_path.name}")
        print(f"  ✓ Transcript: {transcript_path.name}")
        print(f"  ✓ Enrichment: {enrichment_path.name}")
        
        # Validate file sizes
        assert audio_path.stat().st_size > 1000, "Audio should be > 1KB"
        assert metadata_path.stat().st_size > 100, "Metadata should be > 100 bytes"
        assert transcript_path.stat().st_size > 100, "Transcript should be > 100 bytes"
        assert enrichment_path.stat().st_size > 100, "Enrichment should be > 100 bytes"
        
        print(f"  ✓ All files have valid sizes")
        
        # ===================================================================
        # FINAL SUMMARY
        # ===================================================================
        print("\n" + "="*70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Video: {metadata['title']}")
        print(f"Duration: {metadata['duration']} seconds")
        print(f"Transcript: {word_count} words")
        print(f"Transcription engine: {engine_used}")
        print(f"Enrichment cost: ${enrichment.metadata.cost_usd:.4f} USD")
        print(f"Total files created: 4")
        print(f"Workspace: {temp_workspace}")
        print("="*70 + "\n")
    
    def test_full_pipeline_with_all_enrichments(self, temp_workspace, check_api_keys):
        """
        Test complete pipeline with ALL enrichment types.
        
        WARNING: This test will cost more (~$0.05-0.15 USD) as it generates
        all enrichment types: summary, tags, chapters, highlights.
        
        Only run this test when you want to verify all enrichment types work.
        """
        pytest.skip("Skipping expensive test - run manually with --run-expensive flag")
        
        # This test would be similar to the above but with:
        # enrichment_types=["summary", "tag", "chapter", "highlight"]
        # And higher max_cost limit


@pytest.mark.integration
@pytest.mark.external
def test_pipeline_cost_estimation_only(temp_workspace, check_api_keys):
    """
    Test pipeline cost estimation without executing (dry-run mode).
    
    This test is FREE - it only estimates costs without making API calls.
    Use this to verify the pipeline works before running expensive tests.
    """
    print("\n" + "="*70)
    print("PIPELINE COST ESTIMATION TEST (DRY-RUN)")
    print("="*70)
    
    # Create a sample transcript (simulating stage 1-2)
    sample_transcript = {
        "transcript_version": "v1",
        "text": "This is a sample transcript about Python programming. " * 50,  # ~50 words
        "metadata": {
            "engine": "test",
            "language": "en",
            "duration": 120.0
        }
    }
    
    # Create LLM provider factory
    llm_config = LLMConfig.load_from_config()
    provider_factory = LLMProviderFactory(llm_config)
    orchestrator = EnrichmentOrchestrator(provider_factory=provider_factory)
    
    # Create dry-run request
    request = EnrichmentRequest(
        transcript_text=sample_transcript["text"],
        language="en",
        duration=120.0,
        enrichment_types=["summary", "tag", "chapter", "highlight"],
        provider="claude",
        model="claude-3-haiku-20240307",
        dry_run=True  # DRY RUN - no API calls
    )
    
    # Execute dry-run
    print("Estimating costs for all enrichment types...")
    result = orchestrator.enrich(request)
    
    # Validate dry-run result
    assert hasattr(result, 'estimate'), "Should return DryRunReport"
    assert result.estimate.total_cost > 0, "Should have estimated cost"
    
    print(f"\n✓ Cost Estimation Complete (NO API CALLS MADE)")
    print(f"  Provider: {result.provider}")
    print(f"  Model: {result.model}")
    print(f"  Enrichment types: {', '.join(result.enrichment_types)}")
    print(f"  Estimated cost: ${result.estimate.total_cost:.4f} USD")
    print(f"  Input tokens: {result.estimate.input_tokens:,}")
    print(f"  Output tokens: {result.estimate.output_tokens:,}")
    print(f"\n  Breakdown:")
    for etype, cost in result.estimate.breakdown.items():
        print(f"    - {etype}: ${cost:.4f}")
    
    print("\n" + "="*70 + "\n")
