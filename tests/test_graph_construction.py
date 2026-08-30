import asyncio
import json
from types import MethodType

import numpy as np

from gos.core import engine as engine_module
from gos.core.engine import SkillGraphRAG
from gos.core.litellm_services import validate_response_model
from gos.core.schema import GOSGraph, GOSRelationList, SkillNode
from gos.core.services import SkillInformationExtractionService


class FakeEmbeddingService:
    model = "fake-embedding"
    embedding_dim = 8

    async def encode(self, texts, model=None):
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


class CompletionLLMService:
    model = "completion-llm"

    def __init__(self, *, inputs=None, outputs=None):
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.extraction_calls = 0

    async def send_message(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        response_model=None,
        **kwargs,
    ):
        if response_model is GOSRelationList:
            return GOSRelationList(relations=[]), []
        if response_model is GOSGraph:
            self.extraction_calls += 1
            return GOSGraph(
                nodes=[
                    {
                        "name": "ignored_by_parser",
                        "description": "Ignored by parser.",
                        "one_line_capability": "Completed capability.",
                        "inputs": self.inputs,
                        "outputs": self.outputs,
                        "domain_tags": ["completed domain"],
                        "tooling": ["completed tool"],
                        "example_tasks": ["completed example"],
                    }
                ],
                edges=[],
            ), []
        return response_model(), []


def _engine(tmp_path, llm):
    return SkillGraphRAG(
        config=SkillGraphRAG.Config(
            llm_service=llm,
            embedding_service=FakeEmbeddingService(),
            working_dir=str(tmp_path),
            use_full_markdown=True,
            enable_semantic_linking=False,
        )
    )


def _metadata(tmp_path, name, content):
    return {
        "source_path": str(tmp_path / name / "SKILL.md"),
        "raw_content": content,
    }


def test_complete_skill_skips_semantic_completion(tmp_path):
    content = """---
name: complete_skill
description: Parse event logs into a normalized incident timeline.
one_line_capability: Normalize event logs into incident timelines.
inputs: [event_log]
outputs: [incident_timeline]
domain: [incident analysis]
tooling: [python]
examples: [build an incident timeline]
---
"""
    llm = CompletionLLMService(inputs=["wrong_input"], outputs=["wrong_output"])

    async def scenario():
        engine = _engine(tmp_path, llm)
        await engine.async_insert_skills(
            [content],
            [_metadata(tmp_path, "complete_skill", content)],
        )
        hydrated = await engine.async_hydrate_skills(["complete_skill"])
        assert hydrated[0].inputs == ["event_log"]
        assert hydrated[0].outputs == ["incident_timeline"]

    asyncio.run(scenario())
    assert llm.extraction_calls == 0


def test_explicit_parser_io_is_not_replaced_by_semantic_completion(tmp_path):
    content = """---
name: partial_skill
description: Parse CSV measurements for downstream analysis.
one_line_capability: Normalize CSV measurements.
inputs: [csv_measurements]
outputs: [normalized_measurements]
---
"""
    llm = CompletionLLMService(inputs=["generic data"], outputs=["generic result"])

    async def scenario():
        engine = _engine(tmp_path, llm)
        await engine.async_insert_skills(
            [content],
            [_metadata(tmp_path, "partial_skill", content)],
        )
        hydrated = await engine.async_hydrate_skills(["partial_skill"])
        assert hydrated[0].inputs == ["csv_measurements"]
        assert hydrated[0].outputs == ["normalized_measurements"]
        assert hydrated[0].one_line_capability == "Normalize CSV measurements."
        assert "completed domain" in hydrated[0].domain_tags

    asyncio.run(scenario())
    assert llm.extraction_calls == 1


def test_generic_wrapper_schema_does_not_create_dependency(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    left = SkillNode.from_lists(
        name="wrapper_a",
        description="Generic automation wrapper A.",
        outputs=["operation result json"],
    )
    right = SkillNode.from_lists(
        name="wrapper_b",
        description="Generic automation wrapper B.",
        inputs=["operation result json"],
    )

    assert engine._dependency_edges_for_pair(left, right) == []


def test_specific_artifact_creates_producer_to_consumer_dependency(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="catalog_reader",
        description="Read and normalize a seismic catalog.",
        outputs=["normalized seismic catalog"],
        domain_tags=["seismology"],
    )
    consumer = SkillNode.from_lists(
        name="phase_associator",
        description="Associate phases from a normalized seismic catalog.",
        inputs=["normalized seismic catalog"],
        domain_tags=["seismology"],
    )

    edges = engine._dependency_edges_for_pair(producer, consumer)

    assert len(edges) == 1
    edge = edges[0]
    assert (edge.source, edge.target) == ("catalog_reader", "phase_associator")
    assert edge.provenance == "deterministic_io"
    assert "catalog" in edge.evidence


def test_generic_dataframe_does_not_create_cross_domain_dependency(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="feature_engineering",
        description="Engineer tabular business features.",
        outputs=["DataFrame with original and engineered columns"],
        domain_tags=["business analytics"],
    )
    consumer = SkillNode.from_lists(
        name="gamma_phase_associator",
        description="Associate seismic phase picks.",
        inputs=["phase picks DataFrame", "stations DataFrame"],
        domain_tags=["seismology"],
    )

    assert engine._dependency_edges_for_pair(producer, consumer) == []


def test_generic_analysis_report_does_not_create_dependency(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="video_analyzer",
        description="Analyze a video.",
        outputs=["video analysis report"],
    )
    consumer = SkillNode.from_lists(
        name="financial_auditor",
        description="Audit financial statements.",
        inputs=["financial analysis report"],
    )

    assert engine._dependency_edges_for_pair(producer, consumer) == []


def test_common_exchange_format_alone_does_not_cross_domains(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="pcap_analysis",
        description="Analyze packet captures.",
        outputs=["network statistics CSV report"],
        domain_tags=["cybersecurity"],
    )
    consumer = SkillNode.from_lists(
        name="flight_search",
        description="Search flights from route data.",
        inputs=["airport routes CSV file"],
        domain_tags=["travel"],
    )

    assert engine._dependency_edges_for_pair(producer, consumer) == []


def test_generic_documentation_and_structured_containers_do_not_link(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    documentation_producer = SkillNode.from_lists(
        name="security-audit",
        description="Audit SSH security.",
        outputs=["security configuration documentation"],
    )
    badge_consumer = SkillNode.from_lists(
        name="badge-evaluation",
        description="Evaluate research artifacts.",
        inputs=["artifact documentation files"],
    )
    structured_producer = SkillNode.from_lists(
        name="sqlite-map-parser",
        description="Parse map data.",
        outputs=["structured JSON data"],
    )
    vulnerability_consumer = SkillNode.from_lists(
        name="vulnerability-reporting",
        description="Report vulnerabilities.",
        inputs=["structured JSON from security scanners"],
    )
    trained_model_producer = SkillNode.from_lists(
        name="data-scientist",
        description="Train predictive models.",
        outputs=["trained model files"],
    )
    training_consumer = SkillNode.from_lists(
        name="gpu-training",
        description="Run training jobs.",
        inputs=["ML model code"],
    )

    assert (
        engine._dependency_edges_for_pair(documentation_producer, badge_consumer) == []
    )
    assert (
        engine._dependency_edges_for_pair(structured_producer, vulnerability_consumer)
        == []
    )
    assert (
        engine._dependency_edges_for_pair(trained_model_producer, training_consumer)
        == []
    )


def test_weak_single_artifact_terms_require_shared_domain(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    email_search = SkillNode.from_lists(
        name="gmail",
        description="Search email.",
        outputs=["search results list"],
        domain_tags=["email"],
    )
    fund_search = SkillNode.from_lists(
        name="fuzzy-fund-search",
        description="Search investment funds.",
        inputs=["search keywords"],
        domain_tags=["finance"],
    )
    html_report = SkillNode.from_lists(
        name="fuzzer",
        description="Generate coverage reports.",
        outputs=["coverage report HTML"],
        domain_tags=["software testing"],
    )
    audiobook = SkillNode.from_lists(
        name="audiobook",
        description="Convert articles to audio.",
        inputs=["HTML content"],
        domain_tags=["audio publishing"],
    )
    gmail = SkillNode.from_lists(
        name="gmail",
        description="Read email messages.",
        outputs=["email data"],
        domain_tags=["email"],
    )
    pdf_redaction = SkillNode.from_lists(
        name="academic-pdf-redaction",
        description="Redact academic PDF identifiers.",
        inputs=["names, affiliations, and emails"],
        domain_tags=["document privacy"],
    )

    assert engine._dependency_edges_for_pair(email_search, fund_search) == []
    assert engine._dependency_edges_for_pair(html_report, audiobook) == []
    assert engine._dependency_edges_for_pair(gmail, pdf_redaction) == []


def test_structured_response_unwraps_minimax_mixed_content_list():
    payload = json.dumps(["", {"relations": []}])

    parsed = validate_response_model(GOSRelationList, payload)

    assert parsed.relations == []


def test_evidence_prefilter_preserves_every_pair_with_exact_pair_evidence(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    nodes = [
        SkillNode.from_lists(
            name="catalog_reader",
            description="Read catalogs.",
            outputs=["normalized seismic catalog"],
            domain_tags=["seismology"],
            tooling=["obspy"],
        ),
        SkillNode.from_lists(
            name="phase_associator",
            description="Associate phases.",
            inputs=["normalized seismic catalog"],
            domain_tags=["seismology"],
            tooling=["obspy"],
        ),
        SkillNode.from_lists(
            name="audio_transcriber",
            description="Transcribe audio.",
            inputs=["WAV audio"],
            domain_tags=["speech"],
            tooling=["whisper"],
        ),
        SkillNode.from_lists(
            name="unrelated",
            description="Unrelated helper.",
            domain_tags=["finance"],
        ),
    ]

    indexes = engine._build_pair_evidence_indexes(nodes)
    for source_index, source in enumerate(nodes):
        prefiltered = engine._evidence_candidate_indices_for_node(source, indexes)
        for candidate_index, candidate in enumerate(nodes):
            if source_index == candidate_index:
                continue
            _, has_evidence = engine._link_pair_feature_score(source, candidate)
            if has_evidence:
                assert candidate_index in prefiltered


def test_alternative_gate_rejects_same_wrapper_and_complementary_outputs(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    tts_a = SkillNode.from_lists(
        name="openai-tts",
        description="Convert text to speech audio with OpenAI voices.",
        inputs=["text input", "voice selection"],
        outputs=["audio file (mp3, wav, opus)"],
    )
    tts_b = SkillNode.from_lists(
        name="google-tts",
        description="Convert text to speech audio with Google voices.",
        inputs=["text string", "language code"],
        outputs=["MP3 audio file", "WAV audio file"],
    )
    wrapper_a = SkillNode.from_lists(
        name="21risk-automation",
        description="Automate 21risk through Rube MCP tools.",
        inputs=["tool request"],
        outputs=["tool execution results"],
    )
    wrapper_b = SkillNode.from_lists(
        name="2chat-automation",
        description="Automate 2chat through Rube MCP tools.",
        inputs=["tool request"],
        outputs=["tool execution results"],
    )
    accommodations = SkillNode.from_lists(
        name="search-accommodations",
        description="Search travel accommodations by city.",
        inputs=["city name"],
        outputs=["accommodation listings"],
    )
    attractions = SkillNode.from_lists(
        name="search-attractions",
        description="Search travel attractions by city.",
        inputs=["city name"],
        outputs=["attraction records"],
    )

    assert engine._alternative_relation_supported(tts_a, tts_b) is True
    assert engine._alternative_relation_supported(wrapper_a, wrapper_b) is False
    assert engine._alternative_relation_supported(accommodations, attractions) is False


def test_llm_dependency_direction_rejects_reversed_io_claim(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    optimizer = SkillNode.from_lists(
        name="optimizer",
        description="Solve a power-system optimization problem.",
        inputs=["power system network data", "initial state vector"],
        outputs=["optimal decision variables", "solver status"],
        domain_tags=["power systems"],
    )
    parser = SkillNode.from_lists(
        name="network-parser",
        description="Parse a network file.",
        inputs=["network JSON file"],
        outputs=["bus array", "branch array", "generator array"],
        domain_tags=["power systems"],
    )
    nodes = {optimizer.name: optimizer, parser.name: parser}

    assert (
        engine._llm_dependency_direction_supported(
            "optimizer",
            "network-parser",
            "optimizer consumes network data; network-parser outputs bus arrays.",
            nodes,
        )
        is False
    )


def test_llm_dependency_direction_accepts_stronger_forward_interface(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    sampler = SkillNode.from_lists(
        name="sampler",
        description="Sample video frames.",
        outputs=["interval instructions", "per-frame masks"],
        domain_tags=["video processing"],
    )
    validator = SkillNode.from_lists(
        name="validator",
        description="Validate sampled outputs.",
        inputs=["interval instructions JSON", "masks NPZ file"],
        outputs=["validation signal"],
        domain_tags=["video processing"],
    )
    nodes = {sampler.name: sampler, validator.name: validator}

    assert (
        engine._llm_dependency_direction_supported(
            "sampler",
            "validator",
            "sampler provides intervals and masks consumed by validator.",
            nodes,
        )
        is True
    )


def test_llm_dependency_direction_downweights_weak_reverse_container(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    extractor = SkillNode.from_lists(
        name="frame-extractor",
        description="Extract frames from video.",
        inputs=["video file"],
        outputs=["extracted image frames"],
        domain_tags=["video processing"],
    )
    estimator = SkillNode.from_lists(
        name="motion-estimator",
        description="Estimate camera motion from frames.",
        inputs=["video frames"],
        outputs=["video trajectory"],
        domain_tags=["video processing"],
    )
    nodes = {extractor.name: extractor, estimator.name: estimator}

    assert (
        engine._llm_dependency_direction_supported(
            "frame-extractor",
            "motion-estimator",
            "frame-extractor outputs image frames; motion-estimator consumes video frames.",
            nodes,
        )
        is True
    )


def test_llm_dependency_rejects_shared_control_vocabulary_without_handoff(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    tuning = SkillNode.from_lists(
        name="mpc-horizon-tuning",
        description="Select MPC prediction horizon and cost matrices.",
        inputs=[
            "system model (A, B matrices)",
            "tension and velocity references",
            "sampling time dt",
        ],
        outputs=[
            "prediction horizon N",
            "state cost matrix Q",
            "control cost matrix R",
            "terminal cost matrix P",
        ],
        domain_tags=["model predictive control", "tension control"],
    )
    integral = SkillNode.from_lists(
        name="integral-action-design",
        description="Add integral action to MPC.",
        inputs=[
            "tension measurement signal",
            "reference tension signal",
            "MPC control output",
        ],
        outputs=["combined control output", "integral control term"],
        domain_tags=["model predictive control", "tension control"],
    )
    nodes = {tuning.name: tuning, integral.name: integral}

    assert (
        engine._llm_dependency_direction_supported(
            tuning.name,
            integral.name,
            "SOURCE produces tuned MPC controller parameters; TARGET consumes MPC control output.",
            nodes,
            [
                "mpc-horizon-tuning outputs prediction horizon and cost matrices",
                "integral-action-design inputs include MPC control output",
            ],
        )
        is False
    )


def test_llm_dependency_rejects_migration_step_with_incompatible_state(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    boot_migration = SkillNode.from_lists(
        name="spring-boot-migration",
        description="Migrate Spring Boot 2 applications to Spring Boot 3.",
        inputs=[
            "pom.xml with Spring Boot 2.x dependencies",
            "Java source files with javax imports",
            "RestTemplate HTTP client code",
        ],
        outputs=[
            "Updated pom.xml with Spring Boot 3.x dependencies",
            "Migrated Java source files with jakarta imports",
            "RestClient-based HTTP client code",
        ],
        domain_tags=["java", "spring-boot", "migration"],
    )
    restclient_migration = SkillNode.from_lists(
        name="restclient-migration",
        description="Migrate RestTemplate code to RestClient.",
        inputs=[
            "RestTemplate Java source files",
            "Spring Boot project with RestTemplate configuration",
        ],
        outputs=["Migrated Java source files using RestClient"],
        domain_tags=["java", "spring-boot", "migration"],
    )
    nodes = {
        boot_migration.name: boot_migration,
        restclient_migration.name: restclient_migration,
    }

    assert (
        engine._llm_dependency_direction_supported(
            boot_migration.name,
            restclient_migration.name,
            "SOURCE produces RestClient code; TARGET converts RestTemplate code.",
            nodes,
            [
                "spring-boot-migration outputs RestClient-based HTTP client code",
                "restclient-migration consumes RestTemplate Java source files",
            ],
        )
        is False
    )


def test_llm_dependency_keeps_concrete_audio_handoff(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    extractor = SkillNode.from_lists(
        name="audio-extractor",
        description="Extract transcription-ready audio from video.",
        inputs=["input video file"],
        outputs=["mono WAV audio (16kHz PCM 16-bit)"],
        domain_tags=["audio processing", "speech"],
    )
    transcriber = SkillNode.from_lists(
        name="whisper-transcription",
        description="Transcribe an audio or video file.",
        inputs=["audio/video file"],
        outputs=["timestamped transcript JSON"],
        domain_tags=["audio processing", "speech"],
    )
    nodes = {extractor.name: extractor, transcriber.name: transcriber}

    assert (
        engine._llm_dependency_direction_supported(
            extractor.name,
            transcriber.name,
            "SOURCE provides mono WAV audio; TARGET consumes the audio file.",
            nodes,
            ["mono WAV audio is accepted as the transcription input"],
        )
        is True
    )


def test_llm_dependency_keeps_multiple_evidence_grounded_metadata_fields(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    media_info = SkillNode.from_lists(
        name="media-info",
        description="Inspect media metadata.",
        outputs=["frame rate", "resolution information", "duration information"],
        domain_tags=["video processing"],
    )
    sampler = SkillNode.from_lists(
        name="sampler",
        description="Plan frame sampling from video metadata.",
        inputs=["video metadata (frame count, fps, resolution, duration)"],
        domain_tags=["video processing"],
    )
    nodes = {media_info.name: media_info, sampler.name: sampler}

    assert (
        engine._llm_dependency_direction_supported(
            media_info.name,
            sampler.name,
            "SOURCE provides frame rate, resolution, and duration metadata consumed by TARGET.",
            nodes,
            ["frame rate, resolution, and duration are sampling inputs"],
        )
        is True
    )


def test_llm_dependency_rejects_tooling_only_or_alternative_environment_setup(
    tmp_path,
):
    engine = _engine(tmp_path, CompletionLLMService())
    uv_manager = SkillNode.from_lists(
        name="uv-package-manager",
        description="Manage Python packages and virtual environments with uv.",
        inputs=["Python project specifications", "requirements.txt files"],
        outputs=[
            "installed Python packages in virtual environments",
            "created virtual environments",
            "requirements.txt exports with optional hashes",
        ],
        domain_tags=["python", "virtual environment", "dependency management"],
        tooling=["uv"],
    )
    badge = SkillNode.from_lists(
        name="badge-evaluation",
        description="Evaluate reproducibility badges.",
        inputs=["research artifact package", "artifact documentation files"],
        outputs=["badge evaluation report"],
        domain_tags=["artifact evaluation", "reproducibility"],
    )
    harbor = SkillNode.from_lists(
        name="harbor",
        description="Run agent evaluations.",
        inputs=["SkillsBench task directory", "agent skill files"],
        outputs=["evaluation reward score", "job output directory"],
        domain_tags=["agent evaluation"],
        tooling=["uv tool"],
    )
    setup = SkillNode.from_lists(
        name="setup-env",
        description="Create Python virtual environments and install dependencies.",
        inputs=["Python project codebase", "pyproject.toml or requirements.txt"],
        outputs=["configured virtual environment", "installed Python dependencies"],
        domain_tags=["python", "virtual environment", "dependency management"],
        tooling=["uv"],
    )
    nodes = {node.name: node for node in (uv_manager, badge, harbor, setup)}

    assert engine._alternative_relation_supported(uv_manager, setup) is True
    assert engine._dependency_edges_for_pair(uv_manager, setup) == []
    for target, evidence in (
        (badge, "uv installs packages required before badge evaluation"),
        (harbor, "harbor uses uv as its CLI execution environment"),
        (setup, "both create environments and install Python dependencies"),
    ):
        assert (
            engine._llm_dependency_direction_supported(
                uv_manager.name,
                target.name,
                evidence,
                nodes,
                [evidence],
            )
            is False
        )


def test_llm_dependency_keeps_handoff_between_related_but_non_substitutable_skills(
    tmp_path,
):
    engine = _engine(tmp_path, CompletionLLMService())
    data = SkillNode.from_lists(
        name="power-flow-data",
        description="Power system network data formats and topology. Use when parsing bus, generator, and branch data for power flow analysis.",
        inputs=["network JSON file (MATPOWER format)"],
        outputs=[
            "bus array (numpy)",
            "generator array (numpy)",
            "branch array (numpy)",
            "gencost array (numpy)",
            "reserve capacity array",
            "bus number to index mapping",
            "per-unit normalized quantities",
        ],
        domain_tags=[
            "power systems",
            "electrical engineering",
            "power flow analysis",
            "MATPOWER format",
            "grid topology",
        ],
        tooling=["Python", "numpy", "json"],
    )
    branch_model = SkillNode.from_lists(
        name="ac-branch-pi-model",
        description="AC branch pi-model power flow equations with transformer tap ratio and phase shift.",
        inputs=[
            "network.json (bus and branch data)",
            "initial voltage magnitudes and angles (per-unit)",
            "baseMVA",
        ],
        outputs=[
            "branch P and Q flows in both directions (per-unit)",
            "apparent power S in MVA (both directions)",
            "bus injection sums for nodal balance",
        ],
        domain_tags=[
            "power systems",
            "AC power flow",
            "transmission lines",
            "transformers",
            "MATPOWER",
            "optimal power flow",
        ],
        tooling=["Python", "numpy", "scripts/branch_flows.py"],
    )
    nodes = {data.name: data, branch_model.name: branch_model}

    assert engine._alternative_relation_supported(data, branch_model) is True
    assert (
        engine._llm_dependency_direction_supported(
            data.name,
            branch_model.name,
            "SOURCE provides bus and branch arrays consumed by TARGET.",
            nodes,
            ["bus and branch arrays are the network input"],
        )
        is True
    )


def test_schema_keeps_artifact_content_introduced_by_with(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="water-data",
        description="Download water observations.",
        outputs=["DataFrame with water level or streamflow values"],
        domain_tags=["hydrology", "water level"],
    )
    consumer = SkillNode.from_lists(
        name="flood-detector",
        description="Detect floods from water levels.",
        inputs=["instantaneous water level data (DataFrame with datetime index)"],
        domain_tags=["hydrology", "water level"],
    )

    edges = engine._dependency_edges_for_pair(producer, consumer)

    assert len(edges) == 1
    assert {"level", "water"} <= set(edges[0].evidence.split(", "))


def test_workflow_direction_requires_forward_evidence(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    city_search = SkillNode.from_lists(
        name="search-cities",
        description="List cities in a state.",
        inputs=["state name"],
        outputs=["list of city names"],
        domain_tags=["travel planning"],
    )
    restaurant_search = SkillNode.from_lists(
        name="search-restaurants",
        description="Find restaurants in a city.",
        inputs=["city name"],
        outputs=["restaurant data"],
        domain_tags=["travel planning"],
    )
    nodes = {
        city_search.name: city_search,
        restaurant_search.name: restaurant_search,
    }

    assert (
        engine._workflow_direction_supported(
            "search-cities",
            "search-restaurants",
            "city lookup precedes restaurants",
            nodes,
        )
        is True
    )
    assert (
        engine._workflow_direction_supported(
            "search-restaurants",
            "search-cities",
            "search-cities provides the city name to search-restaurants",
            nodes,
        )
        is False
    )


def test_semantic_gate_rejects_generic_wrappers_but_keeps_narrow_domain(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    wrapper_a = SkillNode.from_lists(
        name="service-a-automation",
        description="Automate Service A through the Rube MCP wrapper.",
        domain_tags=["service a"],
    )
    wrapper_b = SkillNode.from_lists(
        name="service-b-automation",
        description="Automate Service B through the Rube MCP wrapper.",
        domain_tags=["service b"],
    )
    chat_wrapper = SkillNode.from_lists(
        name="-2chat-automation",
        description="Automate 2Chat tasks via Rube MCP (Composio).",
        domain_tags=["messaging", "automation", "mcp", "composio", "2chat"],
        tooling=["Rube MCP", "Composio 2Chat toolkit"],
    )
    ably_wrapper = SkillNode.from_lists(
        name="ably-automation",
        description="Automate Ably tasks via Rube MCP (Composio).",
        domain_tags=[
            "messaging-automation",
            "automation",
            "rube-mcp",
            "composio",
            "ably",
        ],
        tooling=["Rube MCP", "Composio Ably toolkit"],
    )
    pcap_a = SkillNode.from_lists(
        name="pcap-triage",
        description="Triage packet captures for incidents.",
        domain_tags=["network forensics", "pcap analysis"],
    )
    pcap_b = SkillNode.from_lists(
        name="pcap-alerts",
        description="Extract security alerts from packet captures.",
        domain_tags=["network forensics", "pcap analysis"],
    )

    assert engine._semantic_relation_supported(wrapper_a, wrapper_b) is False
    assert engine._semantic_relation_supported(chat_wrapper, ably_wrapper) is False
    assert engine._semantic_relation_supported(pcap_a, pcap_b) is True


def test_evidence_prefilter_omits_unrelated_nodes(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    source = SkillNode.from_lists(
        name="catalog_reader",
        description="Read catalogs.",
        outputs=["normalized seismic catalog"],
        domain_tags=["seismology"],
    )
    unrelated = [
        SkillNode.from_lists(
            name=f"unrelated_{index}",
            description=f"Unrelated capability {index}.",
            domain_tags=[f"uniquearea{index}"],
        )
        for index in range(50)
    ]
    consumer = SkillNode.from_lists(
        name="phase_associator",
        description="Associate phases.",
        inputs=["normalized seismic catalog"],
        domain_tags=["seismology"],
    )
    nodes = [source, *unrelated, consumer]

    indexes = engine._build_pair_evidence_indexes(nodes)
    candidates = engine._evidence_candidate_indices_for_node(
        source, indexes, node_index=0
    )

    assert candidates == {len(nodes) - 1}


def test_construction_code_hash_includes_artifact_policy_constants(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    before = engine._construction_code_sha256()
    marker = "temporary_test_artifact_policy_token"
    engine_module.GENERIC_SCHEMA_TOKENS.add(marker)
    try:
        after = engine._construction_code_sha256()
    finally:
        engine_module.GENERIC_SCHEMA_TOKENS.remove(marker)

    assert after != before


def test_signature_tokens_preserve_short_words_ending_in_s(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())

    assert "bus" in engine._signature_tokens(["bus array"])


def test_artifact_modifier_does_not_create_dependency(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="video_analyzer",
        description="Analyze a video and return a summary.",
        outputs=["video summary", "structured count data from video footage"],
    )
    consumer = SkillNode.from_lists(
        name="video_editor",
        description="Edit an input video file.",
        inputs=["video file"],
    )

    assert engine._dependency_edges_for_pair(producer, consumer) == []


def test_artifact_head_or_concrete_format_creates_dependency(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    video_producer = SkillNode.from_lists(
        name="video_filter",
        description="Filter a video.",
        outputs=["filtered video file"],
        domain_tags=["video processing"],
    )
    video_consumer = SkillNode.from_lists(
        name="video_editor",
        description="Edit an input video file.",
        inputs=["video file"],
        domain_tags=["video processing"],
    )
    table_producer = SkillNode.from_lists(
        name="pdf_table_extractor",
        description="Extract tables from PDF files.",
        outputs=["extracted tables (CSV/Excel)"],
        domain_tags=["data processing"],
    )
    table_consumer = SkillNode.from_lists(
        name="csv_processor",
        description="Process CSV files.",
        inputs=["CSV file"],
        domain_tags=["data processing"],
    )

    video_edges = engine._dependency_edges_for_pair(video_producer, video_consumer)
    table_edges = engine._dependency_edges_for_pair(table_producer, table_consumer)

    assert len(video_edges) == 1
    assert video_edges[0].evidence == "video"
    assert len(table_edges) == 1
    assert "csv" in table_edges[0].evidence


def test_multi_token_artifact_signature_survives_container_head_mismatch(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="dispatch-data",
        description="Prepare reserve data.",
        outputs=["reserve capacity array"],
        domain_tags=["power systems"],
    )
    consumer = SkillNode.from_lists(
        name="economic-dispatch",
        description="Solve reserve-aware dispatch.",
        inputs=["reserve capacity array"],
        domain_tags=["power systems"],
    )

    edges = engine._dependency_edges_for_pair(producer, consumer)

    assert len(edges) == 1
    assert edges[0].evidence == "capacity, reserve"


def test_deterministic_dependency_rejects_ambiguous_cross_domain_schema(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    packet_analyzer = SkillNode.from_lists(
        name="packet-analyzer",
        description="Infer a packet network topology.",
        outputs=["network topology data"],
        domain_tags=["network security", "packet analysis"],
    )
    grid_optimizer = SkillNode.from_lists(
        name="grid-optimizer",
        description="Optimize an electric grid.",
        inputs=["network topology data"],
        domain_tags=["power systems", "electric grid"],
    )

    assert engine._dependency_edges_for_pair(packet_analyzer, grid_optimizer) == []


def test_deterministic_dependency_allows_concrete_contract_when_domain_is_missing(
    tmp_path,
):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="catalog-reader",
        description="Normalize a seismic event catalog.",
        outputs=["normalized seismic catalog"],
    )
    consumer = SkillNode.from_lists(
        name="phase-associator",
        description="Associate phases from a catalog.",
        inputs=["seismic catalog"],
    )

    edges = engine._dependency_edges_for_pair(producer, consumer)

    assert len(edges) == 1
    assert edges[0].source == "catalog-reader"
    assert edges[0].target == "phase-associator"
    assert set(edges[0].evidence.split(", ")) == {"catalog", "seismic"}


def test_deterministic_dependency_requires_strong_domain_match_for_weak_artifacts(
    tmp_path,
):
    engine = _engine(tmp_path, CompletionLLMService())
    light_curve = SkillNode.from_lists(
        name="light-curve-preprocessing",
        description="Clean astronomical light curves.",
        outputs=["cleaned light curve time series"],
        domain_tags=[
            "astronomy",
            "exoplanet detection",
            "time series analysis",
            "photometry",
            "light curves",
            "stellar variability",
        ],
    )
    economic_detrending = SkillNode.from_lists(
        name="timeseries-detrending",
        description="Detrend economic business-cycle series.",
        inputs=["economic time series data"],
        domain_tags=[
            "macroeconomics",
            "time series analysis",
            "business cycle analysis",
            "econometrics",
        ],
    )
    raw_audio = SkillNode.from_lists(
        name="audio-normalizer",
        description="Normalize an audio file.",
        outputs=["audio file"],
        domain_tags=[
            "audio processing",
            "video editing",
            "audio mixing",
            "normalization",
            "audio filters",
        ],
    )
    clustering = SkillNode.from_lists(
        name="speaker-clustering",
        description="Cluster speaker embeddings.",
        inputs=["speaker embeddings list", "VAD-segmented audio"],
        domain_tags=[
            "speaker diarization",
            "clustering",
            "audio processing",
            "speaker recognition",
        ],
    )

    assert engine._dependency_edges_for_pair(light_curve, economic_detrending) == []
    assert engine._dependency_edges_for_pair(raw_audio, clustering) == []


def test_code_fence_marker_is_not_an_output_artifact(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="bad_completion",
        description="A malformed semantic completion.",
        outputs=["```python"],
    )
    consumer = SkillNode.from_lists(
        name="python_optimizer",
        description="Optimize Python source code.",
        inputs=["Python source code"],
    )

    assert engine._dependency_edges_for_pair(producer, consumer) == []


def test_programming_language_only_code_container_requires_llm_validation(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="java_migration",
        description="Migrate a Java project.",
        outputs=["migrated Java source code"],
        domain_tags=["software engineering"],
    )
    python_consumer = SkillNode.from_lists(
        name="python_optimizer",
        description="Optimize Python source code.",
        inputs=["Python source code"],
        domain_tags=["software engineering"],
    )
    java_consumer = SkillNode.from_lists(
        name="java_analyzer",
        description="Analyze Java source code.",
        inputs=["Java source code"],
        domain_tags=["software engineering"],
    )

    assert engine._dependency_edges_for_pair(producer, python_consumer) == []
    assert engine._dependency_edges_for_pair(producer, java_consumer) == []


def test_generic_result_suffixes_do_not_promote_domain_modifiers(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    java_scaffolder = SkillNode.from_lists(
        name="senior-java",
        description="Scaffold Spring Boot applications.",
        inputs=["Security configuration parameters"],
        outputs=["Security configuration"],
        domain_tags=["java", "spring-boot"],
    )
    boot_migration = SkillNode.from_lists(
        name="spring-boot-migration",
        description="Migrate Spring Boot 2 to 3.",
        inputs=["Spring Security configuration files"],
        outputs=["Updated Spring Security configuration"],
        domain_tags=["java", "spring-boot", "migration"],
    )
    ably = SkillNode.from_lists(
        name="ably-automation",
        description="Automate Ably through Rube MCP.",
        inputs=["tool execution requests"],
        outputs=["tool execution results"],
        domain_tags=["messaging", "automation"],
    )
    chat = SkillNode.from_lists(
        name="-2chat-automation",
        description="Automate 2Chat through Rube MCP.",
        inputs=["tool execution requests"],
        outputs=["tool execution results"],
        domain_tags=["messaging", "automation"],
    )
    vision = SkillNode.from_lists(
        name="vision-analyzer",
        description="Describe an image.",
        outputs=["image description text"],
        domain_tags=["computer vision"],
    )
    counter = SkillNode.from_lists(
        name="object-counter",
        description="Count objects in an image.",
        inputs=["image file"],
        domain_tags=["computer vision"],
    )
    solver = SkillNode.from_lists(
        name="solver",
        description="Solve an optimization problem.",
        outputs=["objective function value"],
        domain_tags=["optimization"],
    )
    problem_builder = SkillNode.from_lists(
        name="problem-builder",
        description="Build an optimization objective.",
        inputs=["objective function"],
        domain_tags=["optimization"],
    )

    assert engine._dependency_edges_for_pair(java_scaffolder, boot_migration) == []
    assert engine._dependency_edges_for_pair(ably, chat) == []
    assert engine._dependency_edges_for_pair(vision, counter) == []
    assert engine._dependency_edges_for_pair(solver, problem_builder) == []


def test_generic_configuration_requires_concrete_type_or_format(tmp_path):
    engine = _engine(tmp_path, CompletionLLMService())
    producer = SkillNode.from_lists(
        name="java_migration",
        description="Update Java persistence configuration.",
        outputs=["updated persistence.xml configuration"],
    )
    generic_consumer = SkillNode.from_lists(
        name="generic_config_reader",
        description="Read a configuration file.",
        inputs=["configuration file"],
    )
    yaml_producer = SkillNode.from_lists(
        name="yaml_writer",
        description="Write YAML configuration.",
        outputs=["YAML configuration file"],
        domain_tags=["configuration management"],
    )
    yaml_consumer = SkillNode.from_lists(
        name="yaml_reader",
        description="Read YAML configuration.",
        inputs=["YAML configuration file"],
        domain_tags=["configuration management"],
    )

    assert engine._dependency_edges_for_pair(producer, generic_consumer) == []
    yaml_edges = engine._dependency_edges_for_pair(yaml_producer, yaml_consumer)
    assert len(yaml_edges) == 1
    assert yaml_edges[0].evidence == "yaml"


def test_semantic_completion_coerces_object_artifacts_and_drops_control_parameters():
    payload = json.dumps(
        {
            "nodes": [
                {
                    "name": "audio-extractor",
                    "description": "Extract audio from video.",
                    "inputs": [
                        {
                            "name": "video",
                            "type": "string",
                            "description": "Path to input video file",
                        },
                        {
                            "name": "output",
                            "type": "string",
                            "description": "Path to output WAV file",
                        },
                        {
                            "name": "sample-rate",
                            "type": "integer",
                            "description": "Audio sample rate in Hz",
                        },
                    ],
                    "outputs": [
                        {
                            "name": "audio",
                            "type": "file",
                            "format": "WAV",
                            "description": "Mono WAV audio file",
                        }
                    ],
                }
            ],
            "edges": [],
        }
    )

    graph = validate_response_model(GOSGraph, payload)

    assert graph.nodes[0].inputs == ["Path to input video file"]
    assert graph.nodes[0].outputs == ["Mono WAV audio file"]


def test_semantic_extraction_uses_bounded_document_concurrency(tmp_path):
    async def scenario():
        engine = _engine(tmp_path, CompletionLLMService())
        service: SkillInformationExtractionService = (
            engine.information_extraction_service
        )
        service.extraction_concurrency = 2
        active = 0
        maximum_active = 0

        async def fake_extract(self, llm, document, prompt_kwargs, entity_types):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return None

        service._extract = MethodType(fake_extract, service)
        futures = service.extract(
            llm=CompletionLLMService(),
            documents=[[object()] for _ in range(5)],
            prompt_kwargs={},
            entity_types=["Skill"],
        )
        await asyncio.gather(*futures)

        assert maximum_active == 2

    asyncio.run(scenario())
