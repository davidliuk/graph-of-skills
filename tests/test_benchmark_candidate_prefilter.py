from __future__ import annotations

from evaluation.analysis.benchmark_candidate_prefilter import benchmark_prefilter
from gos.core.schema import SkillNode


def test_prefilter_benchmark_preserves_ranked_outputs_and_reduces_pairs():
    nodes = [
        SkillNode.from_lists(
            name="producer",
            description="Produce a catalog.",
            outputs=["normalized seismic catalog"],
            domain_tags=["seismology"],
        ),
        SkillNode.from_lists(
            name="consumer",
            description="Consume a catalog.",
            inputs=["normalized seismic catalog"],
            domain_tags=["seismology"],
        ),
        *[
            SkillNode.from_lists(
                name=f"unrelated-{index}",
                description="Unrelated.",
                domain_tags=[f"uniquearea{index}"],
            )
            for index in range(20)
        ],
    ]

    result = benchmark_prefilter(nodes, sample_size=len(nodes), candidate_top_k=8)

    assert result["exact_output_match"] is True
    assert result["prefiltered_pair_scores"] < result["brute_pair_scores"]
