# Offline Analysis Tools

Command-line utilities used for the graph-quality, robustness, and matched-retrieval
analyses reported in the paper. They operate on an indexed GoS workspace and do not
require running an agent.

All tools are run as modules:

```bash
python -m evaluation.analysis.<tool> --help
```

## Graph quality

| Tool | What it does |
|------|--------------|
| `graph_diagnostics` | Edge-type counts, isolate counts, and degree statistics for a graph bundle |
| `compare_graph_runs` | Compare two independent builds (typed-directed Jaccard, per-type agreement) |
| `revalidate_graph_bundle` | Replay the current deterministic and LLM acceptance gates over an existing bundle |

## Retrieval

| Tool | What it does |
|------|--------------|
| `run_retrieval_ablation` | Budget-matched offline retrieval across graph operations (reverse-aware PPR, forward-only, no-graph, one-hop) |
| `aggregate_retrieval_runs` | Aggregate retrieval metrics across graph builds, with bootstrap intervals |
| `benchmark_candidate_prefilter` | Benchmark the evidence-token postings index against all-pairs scoring |

## Robustness

| Tool | What it does |
|------|--------------|
| `metadata_stress` | Documentation-quality ablations (drop or genericize I/O fields and descriptions) |
| `prepare_scale_workspace` | Build an edge-free directed workspace for library-size scaling runs |
| `scale_summary` | Summarize construction cost and edge counts across library sizes |

## Edge audit

| Tool | What it does |
|------|--------------|
| `aggregate_edge_annotations` | Aggregate provenance-blinded edge annotations |
| `apply_edge_annotation_decisions` | Apply frozen blind decisions back onto a bundle |

## Benchmark metadata

| Tool | What it does |
|------|--------------|
| `skillsbench_oracles` | Audit SkillsBench oracle-skill coverage per task |
| `alfworld_metadata_audit` | Audit ALFWorld skill metadata completeness |

## Internal helpers

`deterministic_edges`, `manifest`, and `workspace_bundle` are library modules used by
the tools above. They have no CLI entry point.

## Example

Inspect the graph produced by a fresh build:

```bash
gos index data/skillsets/skills_200 --workspace data/gos_workspace/skills_200 --clear
python -m evaluation.analysis.graph_diagnostics --workspace data/gos_workspace/skills_200
```

`construction_report.json` in the workspace records the construction funnel
(candidate bound, submitted pairs, validator requests, proposals, accept/reject,
dedup) when `GOS_CONSTRUCTION_REPORT=true`.
