# Container Layout

Use this when wiring the skill into a Docker or eval image.

## Expected runtime

- `graphskills-query` is installed on `PATH`
- `GOS_WORKING_DIR` points to a writable runtime workspace
- `GOS_PREBUILT_WORKING_DIR` optionally points to a read-only prebuilt workspace baked into the image

Recommended values:

```bash
ENV GOS_PREBUILT_WORKING_DIR=/opt/graphskills/prebuilt
ENV GOS_WORKING_DIR=/opt/graphskills/runtime
```

At first use, GoS will clone the prebuilt workspace into `GOS_WORKING_DIR` if the runtime workspace is empty.

## Dockerfile pattern

```dockerfile
COPY graph_workspace /opt/graphskills/prebuilt

ENV GOS_PREBUILT_WORKING_DIR=/opt/graphskills/prebuilt
ENV GOS_WORKING_DIR=/opt/graphskills/runtime

# Install the GoS package so `gos` and `graphskills-query` are on PATH.
COPY . /opt/graphskills/src
RUN pip install /opt/graphskills/src

# Load the agent skill into the paths used by your agent runtime.
COPY agent_skills/graph-skills-retriever /root/.codex/skills/graph-skills-retriever
COPY agent_skills/graph-skills-retriever /root/.claude/skills/graph-skills-retriever
COPY agent_skills/graph-skills-retriever /root/.gemini/skills/graph-skills-retriever
```

## Notes

- If the prebuilt workspace itself is writable and you do not need copy-on-write isolation, set only `GOS_WORKING_DIR=/opt/graphskills/prebuilt`.
- If you are using a benchmark image that already provides a `graphskills-query` shim, this skill still works unchanged.
