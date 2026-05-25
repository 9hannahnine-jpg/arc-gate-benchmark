# arc-gate-benchmark

Reproducible benchmark harness for Arc Gate runtime governance.

## Benchmarks

- **AgentDojo v1** (ETH Zurich, ICLR 2024) — agentic tool poisoning across banking, slack, travel, workspace
- **InjecAgent** (University of Illinois, ACL 2024) — indirect prompt injection through tool output
- **Multi-turn escalation** — session state governance across fresh sessions, after probing, after legitimate history

## Usage

```bash
pip install requests datasets
git clone https://github.com/uiuc-kang-lab/InjecAgent.git  # for InjecAgent
export OPENAI_API_KEY=sk-...
export ARC_GATE_URL=https://your-arc-gate.up.railway.app/v1/chat/completions
python benchmark.py
```

## Results (Arc Gate v1.0, May 2026)

| Benchmark | TPR | FPR |
|---|---|---|
| AgentDojo v1 | 100% | 0% |
| InjecAgent (200 sampled) | 99% | — |
| Multi-turn escalation | 100% (4/4) | 0% |

## Target

Default target: `https://web-production-6e47f.up.railway.app/v1/chat/completions`

Override with `ARC_GATE_URL` environment variable to test your own instance.

## Related

- [arc-gate](https://github.com/9hannahnine-jpg/arc-gate) — Runtime governance proxy for LLM agents
- [arc-sentry](https://github.com/9hannahnine-jpg/arc-sentry) — Whitebox detector for self-hosted models
- [arc-gate-mcp](https://github.com/9hannahnine-jpg/arc-gate-mcp) — Runtime governance for MCP tool calls
- [arc-gate-benchmark](https://github.com/9hannahnine-jpg/arc-gate-benchmark) — Reproducible benchmark harness
