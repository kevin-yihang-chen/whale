# Proposer options and costs

Current verification,2026-09-15: the official domestic [detailed price table](https://docs.bigmodel.cn/cn/guide/start/pricing)
lists GLM-5.3-Flash at0.8CNY per million input tokens,2.8CNY output, and0.23CNY
cache reads. Its official model page confirms1M context and128K maximum output.
The project's existing1/4 rates remain conservative; no historical settlement
or unresolved reservation was rewritten. Public source copies and hashes are in
`results/fast-chart-glm-public-pricing-20260915-v1.json`. This check made no paid
request and did not query the provider invoice or account balance.

The following September9 measurements and comparisons are historical:

Updated 2026-09-09. The user authorized the domestic Zhipu API and a45 CNY first
round. GLM-5.3-Flash has completed real candidate generation through WHALE's
original wrapper on compute node gpucluster-g18. A bounded native Chess search
evaluated h0 and h1 at0/8 each; h1 was accepted on a filesystem-order tie, not a
score gain. Formal VLM comparisons remain unrun. The project now has15 real
requests,10809 input plus27072 cache-read input and2344 output tokens; conservative
accounting totals0.047257CNY with no held reservations. This is not the provider's
bill or account balance. The earlier eight diagnostic requests cost0.008267CNY
under the same accounting rules.

The key is kept outside Git in a0600 private file. The validated route is
`https://open.bigmodel.cn/api/anthropic`, documented by
[Zhipu](https://docs.bigmodel.cn/cn/guide/develop/claude/introduction), using the
[official model code](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash).
`ours.glm_gateway` keeps a persistent45 CNY project journal and sends the key only
to that route. Its1/4 CNY per million input/output rates are conservative budget
assumptions, not a quote of domestic promotional prices. CLI dollar estimates
must not be substituted for provider accounting. The real API key is never
passed to the native tool subprocess. See [the measured budget record](../results/glm-budget-summary-20260909-search.json).

Compute jobs must not inherit a login-node proxy listening on127.0.0.1. The
launcher clears those proxy variables and the CLI explicitly bypasses proxies
for the local budget relay. Direct unauthenticated HTTP401 and subsequent real
GLM responses verify this node's route; other nodes require their own checks.
The CLI is the retained tool executor, while the paid model is GLM. Its displayed
USD estimate does not imply a Claude API charge. A validated cached candidate
can be evaluated entirely locally, without another proposer request.

The USD scenarios below are the earlier international-service comparison. They
exclude taxes/GPU/retries and are planning examples, not measurements or caps.

WHALE uses two distinct models: a **proposer** edits harness programs, while the
**target model** produces task trajectories and receives RSFT updates. Replacing
Claude in the proposer does not require replacing the target VLM or Method E1–E4.
All comparison arms must use the same proposer, tools, search scope and budget.
Changing proposer is a declared reproduction variation, not exact reproduction of
the paper's Claude setting.

## Earlier international API comparison

The [official Z.ai price table](https://docs.z.ai/guides/overview/pricing) lists
regular input/output rates of $0.15/$0.50 per million tokens. Its 50% launch
discount ends **2026-09-09 24:00 UTC+8**. Plan future work at regular prices.
The [Anthropic price table](https://platform.claude.com/docs/en/about-claude/pricing)
lists Opus 4.7 at $5/$25. At equal 200k input plus 20k output tokens per proposer
session, no cache discounts:

| Scope | Opus 4.7 | GLM-5.3-Flash regular | GLM promotion |
|---|---:|---:|---:|
| One session | $1.50 | $0.04 | $0.02 |
| 20-session engineering allowance | $30 | $0.80 | $0.40 |
| Default Chess WHALE run: 95 sessions | $142.50 | $3.80 | $1.90 |
| Illustrative 60-run matrix: 3,900 sessions | $5,850 | $156 | $78 |

The last line assumes both domains/sizes use the default Chess schedule:
`(ceil(256/13)-1)*5=95` sessions for each FST/WHALE/VETO run; harness-only has 40;
weight-only has zero. Multiply their sum by 2 domains×2 sizes×3 seeds. Each session
already proposes three candidates; do not multiply this cost by three again.
Actual visual schedules are not frozen. Additional ablations and F0 reproduction
are excluded. Tokenizer, tool-call count, retries and candidate quality can change
the totals. Equal-token price ratio is 37.5×, not a measured end-to-end speedup.

`python -m ours.budget` reproduces these arithmetic scenarios. The original Opus
estimate is preserved in `results/proposer-budget-20260909.json`; the extended
estimate is `results/proposer-budget-with-glm-20260909.json`.

## Tool interface and verified domestic route

[Z.ai documents using GLM inside Claude Code](https://docs.z.ai/devpack/tool/claude).
The tool application and the model provider are separate. Its documented endpoint
is `https://api.z.ai/api/anthropic`; WHALE's proposer model flag must explicitly
become `glm-5.3-flash`. Changing only a default alias may be overridden by the
upstream explicit `--model claude-opus-4-7` argument.

`ours.proposer_profiles.GLM_FLASH.claude_environment(os.environ)` prepares an
isolated child-process environment from `ZAI_API_KEY`, maps it to
`ANTHROPIC_AUTH_TOKEN`, and sets the endpoint/model. It neither stores a key nor
modifies global Claude settings. Only the key variable's presence is printed by
the CLI. That international profile remains configuration-only. The new domestic
`GLM_FLASH_CN` profile uses `ZHIPU_API_KEY`; the actual test uses a parent budget
relay and an ephemeral loopback token instead of giving the real key to the child.

The signed native Claude Code2.1.236 binary is now installed locally under `data/`,
with global settings and existing environments unchanged. `run_proposer_probe.py`
uses the native WHALE wrapper with bounded tools/turns/time and Landlock file
confinement. After four documented post-Read timeouts, offline replay isolated an
HTTP response-delimitation issue. Buffering the unchanged SSE body and supplying
Content-Length enabled the fifth attempt to finish Read→Edit→DONE. An AST check
verified exactly the requested USER_PROMPT edit. This is a transport check,
not evidence of proposer search quality or equivalence to Claude's decisions.
The official integration pages also discuss Coding Plan usage; the API receipt's
usage and local journal are not a query of the user's invoice or billing route.

## Local deployment

The [official GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash)
provides open weights and vLLM/SGLang serving references. It has about 320B total
parameters and 18B active. Parameter arithmetic gives roughly 640 GB BF16,
320 GB FP8 or 160 GB ideal 4-bit weight storage, **before** scales, buffers and KV
cache. Active parameters do not determine total weight residency. Actual checkpoint
storage and compatible quantization must be checked before any download.

Today's account lookup allowed at most 4 concurrent GPUs and 48 CPUs. Four 24-GB
4090 cards cannot hold even the ideal 4-bit weights entirely in VRAM. Four 80-GB
H800s might support a suitable quantized deployment, but require a real memory and
throughput test and would occupy the GPU concurrency budget needed by the target.
No throughput, wall time or GPU-hours for this deployment have been measured.
Before the subsequent4.55GB Qwen3.5-2B download, the disk lookup showed approximately
293GiB available. Refresh storage before considering a full GLM FP8 download.

A smaller local coding model is another option. Qwen3-VL-32B-Instruct is already
present in the local HF cache, although completeness, suitable quantization and
proposer quality have not been verified. Do not infer coding suitability from cache
presence. An OpenAI-compatible local server needs a tool-agent adapter; it cannot
be substituted as an Anthropic endpoint by changing a URL alone. The local profile
records this distinction and refuses that unsupported conversion.

The local **target** connection was subsequently tested with cached
Qwen2.5-VL-3B-Instruct: job221552 completed 32 real visual/no-image generations,
one H800, 44 allocated seconds, 0 external API cost. This establishes local visual
inference, not proposer coding quality or a running local tool-agent server.
The raw outputs and independent rescore are recorded in `EXPERIMENTS.md` and
`results/visual-smoke-221552-summary.json`. No model download was needed.

A later baseline runtime check downloaded pinned official Qwen3.5-2B weights and
ran the original Chess evaluator with a new local HF compatibility client:
job221577,8 generations,25 allocated seconds,0 external API cost. The reduced
non-thinking engineering configuration yielded0/8 solved and is not the paper's
baseline configuration. This client implements **target** inference, not the
proposer's file-editing tool loop. Neither local smoke establishes proposer quality.

Choose using *total completion time*: GPU allocation/queue, model loading, proposer
latency, target runtime and retries. GLM API presently looks more economical for
the first engineering comparison because it leaves GPUs available for target work;
this is a planning judgment, not a measured performance result. Before formal runs,
measure a fixed set of proposal tasks for valid-code rate, contract compliance,
successful evaluation rate, token cost and latency. Lower API prices alone do not
establish that the resulting search is equally effective.
