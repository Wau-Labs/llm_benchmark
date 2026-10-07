# llm-benchmark

Benchmarks an OpenAI-compatible endpoint (vLLM, SGLang, TGI, llama.cpp, LM Studio, etc.) across a
grid of **concurrency × input length**. Each request asks for a fixed number of output tokens
(1024 by default), so you get comparable tokens/s numbers.

## Setup

```bash
uv venv --allow-existing
uv pip install git+https://github.com/Wau-Labs/llm_benchmark.git
```

## Usage

```bash
# Model is auto-detected from /v1/models
uv run llm-benchmark http://localhost:8000

# Custom grid
uv run llm-benchmark http://gpu-box:8000/v1 -c 1,8,32,128 -i 256,2048,8192

# With auth / explicit model
uv run llm-benchmark https://api.example.com --api-key sk-... --model my-model
```

Key options (`--help` lists all of them):

| Option | Default | Meaning |
|---|---|---|
| `-c/--concurrency` | `1,2,4,8,16,32,64` | Concurrent in-flight requests |
| `-i/--input-tokens` | `128,512,1024,2048,4096` | Target prompt length in tokens |
| `-o/--output-tokens` | `128` | `max_tokens` per request |
| `--rounds` | `1` | Requests per level = concurrency × rounds |
| `--api` | `chat` | `chat` or `completions` endpoint |
| `--no-ignore-eos` | off | Don't send `ignore_eos` (for servers that reject unknown fields) |

## How it works

- **Prompt length**: the script sends two probe prompts and reads `usage.prompt_tokens` to
  calibrate words→tokens using the server's own tokenizer, including chat template overhead.
  Each request gets different random text, so prefix caching doesn't skew results.
- **Fixed output**: sends `ignore_eos: true` so the model generates exactly `max_tokens`
  (supported by vLLM/SGLang). Token counts come from the server's `usage` field.
- **Warmup**: one request runs before the sweep.

## Metrics

- **Total out tok/s**: all output tokens ÷ wall time for the level (aggregate throughput)
- **1-req tok/s**: average decode speed of a single stream (1 / TPOT)
- **TTFT**: time to first token (avg and p99)
- **TPOT**: time per output token after the first one
- **E2E**: end-to-end request latency

Results print as a table and are saved to `results/<model>_<timestamp>.{json,csv}`, which also
include extra fields (total tok/s with prompt tokens, req/s, ITL p99, and the first error).
