"""Benchmark an OpenAI-compatible LLM endpoint across concurrency levels and input lengths."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

# Common English words; most map to a single token in typical BPE vocabularies.
WORDS = (
    "the of and to in is was for on that with as by at from his her they this have had are not "
    "but be or an which one you were all we she there been their has more time if will would "
    "about into who what so can out up said only other new some could these two may first then "
    "do any like my now over such our man me even most made after also did many before must "
    "through back years where much your way well down should because each just those people how "
    "too little state good very make world still own see men work long get here between both life "
    "being under never day same another know while last might us great old year off come since "
    "against go came right used take three house water room place system story river music city"
).split()


# --------------------------------------------------------------------------- data


@dataclass
class RequestResult:
    ok: bool
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft: float = 0.0  # seconds to first token
    latency: float = 0.0  # end-to-end seconds
    itls: list[float] = field(default_factory=list)  # inter-chunk latencies
    error: str = ""

    @property
    def tpot(self) -> float:
        """Time per output token after the first one."""
        if self.output_tokens <= 1:
            return 0.0
        return (self.latency - self.ttft) / (self.output_tokens - 1)


@dataclass
class LevelSummary:
    input_tokens_target: int
    concurrency: int
    requests: int
    successes: int
    failures: int
    duration_s: float
    req_per_s: float
    avg_prompt_tokens: float
    avg_output_tokens: float
    output_tok_per_s: float  # aggregate decode throughput across all requests
    total_tok_per_s: float  # (prompt + output) / duration
    per_req_tok_per_s: float  # mean single-stream decode speed
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_mean_ms: float
    itl_p99_ms: float
    latency_mean_s: float
    latency_p99_s: float
    first_error: str = ""


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


# --------------------------------------------------------------------------- client


class Bench:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base = normalize_base_url(args.endpoint)
        headers = {"Content-Type": "application/json"}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        self.client = httpx.AsyncClient(
            base_url=self.base,
            headers=headers,
            timeout=httpx.Timeout(args.timeout, connect=30),
            limits=limits,
        )
        self.model: str = args.model or ""
        self.tok_per_word = 1.0
        self.overhead = 0.0

    async def close(self) -> None:
        await self.client.aclose()

    async def detect_model(self) -> str:
        r = await self.client.get("/models")
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        if not models:
            raise RuntimeError(f"No models returned by {self.base}/models")
        if len(models) > 1:
            console.print(f"[yellow]Multiple models found: {models}; using the first one.[/]")
        return models[0]

    def payload(self, prompt: str, max_tokens: int, stream: bool) -> dict:
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": stream,
        }
        if self.args.api == "chat":
            body["messages"] = [{"role": "user", "content": prompt}]
        else:
            body["prompt"] = prompt
        if stream:
            body["stream_options"] = {"include_usage": True}
        if self.args.ignore_eos:
            # vLLM / SGLang / TGI-style extension: keep generating until max_tokens.
            body["ignore_eos"] = True
        return body

    @property
    def path(self) -> str:
        return "/chat/completions" if self.args.api == "chat" else "/completions"

    # ---- prompt construction -------------------------------------------------

    def make_words(self, n_words: int, rng: random.Random) -> str:
        # A random prefix per request defeats prefix caching across requests.
        return " ".join(rng.choice(WORDS) for _ in range(max(n_words, 1)))

    async def count_prompt_tokens(self, prompt: str) -> int:
        r = await self.client.post(self.path, json=self.payload(prompt, 1, stream=False))
        r.raise_for_status()
        return int(r.json()["usage"]["prompt_tokens"])

    async def calibrate(self) -> None:
        """Fit prompt_tokens ~= overhead + tok_per_word * words using the server's tokenizer."""
        rng = random.Random(0)
        n1, n2 = 200, 2000
        t1 = await self.count_prompt_tokens(self.make_words(n1, rng))
        t2 = await self.count_prompt_tokens(self.make_words(n2, rng))
        self.tok_per_word = max((t2 - t1) / (n2 - n1), 0.1)
        self.overhead = max(t1 - n1 * self.tok_per_word, 0.0)
        console.print(
            f"Calibration: {self.tok_per_word:.3f} tokens/word, "
            f"~{self.overhead:.0f} tokens template overhead"
        )

    def build_prompt(self, target_tokens: int, rng: random.Random) -> str:
        n_words = round((target_tokens - self.overhead) / self.tok_per_word)
        return self.make_words(n_words, rng)

    # ---- one request ---------------------------------------------------------

    async def one_request(self, prompt: str) -> RequestResult:
        body = self.payload(prompt, self.args.output_tokens, stream=True)
        start = time.perf_counter()
        first = 0.0
        last = start
        itls: list[float] = []
        chunks = 0
        usage: dict | None = None
        try:
            async with self.client.stream("POST", self.path, json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode(errors="replace")
                    return RequestResult(ok=False, error=f"HTTP {r.status_code}: {text[:300]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    if not has_token(obj):
                        continue
                    now = time.perf_counter()
                    if not first:
                        first = now
                    else:
                        itls.append(now - last)
                    last = now
                    chunks += 1
        except Exception as e:  # noqa: BLE001
            return RequestResult(ok=False, error=f"{type(e).__name__}: {e}")
        end = time.perf_counter()
        if not first:
            return RequestResult(ok=False, error="no tokens received")
        return RequestResult(
            ok=True,
            prompt_tokens=int(usage.get("prompt_tokens", 0)) if usage else 0,
            output_tokens=int(usage.get("completion_tokens", chunks)) if usage else chunks,
            ttft=first - start,
            latency=end - start,
            itls=itls,
        )

    # ---- one (input_len, concurrency) level ----------------------------------

    async def run_level(self, input_tokens: int, concurrency: int) -> LevelSummary:
        n_requests = max(concurrency * self.args.rounds, self.args.min_requests)
        rng = random.Random(input_tokens * 1000 + concurrency)
        prompts = [self.build_prompt(input_tokens, rng) for _ in range(n_requests)]
        queue: asyncio.Queue[str] = asyncio.Queue()
        for p in prompts:
            queue.put_nowait(p)
        results: list[RequestResult] = []

        async def worker() -> None:
            while True:
                try:
                    p = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results.append(await self.one_request(p))

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        duration = time.perf_counter() - start
        return summarize(input_tokens, concurrency, results, duration)


def has_token(obj: dict) -> bool:
    for c in obj.get("choices") or []:
        delta = c.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
            return True
        if c.get("text"):
            return True
    return False


def summarize(
    input_tokens: int, concurrency: int, results: list[RequestResult], duration: float
) -> LevelSummary:
    ok = [r for r in results if r.ok]
    errors = [r.error for r in results if not r.ok]
    out_total = sum(r.output_tokens for r in ok)
    in_total = sum(r.prompt_tokens for r in ok)
    tpots = [r.tpot for r in ok if r.tpot > 0]
    ttfts = [r.ttft * 1000 for r in ok]
    lats = [r.latency for r in ok]
    itls = [x * 1000 for r in ok for x in r.itls]
    return LevelSummary(
        input_tokens_target=input_tokens,
        concurrency=concurrency,
        requests=len(results),
        successes=len(ok),
        failures=len(errors),
        duration_s=duration,
        req_per_s=len(ok) / duration if duration else 0.0,
        avg_prompt_tokens=in_total / len(ok) if ok else 0.0,
        avg_output_tokens=out_total / len(ok) if ok else 0.0,
        output_tok_per_s=out_total / duration if duration else 0.0,
        total_tok_per_s=(out_total + in_total) / duration if duration else 0.0,
        per_req_tok_per_s=mean([1 / t for t in tpots]),
        ttft_mean_ms=mean(ttfts),
        ttft_p50_ms=pct(ttfts, 50),
        ttft_p99_ms=pct(ttfts, 99),
        tpot_mean_ms=mean(tpots) * 1000,
        itl_p99_ms=pct(itls, 99),
        latency_mean_s=mean(lats),
        latency_p99_s=pct(lats, 99),
        first_error=errors[0] if errors else "",
    )


def normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    for suffix in ("/chat/completions", "/completions", "/models"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    if not url.endswith("/v1"):
        url += "/v1"
    return url


# --------------------------------------------------------------------------- output


def results_table(rows: list[LevelSummary], title: str) -> Table:
    t = Table(title=title, header_style="bold cyan")
    for name in ["Input\ntok", "Conc", "OK", "Total out\ntok/s", "1-req\ntok/s", "TTFT\navg ms", "TTFT\np99", "TPOT\nms", "E2E\navg s"]:
        t.add_column(name, justify="right", no_wrap=True)
    for r in rows:
        ok = f"{r.successes}/{r.requests}"
        t.add_row(
            str(r.input_tokens_target),
            str(r.concurrency),
            ok if r.failures == 0 else f"[red]{ok}[/]",
            f"[bold green]{r.output_tok_per_s:,.0f}[/]",
            f"{r.per_req_tok_per_s:.1f}",
            f"{r.ttft_mean_ms:,.0f}",
            f"{r.ttft_p99_ms:,.0f}",
            f"{r.tpot_mean_ms:.1f}",
            f"{r.latency_mean_s:.1f}",
        )
    return t


def save_results(rows: list[LevelSummary], meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = meta["model"].replace("/", "_")
    base = out_dir / f"{safe_model}_{stamp}"
    with open(base.with_suffix(".json"), "w") as f:
        json.dump({"meta": meta, "results": [asdict(r) for r in rows]}, f, indent=2)
    with open(base.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    return base


# --------------------------------------------------------------------------- main


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible LLM endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("endpoint", help="Endpoint address, e.g. http://localhost:8000 or http://host:8000/v1")
    p.add_argument("--model", help="Model name (auto-detected from /v1/models if omitted)")
    p.add_argument("--api-key", default=None, help="API key (Bearer token), or set OPENAI_API_KEY")
    p.add_argument("-c", "--concurrency", type=parse_int_list, default=[1, 2, 4, 8, 16, 32, 64],
                   help="Comma-separated concurrency levels")
    p.add_argument("-i", "--input-tokens", type=parse_int_list, default=[128, 512, 1024, 2048, 4096],
                   help="Comma-separated input (prompt) token lengths")
    p.add_argument("-o", "--output-tokens", type=int, default=128, help="max_tokens per request")
    p.add_argument("--rounds", type=int, default=1,
                   help="Requests per level = concurrency * rounds (at least --min-requests)")
    p.add_argument("--min-requests", type=int, default=4, help="Minimum requests per level")
    p.add_argument("--api", choices=["chat", "completions"], default="chat",
                   help="Use /v1/chat/completions or /v1/completions")
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                   help="Don't send ignore_eos (needed for servers that reject unknown params)")
    p.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds")
    p.add_argument("--out-dir", type=Path, default=Path("results"), help="Where to write JSON/CSV")
    args = p.parse_args(argv)
    if args.api_key is None:
        import os

        args.api_key = os.environ.get("OPENAI_API_KEY")
    return args


async def amain(args: argparse.Namespace) -> int:
    bench = Bench(args)
    try:
        if not bench.model:
            bench.model = await bench.detect_model()
        console.print(f"Endpoint: [bold]{bench.base}[/]   Model: [bold]{bench.model}[/]")

        await bench.calibrate()

        console.print("Warming up...")
        warm = await bench.one_request(bench.build_prompt(64, random.Random(1)))
        if not warm.ok:
            console.print(f"[red]Warmup failed:[/] {warm.error}")
            if "ignore_eos" in warm.error:
                console.print("Hint: retry with --no-ignore-eos")
            return 1
        if warm.output_tokens < args.output_tokens * 0.9:
            console.print(
                f"[yellow]Warmup produced only {warm.output_tokens}/{args.output_tokens} tokens; "
                "the server may be ignoring ignore_eos, so tok/s figures use actual counts.[/]"
            )

        rows: list[LevelSummary] = []
        total = len(args.input_tokens) * len(args.concurrency)
        n = 0
        for input_tokens in args.input_tokens:
            for conc in args.concurrency:
                n += 1
                with console.status(f"[{n}/{total}] input={input_tokens} concurrency={conc} ..."):
                    s = await bench.run_level(input_tokens, conc)
                rows.append(s)
                console.print(
                    f"[{n}/{total}] input={input_tokens:<5} conc={conc:<4} "
                    f"ok={s.successes}/{s.requests}  out={s.output_tok_per_s:,.1f} tok/s  "
                    f"per-req={s.per_req_tok_per_s:.1f} tok/s  TTFT={s.ttft_mean_ms:,.0f} ms"
                )
                if s.first_error:
                    console.print(f"   [red]error:[/] {s.first_error}")

        console.print()
        console.print(results_table(rows, f"{bench.model} @ {bench.base}"))
        meta = {
            "endpoint": bench.base,
            "model": bench.model,
            "api": args.api,
            "output_tokens": args.output_tokens,
            "ignore_eos": args.ignore_eos,
            "timestamp": datetime.now().isoformat(),
        }
        base = save_results(rows, meta, args.out_dir)
        console.print(f"Saved {base}.json and {base}.csv")
        return 0
    finally:
        await bench.close()


def main() -> None:
    args = parse_args()
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/]")
        sys.exit(130)
    except httpx.HTTPError as e:
        console.print(f"[red]HTTP error:[/] {e}")
        sys.exit(1)
