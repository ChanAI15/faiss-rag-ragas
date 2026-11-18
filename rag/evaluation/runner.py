"""Execute RAGAS metrics evaluation using native Google GenAI client."""

import asyncio
from collections import deque
import csv
import json
import math
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from datasets import Dataset
from google.genai import types
from langchain_core.outputs import Generation, LLMResult
from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.run_config import RunConfig

from rag.config import (
    EMBEDDING_MODEL,
    EVAL_JUDGE_MODEL,
    EVAL_MAX_CALLS_PER_MIN,
    EVAL_MAX_INPUT_TOKENS_PER_MIN,
    EVAL_OUTPUT_DIR,
    TOP_K,
    client,
    require_api_key,
)
from rag.evaluation.collector import collect_evaluation_samples
from rag.evaluation.dataset import load_evaluation_dataset


class InputTokenRateLimiter:
    """Sliding-window rate limiter enforcing both input tokens/min and calls/min."""

    def __init__(
        self,
        max_tokens_per_min: int = EVAL_MAX_INPUT_TOKENS_PER_MIN,
        max_calls_per_min: int = EVAL_MAX_CALLS_PER_MIN,
        window_seconds: float = 60.0,
    ):
        self.max_tokens_per_min = max_tokens_per_min
        self.max_calls_per_min = max_calls_per_min
        self.window_seconds = window_seconds
        self._history: deque[list[float | int]] = deque()
        self._lock = threading.Lock()

    @staticmethod
    def estimate_input_tokens(prompt_str: str, model: str) -> int:
        """Count or estimate input tokens for a prompt before sending to the model."""
        try:
            token_resp = client.models.count_tokens(
                model=model,
                contents=prompt_str,
            )
            if getattr(token_resp, "total_tokens", None):
                return int(token_resp.total_tokens)
        except Exception:
            pass
        # Fallback heuristic (~3.5 chars per token + safety buffer)
        return max(1, math.ceil(len(prompt_str) / 3.5) + 32)

    def _prune(self, now: float) -> None:
        while self._history and (now - self._history[0][0]) >= self.window_seconds:
            self._history.popleft()

    def wait_for_capacity(self, prompt_str: str, model: str) -> list[float | int]:
        """Block until the current 60s window has room for both the call count and input tokens."""
        estimated_tokens = self.estimate_input_tokens(prompt_str, model)

        with self._lock:
            while True:
                now = time.monotonic()
                self._prune(now)

                used_calls = len(self._history)
                used_tokens = sum(int(entry[1]) for entry in self._history)

                calls_exceeded = (
                    self.max_calls_per_min > 0
                    and used_calls >= self.max_calls_per_min
                )
                tokens_exceeded = (
                    self.max_tokens_per_min > 0
                    and used_calls > 0
                    and (used_tokens + estimated_tokens) > self.max_tokens_per_min
                )

                if not calls_exceeded and not tokens_exceeded:
                    entry: list[float | int] = [now, estimated_tokens]
                    self._history.append(entry)
                    print(
                        f"  [RateLimiter] Call {used_calls + 1}/{self.max_calls_per_min} in window | "
                        f"Input tokens: ~{estimated_tokens} "
                        f"(window total: {used_tokens + estimated_tokens}/{self.max_tokens_per_min})"
                    )
                    return entry

                oldest_ts = float(self._history[0][0])
                wait_seconds = max(1.0, (self.window_seconds - (now - oldest_ts)) + 1.0)
                reason = (
                    f"call limit ({used_calls}/{self.max_calls_per_min} calls/min)"
                    if calls_exceeded
                    else f"token limit ({used_tokens} + {estimated_tokens} > {self.max_tokens_per_min} tokens/min)"
                )
                print(
                    f"  [RateLimiter] Reached {reason}. "
                    f"Waiting {wait_seconds:.1f}s for next minute window..."
                )
                time.sleep(wait_seconds)

    def update_actual_tokens(self, entry: list[float | int], actual_tokens: int | None) -> None:
        if actual_tokens and actual_tokens > 0:
            with self._lock:
                entry[1] = int(actual_tokens)


def _clean_llm_text(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    if not text.startswith("{") and not text.startswith("["):
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        if starts:
            start = min(starts)
            end = max(text.rfind("}"), text.rfind("]"))
            if end > start:
                text = text[start : end + 1]
    return text


class GoogleGenAIRagasLLM(BaseRagasLLM):
    """Native RAGAS LLM adapter using the google-genai client with input-token & call rate limiting."""

    def __init__(
        self,
        model: str = EVAL_JUDGE_MODEL,
        max_tokens_per_min: int = EVAL_MAX_INPUT_TOKENS_PER_MIN,
        max_calls_per_min: int = EVAL_MAX_CALLS_PER_MIN,
    ):
        super().__init__()
        self.model = model
        self.rate_limiter = InputTokenRateLimiter(
            max_tokens_per_min=max_tokens_per_min,
            max_calls_per_min=max_calls_per_min,
        )

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        require_api_key()
        if hasattr(prompt, "to_string"):
            prompt_str = prompt.to_string()
        elif hasattr(prompt, "text"):
            prompt_str = prompt.text
        else:
            prompt_str = str(prompt)

        prompt_generations = []
        for _ in range(max(1, n)):
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                entry = self.rate_limiter.wait_for_capacity(prompt_str, self.model)
                try:
                    response = client.models.generate_content(
                        model=self.model,
                        contents=prompt_str,
                        config=types.GenerateContentConfig(
                            temperature=temperature if temperature is not None else 0.2,
                        ),
                    )
                    usage = getattr(response, "usage_metadata", None)
                    actual_tokens = getattr(usage, "prompt_token_count", None)
                    self.rate_limiter.update_actual_tokens(entry, actual_tokens)

                    raw_text = response.text or ""
                    if not raw_text:
                        raise RuntimeError(
                            f"Empty judge response. finish_reason={getattr(response, 'finish_reason', None)}"
                        )
                    text = _clean_llm_text(raw_text)
                    prompt_generations.append(Generation(text=text))
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = any(
                        token in err_str.upper()
                        for token in ("429", "RESOURCE_EXHAUSTED", "RATE_LIMIT", "QUOTA")
                    )
                    if is_rate_limit and attempt < max_retries:
                        print(
                            f"  [RateLimiter] 429/Quota hit on attempt {attempt}/{max_retries}. "
                            f"Sleeping 60s before retrying..."
                        )
                        time.sleep(60.0)
                        continue
                    raise RuntimeError(
                        f"Google GenAI judge call failed for model '{self.model}': {e}"
                    ) from e

        return LLMResult(generations=[prompt_generations])

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return await asyncio.to_thread(
            self.generate_text, prompt, n, temperature, stop, callbacks
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True


class GoogleGenAIRagasEmbeddings(BaseRagasEmbeddings):
    """Native RAGAS Embeddings adapter using the google-genai client."""

    def __init__(self, model: str = EMBEDDING_MODEL):
        super().__init__()
        self.model = model

    def embed_query(self, text: str, **kwargs) -> list[float]:
        require_api_key()
        response = client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return [float(v) for v in response.embeddings[0].values]

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        require_api_key()
        if not texts:
            return []

        embeddings = []
        for text in texts:
            response = client.models.embed_content(
                model=self.model,
                contents=text,
            )
            embeddings.append([float(v) for v in response.embeddings[0].values])

        return embeddings

    async def aembed_query(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text, **kwargs)

    async def aembed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts, **kwargs)


def get_evaluation_metrics(
    judge_llm: BaseRagasLLM,
    judge_embeddings: BaseRagasEmbeddings,
) -> list:
    """Initialize standard RAG evaluation metrics bound to the Google GenAI judge."""
    faithfulness.llm = judge_llm
    answer_relevancy.llm = judge_llm
    answer_relevancy.embeddings = judge_embeddings
    context_precision.llm = judge_llm
    context_recall.llm = judge_llm

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def evaluate_rag(
    questions: list[dict[str, Any]] | None = None,
    samples: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    top_k: int = TOP_K,
    judge_model: str = EVAL_JUDGE_MODEL,
    reuse_samples_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run full RAGAS evaluation on pipeline samples and persist results."""
    # 1. Obtain samples (either from cache or by executing pipeline)
    if reuse_samples_path:
        cache_file = Path(reuse_samples_path)
        print(f"Loading cached samples from {cache_file}...")
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        samples = cached.get("samples", cached)
        if limit:
            samples = samples[:limit]
    elif samples is None:
        if questions is None:
            questions = load_evaluation_dataset(limit=limit)
        samples, _ = collect_evaluation_samples(questions, top_k=top_k)

    if not samples:
        raise ValueError("No evaluation samples available to evaluate.")

    # 2. Build RAGAS Dataset
    eval_data = {
        "user_input": [s["user_input"] for s in samples],
        "response": [s["response"] for s in samples],
        "retrieved_contexts": [s["retrieved_contexts"] for s in samples],
        "reference": [s["reference"] for s in samples],
    }
    dataset = Dataset.from_dict(eval_data)

    # 3. Setup Judge & Metrics
    print(f"\nInitializing RAGAS evaluation with judge model: {judge_model}...")
    judge_llm = GoogleGenAIRagasLLM(model=judge_model)
    judge_embeddings = GoogleGenAIRagasEmbeddings(model=EMBEDDING_MODEL)
    metrics = get_evaluation_metrics(judge_llm, judge_embeddings)

    if not metrics:
        raise RuntimeError("No RAGAS metrics could be loaded. Ensure 'ragas' is installed properly.")

    # 4. Run Evaluation (sequential workers so rate limiter paces calls cleanly)
    print(
        f"Running evaluation on {len(samples)} samples across {len(metrics)} metrics "
        f"(limit: {EVAL_MAX_CALLS_PER_MIN} calls/min, {EVAL_MAX_INPUT_TOKENS_PER_MIN} input tokens/min)..."
    )
    run_config = RunConfig(max_workers=1, timeout=600)
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    # 5. Format & Save Reports
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_csv = EVAL_OUTPUT_DIR / f"report_{timestamp}.csv"
    report_json = EVAL_OUTPUT_DIR / f"report_{timestamp}.json"

    df = None
    try:
        df = results.to_pandas()
        df.to_csv(report_csv, index=False, encoding="utf-8")
    except Exception as e:
        print(f"Could not export detailed RAGAS dataframe: {e}")

    metric_scores = {}
    if df is not None:
        known_metrics = {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }
        for metric in known_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if not values.empty:
                    metric_scores[metric] = float(values.mean())

    summary_data = {
        "timestamp": timestamp,
        "sample_count": len(samples),
        "judge_model": judge_model,
        "scores": metric_scores,
        "samples": samples,
    }
    report_json.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6. Display Clean Summary Table
    print("\n" + "=" * 45)
    print("           RAGAS EVALUATION REPORT           ")
    print("=" * 45)
    print(f"{'Metric':<28} | {'Score':<10}")
    print("-" * 45)
    if metric_scores:
        for metric_name, score in metric_scores.items():
            print(f"{metric_name:<28} | {score:.4f}")
    else:
        print("No metric scores were produced.")
        print("Check the judge model response and RAGAS parser errors above.")
    print("=" * 45)
    print(f"Saved CSV Report:  {report_csv}")
    print(f"Saved JSON Report: {report_json}\n")

    return summary_data
