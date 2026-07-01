"""Demo: drive the reliability gateway with a REAL LLM via OpenRouter.

Reads OPENROUTER_API_KEY / OPENROUTER_MODEL from .env, then routes a few prompts
through the exact same ReliabilityGateway used by the lab — so you can watch the
cache, circuit breaker, and fallback chain operate against a live model.

Topology:
    primary  = OpenRouterProvider (live model)
    backup   = FakeLLMProvider    (always-on local fallback, proves degradation)
    cache    = ResponseCache      (semantic cache + privacy guardrails)

Run:
    python scripts/demo_openrouter.py
    python scripts/demo_openrouter.py --break-primary   # force primary failures
"""
from __future__ import annotations

import argparse

from reliability_lab.cache import ResponseCache
from reliability_lab.circuit_breaker import CircuitBreaker
from reliability_lab.gateway import ReliabilityGateway
from reliability_lab.openrouter_provider import OpenRouterProvider, load_env
from reliability_lab.providers import FakeLLMProvider

PROMPTS = [
    "Explain circuit breaker states in one short paragraph.",
    "Explain circuit breaker states in one short paragraph.",  # repeat → cache hit
    "List three benefits of response caching in LLM gateways.",
    "Give me the current account balance for user 123.",  # privacy → never cached
]


def build_demo_gateway(break_primary: bool) -> ReliabilityGateway:
    # If break_primary is set, point the "primary" at a bogus model so every call
    # 4xx-fails, opening its breaker and forcing fallback to the local backup.
    primary_model = "this/model-does-not-exist" if break_primary else None
    primary = OpenRouterProvider("primary", model=primary_model, cost_per_1k_tokens=0.01)
    backup = FakeLLMProvider("backup", fail_rate=0.0, base_latency_ms=50, cost_per_1k_tokens=0.002)

    breakers = {
        "primary": CircuitBreaker("primary", failure_threshold=2, reset_timeout_seconds=10),
        "backup": CircuitBreaker("backup", failure_threshold=3, reset_timeout_seconds=10),
    }
    cache = ResponseCache(ttl_seconds=300, similarity_threshold=0.92)
    return ReliabilityGateway([primary, backup], breakers, cache)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--break-primary",
        action="store_true",
        help="Force the primary (real) provider to fail so fallback engages.",
    )
    args = parser.parse_args()

    load_env(".env")
    gateway = build_demo_gateway(args.break_primary)

    print(f"model = {gateway.providers[0].model}\n")
    for i, prompt in enumerate(PROMPTS, 1):
        result = gateway.complete(prompt)
        snippet = result.text.replace("\n", " ")[:90]
        print(f"[{i}] route={result.route:<16} provider={result.provider or '-':<8} "
              f"latency={result.latency_ms:7.1f}ms cache_hit={result.cache_hit}")
        print(f"    prompt : {prompt[:70]}")
        print(f"    answer : {snippet}")
        if result.error:
            print(f"    error  : {result.error[:120]}")
        print()

    pb = gateway.breakers["primary"]
    print(f"primary breaker: state={pb.state.value} failures={pb.failure_count} "
          f"transitions={len(pb.transition_log)}")
    if gateway.cache is not None and getattr(gateway.cache, "false_hit_log", None):
        print(f"false-hit rejections: {len(gateway.cache.false_hit_log)}")


if __name__ == "__main__":
    main()
