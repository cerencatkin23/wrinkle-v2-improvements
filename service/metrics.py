from __future__ import annotations

try:
    import prometheus_client  # noqa: F401
    from prometheus_client import Counter, Histogram

    PROM_ENABLED = True
except ImportError:  # pragma: no cover - fallback for optional dependency
    PROM_ENABLED = False

    class _NoopMetric:
        def labels(self, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

    Counter = lambda *args, **kwargs: _NoopMetric()  # type: ignore
    Histogram = lambda *args, **kwargs: _NoopMetric()  # type: ignore

REQUEST_COUNT = Counter(
    "wrinkle_requests_total",
    "Total HTTP requests",
    ["endpoint", "http_status"],
)

RESULT_COUNT = Counter(
    "wrinkle_results_total",
    "Total analyze results by status",
    ["result_status"],
)

LATENCY_MS = Histogram(
    "wrinkle_request_latency_ms",
    "Request latency in milliseconds",
    ["endpoint"],
    buckets=(5, 10, 25, 50, 75, 100, 150, 250, 500, 1000, 2000, 5000),
)

NO_SCORE_REASONS = Counter(
    "wrinkle_no_score_reasons_total",
    "Count of NO_SCORE reasons",
    ["reason"],
)
