# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

_ASSETS = Path(__file__).parent.parent / "assets" / "synthetic_themes"

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous agent. Use the available tools to complete the given task, "
    "reason step by step, and produce a concise final answer. Prefer read-only actions first."
)


class Theme(BaseModel):
    """A synthetic-session theme: the content layer that makes a generated session
    look like a real workload in some domain.

    Required fields (`verbs`, `entities`, `tool_names`, `result_templates` with a
    `default` key, `objective_template`) define the core content. The remaining
    fields are optional, each with a safe empty/`None` default:

    - `tool_descriptions`: per-tool one-line descriptions (keyed by base tool name),
      emitted into both the top-level and nested `function.description` of the tool
      schema. Missing entries fall back to a generic sentence; suffixed duplicates
      (`get_bp_stats_7`) reuse their base tool's description.
    - `tool_parameters`: per-tool JSON-Schema `parameters` object; tools without one
      get a generic non-empty schema so no tool is parameterless.
    - `intro_doc_templates`: long "someone pasted this" documents (an incident ticket,
      a metrics dump, a config excerpt) that open a session's first user turn.
    - `filler_templates`: domain snippets (log lines, metric rows, stack frames) that
      build the per-theme filler word pool used to pad turns; empty -> the shared
      Shakespeare corpus.
    - `payload_templates`: domain snippets for large tool-call payload args (code,
      SQL, a drafted answer); empty -> falls back to `filler_templates`.
    - `compaction_summary_template`: recap sentence for a context-compaction round.
    - `followup_templates` / `followup_connectives`: phrasing for follow-up turns.

    Template placeholders follow field-name heuristics (`{tN}` time, `{nN}` number,
    a name matching an `entities` category -> that category's pool).
    """

    name: str
    system_prompt: Optional[str] = None
    verbs: list[str]
    entities: dict[str, list[str]]
    tool_names: list[str]
    tool_descriptions: dict[str, str] = {}
    # Per-tool JSON-Schema `parameters` object (keyed by BASE tool name):
    # {"type":"object","properties":{...},"required":[...]}. `_tool_definitions`
    # emits it as the tool's parameters; tools without an entry (and synthetic
    # suffixed duplicates) fall back to a generic non-empty schema so NO tool is
    # ever parameterless -- a parameterless forced tool_choice makes some models
    # emit empty args and then fail to stop, leaking chat-template tokens into
    # `arguments` (observed on Qwen). Empty default so other themes still load.
    tool_parameters: dict[str, dict] = {}
    result_templates: dict[str, str]
    objective_template: str
    followup_templates: list[str] = []
    followup_connectives: list[str] = []
    intro_doc_templates: list[str] = []
    filler_templates: list[str] = []
    # Optional filler source for LARGE tool-call PAYLOAD args (content/code/patch/body/...)
    # -- distinct from `filler_templates` (which pads turns and reads like telemetry).
    # A payload should look like what the tool actually carries: a coding tool's payload
    # is CODE, a DBA tool's is SQL, a research tool's is a DRAFTED ANSWER. Rendered like
    # filler_templates (each snippet gets seeded field values) into a word pool the payload
    # is drawn from. Omit -> payloads fall back to `filler_templates`, then the shared corpus.
    payload_templates: list[str] = []
    # Optional recap sentence prepended to a context-compaction round's fresh prompt,
    # standing in for the dropped transcript. Filled like objective_template ({verb} +
    # entity/pinned placeholders) PLUS {tool_a}/{tool_b}/{tool_c} drawn from this theme's
    # tool catalog, so the recap names the session's real subject and the real tools it
    # used. Omit -> compaction falls back to a bare "Summary of prior context:" marker.
    compaction_summary_template: str = ""


def _validate(theme: Theme) -> Theme:
    if not theme.verbs:
        raise ValueError(f"theme {theme.name}: 'verbs' must be non-empty")
    if not theme.tool_names:
        raise ValueError(f"theme {theme.name}: 'tool_names' must be non-empty")
    if "default" not in theme.result_templates:
        raise ValueError(f"theme {theme.name}: 'result_templates' must include a 'default' key")
    # Any provided tool_parameters spec must be a well-formed JSON-Schema object
    # (fail fast on a malformed theme rather than emitting a broken tool schema).
    for tool, spec in theme.tool_parameters.items():
        if not isinstance(spec, dict) or spec.get("type") != "object" or not isinstance(spec.get("properties"), dict):
            raise ValueError(
                f"theme {theme.name}: tool_parameters[{tool!r}] must be a JSON-Schema object "
                f"with type=='object' and a 'properties' dict"
            )
    return theme


def load_theme(name: str) -> Theme:
    path = _ASSETS / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown synthetic theme {name!r} (looked in {_ASSETS})")
    data = json.loads(path.read_text())
    return _validate(Theme(**data))


# A believable generic ops/SRE incident: a checkout/payments service degrading
# under load. Tools mirror a real on-call toolbox (dashboards, logs, traces,
# deploys, feature flags, dependency health). Every tool has a description and a
# per-tool result template with a realistic shape; the intro doc is a pageable
# incident ticket + a metrics excerpt; filler is more log/metric lines.
GENERIC_THEME = Theme(
    name="generic",
    system_prompt=(
        "You are an on-call site-reliability engineer investigating a production incident. "
        "Use the available observability and deploy tools to find the root cause, reason step "
        "by step, and produce a concise incident summary with a recommended remediation. "
        "Prefer read-only diagnostics before proposing any change."
    ),
    verbs=["Investigate", "Diagnose", "Triage", "Analyze", "Root-cause", "Assess"],
    entities={
        "service": ["checkout-api", "payments-worker", "cart-service", "inventory-svc", "session-gateway"],
        "symptom": [
            "elevated p99 latency",
            "5xx error-rate spike",
            "connection-pool exhaustion",
            "rising GC pause time",
            "request timeouts",
        ],
        "dep": ["postgres-primary", "redis-cache", "kafka-broker", "auth-service", "s3-uploads"],
        "region": ["us-east-1", "us-west-2", "eu-central-1"],
    },
    tool_names=[
        "get_service_health",
        "query_metrics",
        "search_logs",
        "list_recent_deploys",
        "get_dependency_status",
        "get_error_budget",
        "check_feature_flags",
        "get_pod_events",
        "run_synthetic_probe",
        "get_exception_trace",
        "get_config_snapshot",
        "apply_remediation",
    ],
    tool_descriptions={
        "get_service_health": "Return the current health summary (status, p50/p99 latency, error rate) for a named service.",
        "query_metrics": "Query a time-series metric (latency, throughput, saturation) over a window and return sampled points.",
        "search_logs": "Full-text search structured application logs for a service, returning matching lines with timestamps.",
        "list_recent_deploys": "List recent deployments for a service with commit sha, author, and rollout timestamps.",
        "get_dependency_status": "Report reachability and latency of a service's upstream dependencies (DBs, caches, brokers).",
        "get_error_budget": "Return the remaining SLO error budget and burn rate for a service over the trailing window.",
        "check_feature_flags": "List feature-flag states recently changed for a service and who toggled them.",
        "get_pod_events": "Return recent Kubernetes pod events (restarts, OOMKills, evictions) for a service's workload.",
        "run_synthetic_probe": "Run an active synthetic request against a service endpoint and return the observed latency/status.",
        "get_exception_trace": "Fetch the most recent unhandled-exception stack trace captured for a service.",
        "get_config_snapshot": "Return the current effective runtime configuration for a service as a JSON object.",
        "apply_remediation": "Apply a submitted remediation to a service: a config patch or runbook script to roll out.",
    },
    # Realistic SRE-toolbox parameter schemas. Property names that match an
    # `entities` category (`service`, `dep`, `region`) are threaded to the
    # round's pinned subject by the generator; enum/int props exercise the other
    # arg types. Several tools are multi-required-param so complex tool calls are
    # generated (query_metrics, search_logs, run_synthetic_probe). apply_remediation
    # carries a large `body` payload (a remediation script/config, from payload_templates).
    tool_parameters={
        "get_service_health": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Name of the service to summarize health for."},
                "window": {
                    "type": "string",
                    "enum": ["5m", "15m", "1h", "24h"],
                    "description": "Trailing time window to summarize over.",
                },
            },
            "required": ["service"],
        },
        "query_metrics": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["latency", "throughput", "error_rate", "saturation"],
                    "description": "Which time-series metric to query.",
                },
                "service": {"type": "string", "description": "Service whose metric to query."},
                "window": {
                    "type": "string",
                    "enum": ["5m", "15m", "1h", "24h"],
                    "description": "Time window to sample over.",
                },
                "step": {"type": "string", "description": "Sampling resolution (e.g. 1m)."},
            },
            "required": ["metric", "service", "window"],
        },
        "search_logs": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service whose logs to search."},
                "query": {"type": "string", "description": "Full-text search query over log lines."},
                "limit": {"type": "integer", "description": "Maximum number of matching lines to return."},
            },
            "required": ["service", "query"],
        },
        "list_recent_deploys": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service to list deployments for."},
                "limit": {"type": "integer", "description": "How many recent deploys to return."},
            },
            "required": ["service"],
        },
        "get_dependency_status": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service whose upstream dependencies to check."},
                "dep": {"type": "string", "description": "Optional specific dependency to focus on."},
            },
            "required": ["service"],
        },
        "get_error_budget": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service to report the SLO error budget for."},
                "window": {
                    "type": "string",
                    "enum": ["1h", "6h", "24h", "30d"],
                    "description": "Trailing window for the burn-rate calculation.",
                },
            },
            "required": ["service"],
        },
        "check_feature_flags": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service whose feature-flag changes to list."},
            },
            "required": ["service"],
        },
        "get_pod_events": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service whose workload pod events to fetch."},
                "limit": {"type": "integer", "description": "Maximum number of events to return."},
            },
            "required": ["service"],
        },
        "run_synthetic_probe": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service to probe."},
                "endpoint": {"type": "string", "description": "Endpoint path to hit (e.g. /healthz)."},
                "region": {"type": "string", "description": "Region to run the probe from."},
            },
            "required": ["service", "endpoint", "region"],
        },
        "get_exception_trace": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service whose latest exception trace to fetch."},
                "limit": {"type": "integer", "description": "How many recent traces to consider."},
            },
            "required": ["service"],
        },
        "get_config_snapshot": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service whose effective config to snapshot."},
            },
            "required": ["service"],
        },
        "apply_remediation": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service to apply the remediation to."},
                "body": {
                    "type": "string",
                    "description": "The remediation script / config block to apply.",
                    "x-payload-tokens": 120,
                },
            },
            "required": ["service", "body"],
        },
    },
    result_templates={
        "get_service_health": (
            "service={service} status=degraded p50_ms={p50_ms} p99_ms={p99_ms} error_rate_pct={error_rate_pct} "
            "req_per_sec={req_per_sec} as_of={t0}"
        ),
        "query_metrics": (
            "metric=latency_ms service={service} window=15m\n"
            "  {t0}  p99={p99_0}  p50={p50_0}  rps={rps0}\n"
            "  {t1}  p99={p99_1}  p50={p50_1}  rps={rps1}\n"
            "  {t2}  p99={p99_2}  p50={p50_2}  rps={rps2}"
        ),
        "search_logs": (
            "matched {count0} lines for service={service}\n"
            "  {t0} ERROR pool: could not acquire connection within {ms0}ms (in_use={in_use0}/{max0})\n"
            "  {t1} WARN  upstream {dep} responded {status0} after {ms1}ms\n"
            "  {t2} ERROR request aborted after {ms2}ms deadline"
        ),
        "list_recent_deploys": (
            "recent deploys for {service}:\n"
            "  {t0}  {service}  sha=a{n0}f  by=eng-{n1}  status=rolled-out\n"
            "  {t1}  {service}  sha=b{n2}c  by=eng-{n3}  status=rolled-out\n"
            "  {t2}  {service}  sha=d{n4}e  by=eng-{n5}  status=partial"
        ),
        "get_dependency_status": (
            "dependencies for {service}:\n"
            "  {dep}  reachable=true   p99_ms={p99_ms0}  errors={errors0}\n"
            "  {dep}  reachable=true   p99_ms={p99_ms1}  errors={errors1}\n"
            "  {dep}  reachable=false  p99_ms={p99_ms2}  errors={errors2}  last_ok={t0}"
        ),
        "get_error_budget": (
            "service={service} slo=99.9% window=30d budget_remaining_pct={budget_remaining_pct} "
            "burn_rate_1h={n1} burn_rate_6h={n2} projected_exhaustion={t0}"
        ),
        "check_feature_flags": (
            "flag changes for {service}:\n"
            "  {t0}  flag=new_pricing_engine  off->on   by=eng-{n0}\n"
            "  {t1}  flag=async_writes        on->off   by=eng-{n1}"
        ),
        "get_pod_events": (
            "pod events for {service} (last 15m):\n"
            "  {t0}  {service}  Restarted   reason=OOMKilled  count={count0}\n"
            "  {t1}  {service}  Unhealthy   probe=readiness   count={count1}\n"
            "  {t2}  {service}  Killing     reason=Evicted    count={count2}"
        ),
        "run_synthetic_probe": (
            "probe service={service} region={region} endpoint=/healthz status={status0} "
            "latency_ms={latency_ms} tls_ok=true at={t0}"
        ),
        # Stack-trace / error-output shape: a multi-line unhandled exception with
        # a couple of frames. No literal braces, so nothing to escape.
        "get_exception_trace": (
            "last unhandled exception for {service} at {t0} (seen {count0}x):\n"
            "Traceback (most recent call last):\n"
            '  File "/app/{service}/handler.py", line {n0}, in handle_request\n'
            "    resp = self.client.call(payload, timeout={ms0})\n"
            '  File "/app/{service}/client.py", line {n1}, in call\n'
            "    conn = self.pool.acquire(deadline={ms1})\n"
            '  File "/usr/lib/python3.11/{dep}/pool.py", line {n2}, in acquire\n'
            '    raise PoolTimeout("no connection acquired before deadline")\n'
            "PoolTimeout: no connection within {ms2}ms (in_use={in_use0}/{max0})"
        ),
        # JSON-object result shape: a small config blob. Literal JSON braces are
        # DOUBLED so str.format_map treats them as literals; only the real
        # placeholders ({service}, {n0}, {t0}, ...) stay single-braced.
        "get_config_snapshot": (
            '{{"service": "{service}", "version": "v{n0}", "replicas": {n1}, '
            '"flags": {{"new_pricing_engine": true, "async_writes": false}}, '
            '"limits": {{"pool_max": {max0}, "timeout_ms": {ms0}}}, '
            '"region": "{region}", "as_of": "{t0}"}}'
        ),
        "apply_remediation": (
            "remediation applied to {service}: rollout {pct0}% complete, {n0} pods updated, "
            "restarts={n1}, health=OK at {t0} (region {region})"
        ),
        "default": "result for {entity}: value={n0} at {t0}",
    },
    objective_template=(
        "{verb} the {symptom} on {service}: identify the root cause and recommend a remediation."
    ),
    followup_templates=[
        "What does the {symptom} on {service} look like over the last hour?",
        "Is {dep} implicated, or is this contained to {service}?",
        "should we roll back the most recent {service} deploy?",
        "Are other services in {region} showing the same {symptom}?",
    ],
    followup_connectives=["Following up, ", "Next, ", "One more thing — ", "OK, and "],
    intro_doc_templates=[
        (
            "----- PAGERDUTY INCIDENT #{n0} -----\n"
            "severity: SEV-2   opened: {t0}   status: TRIAGING\n"
            "service: {service}   region: {region}\n"
            "summary: {service} is reporting {symptom}. Customer-facing checkout success rate\n"
            "dropped from 99.9% to {drop_pct}% over ~{n2} minutes. On-call paged at {t1}.\n"
            "\n"
            "Recent context:\n"
            "  - deploy sha-a{n3}f rolled out to {rollout_pct}% of fleet at {t2}\n"
            "  - {dep} dependency latency began climbing at {t3}\n"
            "  - connection pool saturation alert fired at {t4} (in_use {in_use0}/{max0})\n"
            "\n"
            "Dashboard snapshot (p99 latency ms, 5m buckets):\n"
            "  {t5}  {p99_ms0}\n"
            "  {t6}  {p99_ms1}\n"
            "  {t7}  {p99_ms2}\n"
            "  {t8}  {p99_ms3}\n"
            "-------------------------------------\n"
        ),
        (
            "Slack thread export (#incident-{n0}):\n"
            "[{t0}] alertmanager: FIRING HighErrorRate service={service} value={err_pct}%\n"
            "[{t1}] oncall: ack, looking. {service} 5xx climbing, {rps0} rps of errors\n"
            "[{t2}] oncall: {dep} dependency looks slow, p99 {p99_ms0}ms\n"
            "[{t3}] sre-bot: error budget burn rate 1h={n4}x, budget remaining {budget_pct}%\n"
            "[{t4}] oncall: last deploy was {service} at {t5}, sha a{n6}f\n"
            "[{t6}] oncall: pool exhaustion on {service}, in_use {in_use0}/{max0}\n"
            "\n"
            "Attached metrics excerpt (requests/sec, error/sec):\n"
            "  {t7}  rps={rps1}  err={errors0}\n"
            "  {t8}  rps={rps2}  err={errors1}\n"
        ),
    ],
    filler_templates=[
        "{t0} INFO  {service} request id=req-{n0} completed status={status0} in {latency_ms}ms",
        "{t0} DEBUG {service} pool acquire waited {ms0}ms in_use={in_use0} idle={idle0} max={max0}",
        "{t0} WARN  {service} upstream {dep} slow: p99={p99_ms0}ms retries={retries0}",
        "{t0} INFO  gc pause={ms0}ms heap_used_mb={n0} heap_max_mb={n1}",
        "{t0} metric service={service} p50={p50_ms0} p99={p99_ms0} rps={rps0} err_rate={err_rate0}",
        "{t0} ERROR {service} deadline exceeded after {ms0}ms downstream={dep}",
        "{t0} INFO  deploy {service} sha=a{n0}f rollout={rollout_pct}% healthy={healthy0} unhealthy={unhealthy0}",
        "{t0} DEBUG trace tr-{n0} span={dep} dur={dur0}ms parent={service}",
    ],
    # Domain PAYLOAD shape for large tool-call body args (apply_remediation): ops
    # config/scripts (YAML-ish blocks, kubectl/shell runbooks) -- NOT prose or logs.
    # Kept free of LITERAL braces: the payload pool is built by splitting rendered
    # snippets into words, so a literal `{`/`}` (even a doubled/escaped one) would
    # survive into a body and trip the no-brace-leak invariant. Only real
    # placeholders ({service}/{dep}/{region}/{nN}/{msN}/{pct0}) appear.
    payload_templates=[
        (
            "kubectl -n prod set env deploy/{service} MAX_CONN={n0} POOL_TIMEOUT_MS={ms0} "
            "&& kubectl -n prod scale deploy/{service} --replicas={n1} "
            "&& kubectl -n prod rollout status deploy/{service} --timeout={ms1}ms"
        ),
        (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: {service}\n"
            "spec:\n"
            "  replicas: {n0}\n"
            "  template:\n"
            "    spec:\n"
            "      restartPolicy: OnFailure\n"
            "      containers:\n"
            "        - name: {service}\n"
            "          resources:\n"
            "            limits:\n"
            "              cpu: {n1}m\n"
            "              memory: {n2}Mi\n"
            "          env:\n"
            "            - name: UPSTREAM\n"
            "              value: {dep}"
        ),
        (
            "for pod in $(kubectl -n prod get pods -l app={service} -o name); do\n"
            "  kubectl -n prod rollout restart $pod\n"
            "  sleep {n0}\n"
            "done\n"
            "# drain connections to {dep} before cycling; target rollout {pct0}% in {region}"
        ),
        (
            "circuitBreaker:\n"
            "  service: {service}\n"
            "  dependency: {dep}\n"
            "  maxConnections: {n0}\n"
            "  timeoutMs: {ms0}\n"
            "  retries: {n1}\n"
            "  ejectAfterErrors: {n2}\n"
            "  region: {region}"
        ),
        (
            "helm upgrade {service} ./charts/{service} --namespace prod --atomic "
            "--set replicas={n0} --set pool.max={n1} --set pool.timeoutMs={ms0} "
            "--set upstream.host={dep} --set region={region} --timeout {ms1}ms"
        ),
    ],
    compaction_summary_template=(
        "{verb} {symptom} on {service} (region {region}, dependency {dep}). "
        "So far: ran {tool_a}, {tool_b}, and {tool_c}; gathered health metrics, "
        "recent deploys, and dependency status across the request path. "
        "Findings are still partial; continuing to narrow the root cause."
    ),
)
