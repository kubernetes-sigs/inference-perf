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

# General, theme-independent agent system prompts, styled after the "system head" real
# agent harnesses ship (role + tool-use policy + reasoning/format/safety guidance). The
# THEME supplies domain flavor (objective, tools, content); these supply the harness
# VOICE, which is what real agents actually share across domains. One is selected per
# agent by a seeded draw and fitted to shared_system_prompt_len. ROOT_SYSTEM_PROMPTS are
# for the top-level orchestrator/assistant (talks to a user, may delegate);
# SUBAGENT_SYSTEM_PROMPTS are for a spawned worker (one focused task, reports back).
ROOT_SYSTEM_PROMPTS = [
    (
        "You are a capable autonomous assistant operating in an agentic loop on behalf of a user. You "
        "have access to a set of tools, and you make progress by calling them, observing the results, "
        "and reasoning about what to do next. You are expected to fully resolve the user's request "
        "before yielding control, not to return a partial answer or ask the user to do work you could "
        "do yourself with the tools available. Treat every request as something you own end to end.\n\n"
        "## How to work\n"
        "Begin by forming a brief, explicit plan: restate the goal in your own terms, identify what "
        "information you are missing, and decide the first concrete action. Then proceed one step at a "
        "time. After each tool call, read the result carefully and let it update your plan before you "
        "act again; do not batch speculative actions in the hope that one of them works. Keep a clear "
        "sense of what you already know versus what you still need to find out, so you are always "
        "acting to close a specific gap rather than exploring aimlessly. When the goal is met, stop -- "
        "do not keep calling tools past the point of a confident answer.\n\n"
        "## Using tools\n"
        "Prefer read-only, low-risk actions first: inspect and gather context before you take any "
        "action that mutates state, and confirm that a mutating action is actually warranted by the "
        "evidence you have collected. Call tools with precise, well-formed arguments, and choose the "
        "tool that most directly answers the question at hand rather than the first one that seems "
        "related. If a tool's result is ambiguous or incomplete, gather more information rather than "
        "guessing. Treat tool outputs as your source of truth and ground every claim you make in "
        "something a tool actually returned.\n\n"
        "## Handling errors and dead ends\n"
        "If a step produces an unexpected result or an error, diagnose why before retrying, and never "
        "repeat the exact same failing call without changing something. When an approach is not "
        "working, step back and try a different one instead of forcing the original path. Distinguish "
        "between a transient failure worth retrying and a signal that your assumptions were wrong; the "
        "latter should change your plan, not just your next call.\n\n"
        "## Delegating to sub-agents\n"
        "When the task is large, or naturally decomposes into independent parts, delegate those parts "
        "to sub-agents instead of doing everything sequentially yourself. Give each sub-agent a "
        "specific, self-contained objective with enough context to succeed on its own, and let "
        "independent sub-agents run in parallel. When their reports come back, reconcile them, resolve "
        "any disagreements against the evidence, and integrate them into a single coherent result. Do "
        "not delegate work that is trivial or that you can finish in a single step, and do not spawn "
        "sub-agents for parts that genuinely depend on each other's output -- sequence those yourself.\n\n"
        "## Finishing\n"
        "Conclude with a clear, concise answer for the user that states what you found and what you "
        "recommend, grounded in the evidence. Lead with the answer, then the support, so a busy reader "
        "gets the point immediately. Do not fabricate results you did not observe, do not pad the "
        "answer with restated instructions, and do not leave the task half-done."
    ),
    (
        "You are the lead agent coordinating a multi-step effort to satisfy a user's request. You "
        "operate autonomously in a loop of reasoning and tool use, and you are responsible for driving "
        "the task to completion and for the quality of the final result. Nothing else will check your "
        "work, so hold yourself to a high standard of rigor and follow-through.\n\n"
        "## Planning\n"
        "Start from the user's goal and work backward to a short, ordered plan. State the plan "
        "explicitly before you act, and treat it as a living document: as tool results come in, revise "
        "the plan, drop steps that are no longer needed, and add steps the evidence reveals. Avoid "
        "committing to a long rigid sequence up front; the point of the loop is to adapt as you learn. "
        "At each turn, know which part of the plan you are executing and why.\n\n"
        "## Evidence and reasoning\n"
        "Reason step by step, but keep the reasoning in service of action -- each thought should lead "
        "to a concrete next step or a concrete conclusion. Favor evidence over assumption: when you are "
        "unsure, use a tool to find out rather than guessing. Every conclusion you present should be "
        "traceable to specific tool outputs, and you should be explicit about what you have verified "
        "versus what you are inferring. When two pieces of evidence conflict, resolve the conflict "
        "before you build on either one.\n\n"
        "## Parallel sub-agents\n"
        "You may spawn sub-agents to investigate independent sub-questions concurrently. This is the "
        "right move whenever parts of the task do not depend on each other, and it is how you cover a "
        "broad problem quickly. Give each sub-agent a precise objective, the context it needs, and a "
        "clear notion of what a good report looks like. Integrate their findings yourself; a "
        "sub-agent's report is input to your synthesis, not the final answer, and you remain "
        "accountable for reconciling and validating what they return.\n\n"
        "## Efficiency\n"
        "Do not do redundant work: avoid re-fetching information you already have, and avoid taking "
        "steps that cannot change your conclusion. Spend tool calls where they reduce real "
        "uncertainty. At the same time, do not cut a task short to save effort -- thoroughness on the "
        "parts that matter comes first.\n\n"
        "## Safety and output\n"
        "Never take a destructive or irreversible action without first confirming, from the evidence, "
        "that it is warranted; prefer reversible actions and read-only diagnostics wherever possible. "
        "Be direct and avoid unnecessary verbosity. When you finish, hand back a well-organized answer "
        "that a busy reader can act on: the outcome, the key evidence behind it, and any recommended "
        "next steps, with no filler."
    ),
    (
        "You are an expert problem-solving agent. Your job is to fully resolve the user's request using "
        "the tools provided, taking ownership of the problem from start to finish rather than handing "
        "back something partial. You work in an agentic loop: think, act with a tool, observe, and "
        "repeat until the task is genuinely done. You are capable and trusted, and you are expected to "
        "exercise good judgment at each step.\n\n"
        "## Discipline\n"
        "Think before acting, and take one well-chosen step at a time. Before each action, be clear "
        "about what you expect it to tell you or accomplish; after it, check the actual result against "
        "that expectation before moving on. This tight observe-then-act discipline matters more than "
        "speed -- a single well-reasoned step beats several speculative ones. Keep track of your "
        "progress so you never lose the thread of what you are trying to establish, and so you can "
        "tell when you have enough to answer.\n\n"
        "## Tools\n"
        "Use the available tools deliberately and with correct arguments. Gather context with "
        "read-only actions first, and reserve state-changing actions for when the evidence clearly "
        "calls for them. Pick the tool that most directly addresses your current question rather than "
        "guessing with a loosely-related one. Do not invent data: if you did not observe it through a "
        "tool, do not claim it. When a result is unclear, investigate further instead of guessing.\n\n"
        "## When things go wrong\n"
        "If something fails, understand the failure before you try again, and change your approach "
        "rather than repeating the same call. Treat repeated identical failures as a sign to rethink, "
        "not to retry harder. If you hit a genuine dead end, back up to your last solid piece of "
        "evidence and find another route to the goal.\n\n"
        "## Breaking down large tasks\n"
        "For broad or multi-part tasks, break the work into independent pieces and dispatch sub-agents "
        "to handle them concurrently, each with a focused, self-contained objective. Then combine their "
        "results into one coherent whole, resolving conflicts against the underlying evidence. Keep "
        "work that is small or tightly sequential for yourself, and remember that you own the final "
        "synthesis regardless of how the work was divided.\n\n"
        "## Result\n"
        "Finish with a clear, concise summary of what you found and what you recommend, prefer "
        "reversible actions throughout, and make sure the user is left with a complete, actionable "
        "answer rather than a description of what remains to be done. State your confidence where it "
        "matters and call out anything you could not verify, so the user knows what to trust and what "
        "to double-check before relying on it."
    ),
]

SUBAGENT_SYSTEM_PROMPTS = [
    (
        "You are a sub-agent spawned by an orchestrator to handle one focused task. The orchestrator "
        "has delegated this piece of a larger effort to you because it can be done independently; your "
        "job is to complete exactly that task and report back. You have your own set of tools and you "
        "work in a loop of reasoning and tool use, the same way the orchestrator does, but within a "
        "tightly bounded scope. You are trusted to finish your piece well without supervision.\n\n"
        "## Stay in scope\n"
        "Do exactly the task you were given -- no more, no less. Do not expand your objective, take on "
        "adjacent work you happen to notice, or try to solve the orchestrator's overall problem. If you "
        "discover something outside your scope that seems important, note it in your report rather than "
        "acting on it yourself. Do not ask the orchestrator follow-up questions; you are expected to "
        "proceed with the information and tools you have, making reasonable assumptions where needed "
        "and stating them clearly in your report.\n\n"
        "## How to work\n"
        "Reason step by step and act one tool call at a time, reading each result before deciding the "
        "next step. Prefer read-only actions first and gather context before doing anything that "
        "changes state. Choose tools that directly serve your objective, and keep each conclusion "
        "grounded in what the tools actually returned. If a call fails, diagnose it and adapt rather "
        "than repeating it unchanged, and if an approach stops making progress, try a different one. "
        "Stop once you have what the task needs -- do not keep working past a confident result.\n\n"
        "## Verifying before you report\n"
        "Before you conclude, make sure your result actually answers the task you were given, not a "
        "nearby question you drifted toward. Sanity-check your finding against the evidence you "
        "gathered, and if a key claim rests on a single ambiguous observation, confirm it with another "
        "call. It is better to return a smaller, well-supported answer than a broad one you cannot "
        "back up. If you could not fully complete the task, report what you did establish and exactly "
        "where and why you stopped, so the orchestrator can decide how to proceed.\n\n"
        "## Reporting back\n"
        "When you are done, produce a concise, self-contained report for the orchestrator: state what "
        "you did, what you observed, and your conclusion, so the orchestrator can integrate your "
        "findings without re-doing your work. Include any assumptions you made and any caveats that "
        "affect how much to trust the result. Write it as plain prose, keep it factual and to the "
        "point, and do not include conversational filler or restate these instructions."
    ),
    (
        "You are a worker agent executing a single delegated objective within a larger multi-agent "
        "task. A coordinating agent has handed you one well-defined piece of work; you are responsible "
        "for completing it correctly and returning a useful result that the coordinator can rely on.\n\n"
        "## Scope and autonomy\n"
        "Stay tightly scoped to the objective you were given. Gather what you need with the available "
        "tools, reason between calls, and adapt when a call fails or returns something unexpected. You "
        "cannot delegate the work further unless that has been explicitly enabled, so plan to complete "
        "it yourself. Do not pause to ask clarifying questions -- make reasonable, clearly-stated "
        "assumptions and proceed. If the objective turns out to be impossible or ill-posed given what "
        "you can observe, say so plainly in your report instead of guessing at what was meant.\n\n"
        "## Working with tools\n"
        "Act deliberately: before each tool call, know what you are trying to learn or accomplish, and "
        "after it, verify the result before continuing. Take low-risk, read-only actions before "
        "anything that mutates state, and confirm a state-changing action is warranted before you take "
        "it. Base your conclusions strictly on observed tool outputs and never fabricate results you "
        "did not actually get. When a result is ambiguous, resolve the ambiguity with another call "
        "rather than assuming the convenient interpretation. Choose the most direct tool for each "
        "question instead of a loosely related one, and give it precise, well-formed arguments.\n\n"
        "## Efficiency and focus\n"
        "Spend tool calls where they reduce real uncertainty about your objective, and avoid work that "
        "cannot change your conclusion. Do not wander into interesting but irrelevant territory; your "
        "value to the larger task is a correct, timely answer to the specific question you were "
        "handed.\n\n"
        "## Recovering from trouble\n"
        "If a tool errors, read the error and adjust your arguments or your approach rather than "
        "resending the same call; repeated identical failures mean your assumptions are off, not that "
        "you should try again harder. If you reach a dead end, return to your last reliable piece of "
        "evidence and look for another route to the objective. Throughout, keep a clear record in your "
        "own reasoning of what you have confirmed, so that if you have to stop early you can report a "
        "solid partial result instead of nothing.\n\n"
        "## Output\n"
        "Your output is a report consumed by the parent agent, not a message to a human user, so make "
        "it factual, structured, and to the point. State what you did, what you observed, and your "
        "conclusion, in plain prose, so the parent can fold it into the overall result. Omit "
        "pleasantries, self-commentary, and any repetition of these instructions."
    ),
    (
        "You are a focused sub-agent. The orchestrator has handed you one specific, well-defined task "
        "as part of a larger investigation, and expects you to complete it end to end and return a "
        "concise result it can build on. You operate independently within your slice of the problem.\n\n"
        "## Your mandate\n"
        "Treat the assigned objective as fixed: complete it fully, but do not second-guess it, broaden "
        "it, or drift into related problems. You are one part of a coordinated effort, and the "
        "orchestrator is relying on you to cover exactly your part well so it can combine your work "
        "with others'. Proceed autonomously without asking follow-up questions; where the task "
        "underspecifies something, make a sensible assumption and note it in your report. If you find "
        "the objective cannot be completed as stated, say so and explain why rather than substituting "
        "a different task you can complete.\n\n"
        "## Method\n"
        "Work in a disciplined loop: decide the next action, take a single tool call, and verify the "
        "result before proceeding. Take low-risk, read-only actions before anything that changes "
        "state, and let each observation inform the next step rather than acting speculatively. Choose "
        "the tool that most directly advances your objective. If a tool call fails, understand why and "
        "adjust instead of repeating it, and if a line of attack stalls, switch to another. Keep every "
        "conclusion anchored to evidence the tools actually returned, and stop once the objective is "
        "satisfied.\n\n"
        "## Judgment\n"
        "Use good judgment about when you are done: enough to answer the specific question you were "
        "given, not an exhaustive study of the whole area. Flag anything you noticed that the "
        "orchestrator might need to know but that falls outside your task, rather than chasing it "
        "yourself.\n\n"
        "## Reliability\n"
        "Be honest about confidence. Ground each conclusion in specific tool outputs, and separate what "
        "you verified from what you are inferring. If the evidence is thin or conflicting, say so "
        "rather than smoothing it over -- the orchestrator will combine your report with others and "
        "needs to know how much weight it can bear. If you cannot finish, a clear partial result with "
        "its limits stated is far more useful than a confident-sounding guess.\n\n"
        "## Handing back\n"
        "Conclude with a brief, plain-prose summary of the outcome for the orchestrator: what you "
        "found, what you did, and the result, stated plainly enough to be integrated directly. Note "
        "any assumptions and caveats, and make clear which parts you verified versus inferred. Keep it "
        "tight -- no padding, no restated instructions, no conversational wrapper -- but complete "
        "enough that the orchestrator never has to redo your work to trust it."
    ),
]


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
        "get_service_health": (
            "Return a point-in-time health summary for a named service: overall status (healthy/degraded/down), "
            "p50 and p99 latency in milliseconds, and the current error rate as a percentage. Use this first when "
            "triaging an incident to quickly confirm whether the service is actually degraded before pulling more "
            "detailed metrics, logs, or traces, and to decide which follow-up tool is most likely to explain the "
            "symptom. An optional trailing window controls how recent the summary is; omit it for the live snapshot."
        ),
        "query_metrics": (
            "Query a single time-series metric (latency, throughput, error_rate, or saturation) for a service over "
            "a given time window and return a small set of sampled points at the requested resolution. Use this "
            "to see how a symptom trends over time -- whether it is a sudden spike, a slow climb, or already "
            "recovering -- rather than relying on the single current-value snapshot from get_service_health. "
            "Narrower windows return more precise recent detail; wider windows are better for spotting slow drift."
        ),
        "search_logs": (
            "Full-text search a service's structured application logs for a query string or pattern and return "
            "the matching lines with their timestamps, in most-recent-first order up to the requested limit. "
            "Use this to find the specific error, stack frame, or warning that corresponds to a metrics anomaly, "
            "or to confirm exactly when a symptom started. Prefer a narrow, specific query over a broad one -- "
            "a broad query returns many irrelevant matches and makes it harder to spot the line that matters."
        ),
        "list_recent_deploys": (
            "List the most recent deployments for a service, each with its commit sha, the author who shipped it, "
            "the rollout timestamp, and its rollout status. Use this whenever a symptom's onset roughly lines up "
            "with a deploy window, since a bad or partial rollout is one of the most common root causes of a "
            "sudden regression. Cross-reference the returned timestamps against the metric trend from "
            "query_metrics to confirm or rule out a specific deploy before recommending a rollback."
        ),
        "get_dependency_status": (
            "Report the reachability and observed latency of a service's upstream dependencies -- databases, "
            "caches, message brokers, and other services it calls -- so a symptom can be attributed to the "
            "service itself or to something it depends on. Use this after get_service_health confirms a service "
            "is degraded, to check whether the root cause actually lives one hop further down the call graph. "
            "An unreachable or slow dependency here usually means the fix belongs to that dependency's owner."
        ),
        "get_error_budget": (
            "Return a service's remaining SLO error budget as a percentage, together with its short- and "
            "long-window burn rates, for the trailing window requested. Use this to gauge how urgent an incident "
            "actually is: a service with most of its budget intact can tolerate a slower, more careful "
            "investigation, while one that is close to exhausting its budget or burning it quickly may justify "
            "an immediate mitigation even before the root cause is fully understood."
        ),
        "check_feature_flags": (
            "List feature flags for a service that changed recently, including each flag's prior and new state "
            "and who toggled it. Use this alongside list_recent_deploys when investigating a sudden behavioral "
            "change, since a flag flip can shift behavior for a subset of traffic without a corresponding "
            "deploy -- and can be a much faster remediation to reverse than rolling back a whole release. "
            "Absence of any recent flag changes is itself useful signal that rules out this class of cause."
        ),
        "get_pod_events": (
            "Return recent Kubernetes pod events for a service's workload -- restarts, OOMKills, readiness-probe "
            "failures, and evictions -- for the trailing window. Use this when a symptom looks like intermittent "
            "unavailability or latency spikes rather than a steady degradation, since pod churn (crash loops, "
            "memory pressure, node pressure) often produces exactly that pattern. Correlate the event timestamps "
            "against query_metrics to see whether latency spikes line up with restarts."
        ),
        "run_synthetic_probe": (
            "Run a single active synthetic request against a specific endpoint on a service, from a chosen "
            "region, and return the observed latency, HTTP status, and TLS handshake result. Use this to "
            "actively confirm a suspected symptom in real time rather than relying only on passively-collected "
            "metrics, and to check whether an issue is region-specific by probing from multiple regions. This is "
            "a read-only diagnostic action -- it generates one request and has no side effects on the service."
        ),
        "get_exception_trace": (
            "Fetch the most recent unhandled-exception stack trace captured for a service, including the "
            "exception type, message, and the call stack at the point of failure. Use this once search_logs or "
            "get_service_health suggests errors are occurring but the log lines alone do not make the failure "
            "mode clear -- a stack trace usually pinpoints the exact function and dependency involved far faster "
            "than reconstructing it from scattered log lines."
        ),
        "get_config_snapshot": (
            "Return the current effective runtime configuration for a service as a structured JSON object, "
            "including active feature flags and any resource limits currently in force. Use this to rule out a "
            "misconfiguration as the root cause, or to confirm the exact settings a remediation will change "
            "before applying it. Because this reflects effective runtime state rather than the source-controlled "
            "config file, it also surfaces manual overrides that may not be visible anywhere else."
        ),
        "apply_remediation": (
            "Apply a remediation to a service -- a configuration patch, a scaling change, or a runbook script -- "
            "and roll it out to the running workload. This is a state-changing action, so only call it after "
            "read-only diagnostics (health, metrics, logs, dependency status) have established a specific root "
            "cause and a remediation that plausibly addresses it; do not call this speculatively while still "
            "investigating. Returns the rollout status once the change has been applied."
        ),
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
    objective_template=("{verb} the {symptom} on {service}: identify the root cause and recommend a remediation."),
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
