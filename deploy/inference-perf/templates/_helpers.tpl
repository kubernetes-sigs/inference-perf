{{/*
Expand the name of the chart.
*/}}
{{- define "inference-perf.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "inference-perf.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "inference-perf.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "inference-perf.labels" -}}
helm.sh/chart: {{ include "inference-perf.chart" . }}
{{ include "inference-perf.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "inference-perf.selectorLabels" -}}
app.kubernetes.io/name: {{ include "inference-perf.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Common Secret Name for HuggingFace credentials
*/}}
{{- define "inference-perf.hfSecret" -}}
{{ include "inference-perf.fullname" . }}-hf-secret
{{- end -}}

{{/*
Common Secret Key for HuggingFace credentials
*/}}
{{- define "inference-perf.hfKey" -}}
{{ include "inference-perf.fullname" . }}-hf-key
{{- end -}}

{{/*
Mount path for config map
*/}}
{{- define "inference-perf.configMount" -}}
/cfg
{{- end -}}

{{/*
Whether inference-perf serves its runtime metrics endpoint. The tool serves it
whenever config.observability.metrics is set, even to {}, so the chart checks
that same key's presence and the two cannot disagree. Setting it to null
removes the key and turns the endpoint off.
*/}}
{{- define "inference-perf.metricsEnabled" -}}
{{- hasKey (.Values.config.observability | default dict) "metrics" -}}
{{- end -}}

{{/*
Port of the runtime metrics endpoint, from config.observability.metrics.port.
An empty block serves on 9464, the tool's DEFAULT_PORT. A test pins the
fallback below to it. Port 0 asks the tool for an ephemeral port, which no
container port or scrape config can name, so the chart refuses it.
*/}}
{{- define "inference-perf.metricsPort" -}}
{{- $raw := dig "observability" "metrics" "port" 9464 .Values.config -}}
{{- if not (regexMatch "^[0-9]+$" (toString $raw)) -}}
{{- fail (printf "config.observability.metrics.port must be a port number, got %q" (toString $raw)) -}}
{{- end -}}
{{- $port := int $raw -}}
{{- if eq $port 0 -}}
{{- fail "config.observability.metrics.port is 0 (ephemeral), which cannot be scraped in-cluster. Set a fixed port." -}}
{{- end -}}
{{- $port -}}
{{- end -}}

{{/*
The three scrape switches under .Values.metrics, each "true" or empty so
templates can test them with a bare if. Every map is defaulted so nulling
metrics or one of its monitors renders instead of failing on a nil pointer.
Annotations go off while a monitor is on: a cluster that also scrapes by
annotation would otherwise collect the pod twice and double-count.
*/}}
{{- define "inference-perf.podMonitorEnabled" -}}
{{- $metrics := .Values.metrics | default dict -}}
{{- if and (eq (include "inference-perf.metricsEnabled" .) "true") ($metrics.podMonitor | default dict).enabled }}true{{ end -}}
{{- end -}}

{{- define "inference-perf.podMonitoringEnabled" -}}
{{- $metrics := .Values.metrics | default dict -}}
{{- if and (eq (include "inference-perf.metricsEnabled" .) "true") ($metrics.podMonitoring | default dict).enabled }}true{{ end -}}
{{- end -}}

{{- define "inference-perf.annotationsEnabled" -}}
{{- $metrics := .Values.metrics | default dict -}}
{{- $monitored := or (include "inference-perf.podMonitorEnabled" .) (include "inference-perf.podMonitoringEnabled" .) -}}
{{- if and (eq (include "inference-perf.metricsEnabled" .) "true") $metrics.annotations (not $monitored) }}true{{ end -}}
{{- end -}}
