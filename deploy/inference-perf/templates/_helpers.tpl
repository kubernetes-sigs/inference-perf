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
Whether inference-perf serves its runtime metrics endpoint. Read from
config.observability.metrics.enabled, the same field the tool reads, so the
chart and the process cannot disagree. The tool defaults it to true.
*/}}
{{- define "inference-perf.metricsEnabled" -}}
{{- dig "observability" "metrics" "enabled" true .Values.config -}}
{{- end -}}

{{/*
Port of the runtime metrics endpoint, from config.observability.metrics.port.
9464 is the tool's default (inference_perf.observability.metrics.prometheus.DEFAULT_PORT;
a test pins the two). Port 0 asks the tool for an ephemeral port, which no
container port or scrape config can name, so the chart refuses it.
*/}}
{{- define "inference-perf.metricsPort" -}}
{{- $port := dig "observability" "metrics" "port" 9464 .Values.config | int -}}
{{- if eq $port 0 -}}
{{- fail "config.observability.metrics.port is 0 (ephemeral), which cannot be scraped in-cluster. Set a fixed port or set config.observability.metrics.enabled to false." -}}
{{- end -}}
{{- $port -}}
{{- end -}}
