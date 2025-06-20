## Run `inference-perf` via Helm Chart

This guide explains how to deploy `inference-perf` to a Kubernetes cluster with Helm.

### Setup

Within the deploy/infernce-perf directory, edit the config.yaml and values.yaml to configure the Helm Chart and infernce-perf options.

### Run

Deploy locally via `helm install test .` optionally including the hfToken via `helm install test . --set hfToken=<token>`.