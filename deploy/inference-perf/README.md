## 🚀 Deploying `inference-perf` via Helm Chart

This guide explains how to deploy `inference-perf` to a Kubernetes cluster with Helm.

---

### 1. Prerequisites

Make sure you have the following tools installed and configured:

* **Kubernetes Cluster:** Access to a functional cluster (e.g., GKE).
* **Helm:** The Helm CLI installed locally.

---

### 2. Configuration (`values.yaml`)

Before deployment, navigate to the **`deploy/inference-perf`** directory and edit the **`values.yaml`** file to customize your deployment and the benchmark parameters.

#### Optional Token Parameters
Hugging Face token can be provided either by providing a value (`hfToken`) or by referencing an existing Kubernetes Secret (`hfSecret.Name` and `hfSecret.Key`).

> If both `hfToken` and the `hfSecret` parameters are provided, the chart logic is configured to prioritize the `hfSecret` reference.

| Key | Description | Default |
| :--- | :--- | :--- |
| `hfToken` | Hugging Face API token. If provided, a Kubernetes `Secret` named `hf-token-secret` will be created for authentication. | `""` |
| `hfSecret.name` | The name of a pre-existing Kubernetes Secret that contains a Hugging Face API token. | `""` |
| `hfSecret.key` | The key within the pre-existing Kubernetes Secret that holds the token value. | `""` |
---

#### Optional Job Parameters

| Key | Description | Default |
| :--- | :--- | :--- |
| `serviceAccountName` | Standard Kubernetes `serviceAccountName`. If not provided, default service account is used. | `""` |
| `nodeSelector` |  Standard Kubernetes `nodeSelector` map to constrain pod placement to nodes with matching labels. | `{}` |
| `resources` | Standard Kubernetes resource requests and limits for the main `inference-perf` container. | `{}` |
---

> **Example Resource Block:**
> ```yaml
> # resources:
> #   requests:
> #     cpu: "1"
> #     memory: "4Gi"
> #   limits:
> #     cpu: "2"
> #     memory: "8Gi"
> ```

#### GKE Specific Parameters

This section details the necessary configuration and permissions for using a Google Cloud Storage (GCS) path to manage your dataset, typical for deployments on GKE.

##### Required IAM Permissions

The identity executing the workload (e.g., the associated Kubernetes Service Account, often configured via **Workload Identity**) must possess the following IAM roles on the target GCS bucket for data transfer:

* **`roles/storage.objectViewer`** (Required to read/download the input dataset from GCS).
* **`roles/storage.objectCreator`** (Required to write/push benchmark results back to GCS).


| Key | Description | Default |
| :--- | :--- | :--- |
| `gcsPath` | A GCS bucket name pointing to the dataset file (e.g., `<my-bucket-path-to-file>/dataset.json`). The file will be automatically copied to the running pod during initialization. The file will be copied to `gcsDataset/dataset.json` | `""` |

---

#### AWS Specific Parameters

This section details the necessary configuration and permissions for using an S3 path to manage your dataset, typical for deployments on AWS EKS.

##### Required IAM Permissions

The identity executing the workload (e.g., the associated Kubernetes Service Account, often configured via IRSA - IAM Roles for Service Accounts) must possess an associated AWS IAM Policy that grants the following S3 Actions on the target S3 bucket for data transfer:

* **S3 Read/Download (Object Access)**
    * Action: `s3:GetObject` (Required to download the input dataset from S3).
    * Action: `s3:ListBucket` (Often required to check for the file's existence and list bucket contents).

* **S3 Write/Upload (Object Creation)**
    * Action: `s3:PutObject` (Required to upload benchmark results back to S3).


| Key | Description | Default |
| :--- | :--- | :--- |
| `s3Path` | An S3 bucket name pointing to the dataset file (e.g., `<my-bucket-path-to-file>/dataset.json`). The file will be automatically copied to the running pod during initialization. The file will be copied to `s3Dataset/dataset.json` | `""` |

---

#### Runtime Metrics Parameters

inference-perf serves its own runtime metrics (stage state, request counts, latencies) on `/metrics` while the run is active; see [`docs/runtime_metrics.md`](../../docs/runtime_metrics.md) for the metric set. Whether the endpoint runs, and on which port, is set in the benchmark config under `config.observability.metrics` (`enabled`, default `true`; `port`, default `9464`). The chart reads the same fields, so the container port and scrape configs always match what the process serves. Port `0` (ephemeral) cannot be scraped and fails the install.

The keys below only decide how Prometheus finds the pod.

| Key | Description | Default |
| :--- | :--- | :--- |
| `metrics.annotations` | Add `prometheus.io/scrape`, `prometheus.io/port` and `prometheus.io/path` pod annotations, for Prometheus setups that discover pods by annotation. | `true` |
| `metrics.interval` | Scrape interval for the `PodMonitor` and `PodMonitoring`. The pod stops serving when the run ends, so keep it short relative to the run. | `15s` |
| `metrics.podMonitor.enabled` | Create a `PodMonitor` for the Prometheus Operator. Requires the `monitoring.coreos.com` CRDs. | `false` |
| `metrics.podMonitor.labels` | Extra labels on the `PodMonitor`, e.g. the `release` label your Prometheus selects on. | `{}` |
| `metrics.podMonitoring.enabled` | Create a `PodMonitoring` for Google Cloud Managed Service for Prometheus on GKE. | `false` |
---

To check the endpoint by hand while a run is active:
```bash
kubectl port-forward job/<release>-inference-perf-job 9464:9464
curl -s localhost:9464/metrics | grep '^inference_perf_'
```

### 3. Run Deployment

Use the **`helm install`** command from the **`deploy/inference-perf`** directory to deploy the chart.

* **Standard Install:** Deploy using the default `values.yaml`.
    ```bash
    helm install test .
    ```

* **Set `hfToken` Override:** Pass the Hugging Face token directly.
    ```bash
    helm install test . --set hfToken="<TOKEN>"
    ```

* **Custom Config Override:** Make changes to the values file for custom settings.
    ```bash
    helm install test . -f values.yaml
    ```

### 4. Cleanup

To remove the benchmark deployment.
```bash
    helm uninstall test
```