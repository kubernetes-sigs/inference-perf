# Storage Component

This directory contains component configuration for storage on the deployment. `base/` contains a template for loading model weights from a PersistentVolumeClaim. The user will need to specify PVC name (PVC_NAME) and model path in the base patch located in `base/deployment.patch.yaml`. The storage component is only targeted to work with vllm at the moment.

## Support
Current storage component supports the following storage systems:

### GKE
- GCSFuse
    - This configuration expects a pre-existing GCSFuse backed PVC (eg: model-pvc)
    - This configuration will require user to manually add/replace MODEL_PATH value with actual GCS model path.
