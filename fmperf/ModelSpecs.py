import yaml
from typing import Optional
import os
import subprocess
import json


class ModelSpec:
    """
    Parent class for model specifications
    """

    def __init__(
        self,
        pvcs: list = None,
        cluster_gpu_name: str = None,
    ):
        self.set_volumes(pvcs)
        self.set_affinity(cluster_gpu_name)

    def force_il(self):
        self.affinity = {}
        self.affinity["nodeAffinity"] = {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "node.kubernetes.io/instance-type",
                                "operator": "In",
                                "values": ["gx2-80x1280x8a100-il-rdma"],
                            }
                        ]
                    },
                ]
            }
        }

    def set_affinity(self, cluster_gpu_name):
        self.affinity = {}
        if cluster_gpu_name is not None:
            self.affinity["nodeAffinity"] = {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "nvidia.com/gpu.product",
                                    "operator": "In",
                                    "values": [cluster_gpu_name],
                                }
                            ]
                        }
                    ]
                }
            }

    def set_volumes(self, pvcs):
        self.volumes = []
        self.volume_mounts = []

        if pvcs is not None:
            for pvc in pvcs:
                name = "volume-%d" % (len(self.volumes))
                claim_name, mount_path = pvc
                self.volumes.append(
                    {
                        "name": name,
                        "persistentVolumeClaim": {
                            "claimName": claim_name,
                        },
                    }
                )
                self.volume_mounts.append({"name": name, "mountPath": mount_path})

    @classmethod
    def from_yaml(cls, file: str):
        out = []
        with open(file, "r") as f:
            out = []
            for x in yaml.safe_load_all(f):
                if x is not None:
                    out.append(cls(**x))
        if len(out) == 0:
            raise ValueError("Could not find any valid model specs")
        elif len(out) == 1:
            return out[0]
        else:
            return out


class TGISModelSpec(ModelSpec):
    def __init__(
        self,
        name: str,
        shortname: str = None,
        max_new_tokens: int = 1024,
        max_sequence_length: int = 2048,
        max_batch_size: int = 12,
        max_concurrent_requests: int = 96,
        max_batch_weight: int = None,
        max_waiting_tokens: int = 12,
        max_prefill_weight: int = None,
        dtype_str: str = "float16",
        deployment_framework: str = "hf_accelerate",
        cpu_limit: int = 16,
        memory_limit: str = "128Gi",
        cpu_request: int = 8,
        image: str = None,
        compile: bool = False,
        trust_remote_code: bool = False,
        flash_attention: bool = False,
        num_gpus: int = 1,  # 4 for selecting different GPUS
        cuda_visible_devices: str = "0",
        download_weights: bool = False,
        pvcs: list = None,
        hf_hub_cache: str = "/models",
        batch_safety_margin: int = 20,
        revision: str = None,
        quantize: str = None,
        output_special_tokens: bool = False,
        cluster_gpu_name: str = None,
        expandable_segments: bool = None,
        paged_attention: bool = False,
        speculator_name: str = None,
        speculator_n_candidates: int = None,
        speculator_max_batch_size: int = None,
        transformers_cache: str = None,  # Legacy way of specifying hf_hub_cache. Supports older deployments of TGIS
        port: int = 3000,
        max_log_len: int = None,
    ):
        self.name = name
        if shortname is None:
            self.shortname = name.split("/")[-1] if "/" in name else name
            self.shortname = self.shortname.replace("_", "-")
        else:
            self.shortname = shortname

        if transformers_cache is not None and hf_hub_cache == "/models":
            # Legacy deployment support
            # If only transformers_cache is specified and _not_ hf_hub_cache, then use transformers_cache
            self.hf_hub_cache = transformers_cache
        else:
            self.hf_hub_cache = hf_hub_cache

        self.max_new_tokens = max_new_tokens
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.max_batch_weight = max_batch_weight
        self.max_waiting_tokens = max_waiting_tokens
        self.max_prefill_weight = max_prefill_weight
        self.dtype_str = dtype_str
        self.deployment_framework = deployment_framework
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.cpu_request = cpu_request
        self.image = image
        self.compile = compile
        self.trust_remote_code = trust_remote_code
        self.flash_attention = flash_attention
        self.num_gpus = num_gpus
        self.cuda_visible_devices = cuda_visible_devices
        self.download_weights = download_weights
        self.hf_hub_cache = hf_hub_cache
        self.batch_safety_margin = batch_safety_margin
        self.revision = revision
        self.quantize = quantize
        self.output_special_tokens = output_special_tokens
        self.expandable_segments = expandable_segments
        self.paged_attention = paged_attention
        self.speculator_name = speculator_name
        self.speculator_n_candidates = speculator_n_candidates
        self.speculator_max_batch_size = speculator_max_batch_size
        self.port = port
        self.max_log_len = max_log_len
        super().__init__(pvcs, cluster_gpu_name)

    def get_vars(self):
        vars = [
            {
                "name": "MODEL_NAME",
                "value": self.name,
            },
            {
                "name": "NUM_GPUS",
                "value": str(self.num_gpus),
            },
            {
                "name": "CUDA_VISIBLE_DEVICES",
                "value": self.cuda_visible_devices,
            },
            {
                "name": "HF_HUB_CACHE",
                "value": self.hf_hub_cache,
            },
            {
                # Legacy support, TRANSFORMERS_CACHE is deprecated and unsupported in newer versions of TGIS
                "name": "TRANSFORMERS_CACHE",
                "value": self.hf_hub_cache,
            },
            {
                "name": "MAX_SEQUENCE_LENGTH",
                "value": str(self.max_sequence_length),
            },
            {
                "name": "MAX_NEW_TOKENS",
                "value": str(self.max_new_tokens),
            },
            {
                "name": "MAX_BATCH_SIZE",
                "value": str(self.max_batch_size),
            },
            {
                "name": "MAX_CONCURRENT_REQUESTS",
                "value": str(self.max_concurrent_requests),
            },
            {"name": "DTYPE_STR", "value": self.dtype_str},
            {
                "name": "DEPLOYMENT_FRAMEWORK",
                "value": self.deployment_framework,
            },
            {
                "name": "PT2_COMPILE",
                "value": str(self.compile),
            },
            {
                "name": "TRUST_REMOTE_CODE",
                "value": str(self.trust_remote_code).lower(),
            },
            {
                "name": "FLASH_ATTENTION",
                "value": str(self.flash_attention).lower(),
            },
            {
                "name": "MAX_WAITING_TOKENS",
                "value": str(self.max_waiting_tokens),
            },
            {
                "name": "BATCH_SAFETY_MARGIN",
                "value": str(self.batch_safety_margin),
            },
            {
                "name": "OUTPUT_SPECIAL_TOKENS",
                "value": str(self.output_special_tokens).lower(),
            },
            {"name": "PAGED_ATTENTION", "value": str(self.paged_attention).lower()},
            {
                "name": "PORT",
                "value": str(self.port),
            },
        ]

        if self.expandable_segments is not None:
            vars.append(
                {
                    "name": "PYTORCH_CUDA_ALLOC_CONF",
                    "value": "expandable_segments:%s" % ("True" if self.expandable_segments else "False"),
                }
            )

        if self.speculator_name is not None:
            vars.append(
                {
                    "name": "SPECULATOR_NAME",
                    "value": str(self.speculator_name),
                }
            )

        if self.speculator_n_candidates is not None:
            vars.append(
                {
                    "name": "SPECULATOR_N_CANDIDATES",
                    "value": str(self.speculator_n_candidates),
                }
            )

        if self.speculator_max_batch_size is not None:
            vars.append(
                {
                    "name": "SPECULATOR_MAX_BATCH_SIZE",
                    "value": str(self.speculator_max_batch_size),
                }
            )

        if self.max_batch_weight is not None:
            vars.append(
                {
                    "name": "MAX_BATCH_WEIGHT",
                    "value": str(self.max_batch_weight),
                }
            )

        if self.max_prefill_weight is not None:
            vars.append(
                {
                    "name": "MAX_PREFILL_WEIGHT",
                    "value": str(self.max_prefill_weight),
                }
            )

        if self.revision is not None:
            vars.append(
                {
                    "name": "REVISION",
                    "value": self.revision,
                }
            )

        if self.quantize is not None:
            vars.append(
                {
                    "name": "QUANTIZE",
                    "value": self.quantize,
                }
            )

        if self.max_log_len is not None:
            vars.append(
                {
                    "name": "MAX_LOG_LEN",
                    "value": self.max_log_len,
                }
            )

        return vars

    def get_command(self):
        # Specifies the Entry Point
        return ["/bin/bash", "-c"]

    def get_url(self):
        # Specifies the url of server
        return "%s:8033"

    def get_args(self):
        if "vllm" in self.image:
            cmd = "python3 -m vllm.entrypoints.openai.api_server"
        else:
            cmd = "HF_HUB_OFFLINE=1 HUGGINGFACE_HUB_CACHE=$TRANSFORMERS_CACHE text-generation-launcher --num-shard $NUM_GPUS"
        if not self.download_weights:
            return [cmd]
        else:
            return ["text-generation-server download-weights %s; %s" % (self.name, cmd)]

    def get_liveness_probe(self):
        # Runs a healthcheck before the service runs
        return {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 100,
            "timeoutSeconds": 30,
        }

    def get_ports(self):
        # Specifies the ports for service
        return [
            {"containerPort": 3000, "name": "http"},
            {"containerPort": 8033, "name": "grpc"},
        ]

    def get_readiness_probe(self):
        # Runs a healthcheck on a running service
        return {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 30,
            "timeoutSeconds": 10,
        }

    def get_startup_probe(self):
        # Runs a healthcheck on a running service
        return {
            "failureThreshold": 10000,
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 30,
        }

    @classmethod
    def from_yaml(cls, file: str):
        return super().from_yaml(file)

    @classmethod
    def from_deployment(cls, tgis_path: str, model: str):
        print("parsing deployment for %s" % (model))
        path = os.path.join(tgis_path, "deployment/models/", model)
        result = subprocess.run(
            ["kustomize", "build", "--load-restrictor", "LoadRestrictionsNone", path],
            capture_output=True,
            text=True,
        )
        data = []
        for x in yaml.safe_load_all(result.stdout):
            data.append(x)

        data = data[1]
        kwargs = {}
        for x in data["spec"]["template"]["spec"]["containers"][0]["env"]:
            k, v = x["name"], x["value"]
            k = k.lower()
            if k == "model_name":
                k = "name"
            if k == "pt2_compile":
                k = "compile"
            if k == "prefix_store_path":
                continue
            kwargs[k] = v

        print(json.dumps(kwargs, indent=4))

        out = cls(**kwargs)

        out.volumes = data["spec"]["template"]["spec"]["volumes"]
        out.volume_mounts = data["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        out.affinity = data["spec"]["template"]["spec"]["affinity"]

        return out


# vLLM Model class
class vLLMModelSpec(ModelSpec):
    def __init__(
        self,
        name: str,
        shortname: str = None,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        download_dir: Optional[str] = None,
        load_format: str = "auto",
        dtype: str = "auto",
        seed: int = 0,
        max_model_len: Optional[int] = None,
        worker_use_ray: bool = False,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        swap_space: int = 4,  # GiB
        gpu_memory_utilization: float = 0.90,
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_log_stats: bool = False,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        quantization: Optional[str] = None,
        image: str = None,
        cpu_limit: int = 16,
        memory_limit: str = "128Gi",
        cpu_request: int = 8,
        pvcs: list = None,
        transformers_cache: str = None,
        hf_hub_cache: str = "/models",
        cluster_gpu_name: str = None,
    ):
        self.name = name
        if shortname is None:
            self.shortname = name.split("/")[-1] if "/" in name else name
            self.shortname = self.shortname.replace("_", "-")
        else:
            self.shortname = shortname

        if transformers_cache is not None and hf_hub_cache == "/models":
            # If only transformers_cache is specified and _not_ hf_hub_cache, then use transformers_cache
            self.hf_hub_cache = transformers_cache
        else:
            self.hf_hub_cache = hf_hub_cache

        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.dtype = dtype
        self.seed = seed
        self.max_model_len = max_model_len
        self.worker_use_ray = worker_use_ray
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.block_size = block_size
        self.swap_space = swap_space
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.disable_log_stats = disable_log_stats
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.cpu_request = cpu_request
        self.num_gpus = tensor_parallel_size
        super().__init__(pvcs, cluster_gpu_name)

    def get_vars(self):
        vars = [
            {
                "name": "HF_HUB_CACHE",
                "value": self.hf_hub_cache,
            },
            {
                "name": "TRANSFORMERS_CACHE",
                "value": "$(HF_HUB_CACHE)",
            },
            {
                "name": "HF_HUB_OFFLINE",
                "value": "0",
            },
            {
                "name": "NUMBA_CACHE_DIR",
                "value": "/tmp",
            },
        ]
        return vars

    def get_command(self):
        # Specifies the Entry Point
        return ["python3"]

    def get_url(self):
        # Specifies the url of server
        return "%s:8000"

    def get_args(self):
        # Specifies the arguments for docker container run-time
        args = []
        args += ["-m", "vllm.entrypoints.openai.api_server"]
        args += ["--model", self.name]
        args += ["--max-num-seqs", str(self.max_num_seqs)]
        args += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        args += ["--dtype", self.dtype]
        args += ["--enforce-eager"]

        if self.max_num_batched_tokens is not None:
            args += ["--max-num-batched-tokens", str(self.max_num_batched_tokens)]

        if self.max_model_len is not None:
            args += ["--max-model-len", str(self.max_model_len)]

        if self.quantization is not None:
            args += ["--quantization", str(self.quantization)]

        return args

    def get_liveness_probe(self):
        # Runs a healthcheck by sending requests periodically to container
        return {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 100,
            "timeoutSeconds": 30,
        }

    def get_ports(self):
        # Specifies the ports for service
        return [{"containerPort": 8000, "name": "http"}]

    def get_readiness_probe(self):
        # Runs a healthcheck on a running service
        return {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 30,
            "timeoutSeconds": 10,
        }

    def get_startup_probe(self):
        # Runs a healthcheck on a running service
        return {
            "failureThreshold": 10000,
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 30,
        }

    @classmethod
    def from_yaml(cls, file: str):
        return super().from_yaml(file)

    @classmethod
    def from_deployment(cls, tgis_path: str, model: str):
        print("parsing deployment for %s" % (model))
        path = os.path.join(tgis_path, "deployment/models/", model)
        result = subprocess.run(
            ["kustomize", "build", "--load-restrictor", "LoadRestrictionsNone", path],
            capture_output=True,
            text=True,
        )
        data = []
        for x in yaml.safe_load_all(result.stdout):
            data.append(x)

        data = data[1]
        kwargs = {}
        for x in data["spec"]["template"]["spec"]["containers"][0]["env"]:
            k, v = x["name"], x["value"]
            k = k.lower()

            if k == "model_name":
                k = "name"

            if k == "dtype_str":
                k = "dtype"

            if k == "quantize":
                k = "quantization"

            if k in ["name", "transformers_cache", "dtype", "quantization"]:
                kwargs[k] = v

        print(json.dumps(kwargs, indent=4))

        out = cls(**kwargs)

        out.volumes = data["spec"]["template"]["spec"]["volumes"]
        out.volume_mounts = data["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        out.affinity = data["spec"]["template"]["spec"]["affinity"]

        return out
