import yaml
from fmperf.Cluster import DeployedModel


class WorkloadSpec:
    def __init__(
        self,
        sample_size: int = 10,
        image: str = "quay.io/fmperf/fmperf:main",
        pvc_name: str = None,
        overwrite: bool = False,
    ):
        self.sample_size = sample_size
        self.image = image
        self.pvc_name = pvc_name
        self.overwrite = overwrite

    @classmethod
    def from_yaml(cls, file: str):
        """
        The function processes the yaml file to create workload specs
        """
        with open(file, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_env(
        self,
        target: str,
        model: "DeployedModel",
        outfile: str,
    ):
        env = [
            {"name": "TARGET", "value": target},
            {"name": "MODEL_ID", "value": model.spec.name},
            {
                "name": "SAMPLE_SIZE",
                "value": str(self.sample_size),
            },
            {
                "name": "URL",
                "value": model.url,
            },
            {
                "name": "OVERWRITE",
                "value": str(self.overwrite),
            },
            {"name": "REQUESTS_FILENAME", "value": outfile},
        ]
        return env


class HomogeneousWorkloadSpec(WorkloadSpec):
    def __init__(
        self,
        input_tokens: int = 500,
        output_tokens: int = 50,
        greedy: bool = True,
        image: str = "fmperf-project/fmperf:local",
        pvc_name: str = None,
        overwrite: bool = False,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.greedy = greedy

        super().__init__(1, image, pvc_name, overwrite)

    @classmethod
    def from_yaml(cls, file: str):
        return super().from_yaml(file)

    def get_args(self):
        return ["python -m fmperf.loadgen.generate-input"]

    def get_env(
        self,
        target: str,
        model: "DeployedModel",
        outfile: str,
    ):
        env = super().get_env(target, model, outfile) + [
            {
                "name": "MIN_INPUT_TOKENS",
                "value": str(self.input_tokens),
            },
            {
                "name": "MAX_INPUT_TOKENS",
                "value": str(self.input_tokens),
            },
            {
                "name": "MIN_OUTPUT_TOKENS",
                "value": str(self.output_tokens),
            },
            {
                "name": "MAX_OUTPUT_TOKENS",
                "value": str(self.output_tokens),
            },
            {
                "name": "FRAC_GREEDY",
                "value": "1.0" if self.greedy else "0.0",
            },
        ]
        return env


class HeterogeneousWorkloadSpec(WorkloadSpec):
    def __init__(
        self,
        min_input_tokens: int = 10,
        max_input_tokens: int = 20,
        min_output_tokens: int = 10,
        max_output_tokens: int = 20,
        frac_greedy: float = 0.5,
        sample_size: int = 10,
        image: str = "quay.io/fmperf/fmperf:main",
        pvc_name: str = None,
        overwrite: bool = False,
    ):
        self.min_input_tokens = min_input_tokens
        self.max_input_tokens = max_input_tokens
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens
        self.frac_greedy = frac_greedy
        super().__init__(sample_size, image, pvc_name, overwrite)

    @classmethod
    def from_yaml(cls, file: str):
        return super().from_yaml(file)

    def get_args(self):
        return ["python -m fmperf.loadgen.generate-input"]

    def get_env(
        self,
        target: str,
        model: "DeployedModel",
        outfile: str,
    ):
        env = super().get_env(target, model, outfile) + [
            {
                "name": "MIN_INPUT_TOKENS",
                "value": str(self.min_input_tokens),
            },
            {
                "name": "MAX_INPUT_TOKENS",
                "value": str(self.max_input_tokens),
            },
            {
                "name": "MIN_OUTPUT_TOKENS",
                "value": str(self.min_output_tokens),
            },
            {
                "name": "MAX_OUTPUT_TOKENS",
                "value": str(self.max_output_tokens),
            },
            {
                "name": "FRAC_GREEDY",
                "value": str(self.frac_greedy),
            },
        ]
        return env


class RealisticWorkloadSpec(WorkloadSpec):
    def __init__(
        self,
        sample_size: int = 10,
        image: str = "quay.io/fmperf/fmperf:main",
        pvc_name: str = None,
        overwrite: bool = False,
    ):
        super().__init__(sample_size, image, pvc_name, overwrite)

    @classmethod
    def from_yaml(cls, file: str):
        return super().from_yaml(file)

    def get_args(self):
        return ["python -m fmperf.loadgen.generate-input --from-model"]
