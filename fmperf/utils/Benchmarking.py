from fmperf import Cluster
from fmperf.ModelSpecs import ModelSpec
from fmperf.utils import parse_results


# Run benchmark for a list of models and a workload spec
def run_benchmark(
    cluster: Cluster,
    model_spec: list[ModelSpec],
    workload_spec: dict,
    repetition: int,
    number_users: int,
    duration: str,
    id: str = "",
) -> None:
    if isinstance(model_spec, ModelSpec):
        model_spec = [model_spec]

    for spec in model_spec:
        # create the inference server
        model = cluster.deploy_model(spec, id)
        # create the jobs for requests
        workload = cluster.generate_workload(model, workload_spec, id=id)
        for rep in range(repetition):
            print(f"Performing sweep with {workload.file}")
            results = []
            for num_users in number_users:
                output, _ = cluster.evaluate(
                    model,
                    workload,
                    num_users=num_users,
                    duration=duration,
                    id=id,
                )
                if output is not None:
                    results.extend(output)
            df = parse_results(results, print_df=True)
            df.to_csv(f"result{rep}.csv")
    cluster.delete_model(model)
