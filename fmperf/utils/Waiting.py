from kubernetes import client, watch
import time
import urllib3


class CloudError(Exception):
    pass


class Waiting:
    def __init__(self, apigetter, logger):
        self.apigetter = apigetter
        self.logger = logger

    def _wait_for(self, name, func, *args, until="ready", timeout_seconds=6400, resource_version=None):
        if until is None:
            return

        t0 = time.time()

        try:
            self.logger.info("waiting for %s until %s (%d seconds remaining)" % (name, until, timeout_seconds))
            w = watch.Watch()
            for event in w.stream(
                func(self.apigetter()),
                *args,
                timeout_seconds=timeout_seconds,
                _request_timeout=20,
                resource_version=resource_version,
            ):
                object = event["object"]
                if not isinstance(object, dict):
                    object = object.to_dict()

                if object["metadata"]["name"] != name:
                    continue

                if until == "delete":
                    if event["type"] == "DELETED":
                        w.stop()
                        return
                elif until == "ready":
                    if "status" in object and "conditions" in object["status"]:
                        # print(object["status"]["conditions"])
                        if object["status"]["conditions"] is None:
                            continue
                        for x in object["status"]["conditions"]:
                            if x["type"] == "Ready" and x["status"] == "True":
                                w.stop()
                                return
                            # handle observed ng provisioing errors on AWS
                            if x["type"] == "Synced" and x["status"] == "False" and x["reason"] == "ReconcileError":
                                raise CloudError(x["message"])
                            # handle observed quota errors on Azure
                            if (
                                x["type"] == "LastAsyncOperation"
                                and x["status"] == "False"
                                and x["reason"] == "ApplyFailure"
                                and "QuotaExceeded" in x["message"]
                            ):
                                raise CloudError(x["message"])
                            # handle observed quota errors on GCP
                            if (
                                x["type"] == "LastAsyncOperation"
                                and x["status"] == "False"
                                and x["reason"] == "ApplyFailure"
                                and "error creating NodePool" in x["message"]
                            ):
                                raise CloudError(x["message"])
                            # handle issue #25
                            if (
                                x["type"] == "PodScheduled"
                                and x["status"] == "False"
                                and x["reason"] == "Unschedulable"
                                and "untolerated taint" not in x["message"]
                            ):
                                raise CloudError(x["message"])

                elif until == "available":
                    if "status" in object and "conditions" in object["status"]:
                        # print(object["status"]["conditions"])
                        if object["status"]["conditions"] is None:
                            continue
                        for x in object["status"]["conditions"]:
                            if x["type"] == "Available" and x["status"] == "True":
                                w.stop()
                                return
                elif until == "complete":
                    if "status" in object and "conditions" in object["status"]:
                        # print(object["status"]["conditions"])
                        if object["status"]["conditions"] is None:
                            continue
                        for x in object["status"]["conditions"]:
                            if x["type"] == "Complete" and x["status"] == "True":
                                w.stop()
                                return
        except CloudError as e:
            retry = False
            err = e
        except Exception as e:
            retry = True
            if isinstance(e, urllib3.exceptions.ReadTimeoutError):
                # pynisher cannot handle this kind of exception; need to remap
                err = TimeoutError("Client-side watch timeout")
            else:
                err = e
        else:
            # we can only reach here in case of server-side timeout
            retry = False
            err = TimeoutError("Server-side watch timeout")

        t_remaining = int(timeout_seconds - time.time() + t0)

        if retry and t_remaining >= 1:
            self._wait_for(name, func, *args, until=until, timeout_seconds=t_remaining)
        else:
            raise err

    def wait_for_node(
        self,
        name: str,
        until="ready",
        timeout_seconds=1800,
    ):
        self._wait_for(
            name,
            lambda x: client.CoreV1Api(x).list_node,
            until=until,
            timeout_seconds=timeout_seconds,
        )

    def wait_for_cluster_custom_object(
        self,
        name: str,
        group: str,
        version: str,
        plural: str,
        until="ready",
        timeout_seconds=1800,
    ):
        self._wait_for(
            name,
            lambda x: client.CustomObjectsApi(x).list_cluster_custom_object,
            group,
            version,
            plural,
            until=until,
            timeout_seconds=timeout_seconds,
        )

    def wait_for_namespaced_custom_object(
        self,
        name: str,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        until="ready",
    ):
        self._wait_for(
            name,
            lambda x: client.CustomObjectsApi(x).list_namespaced_custom_object,
            group,
            version,
            namespace,
            plural,
            until=until,
        )

    def wait_for_namespaced_deployment(self, name: str, namespace: str, until="ready", resource_version=None):
        self._wait_for(
            name,
            lambda x: client.AppsV1Api(x).list_namespaced_deployment,
            namespace,
            until=until,
            resource_version=resource_version,
        )

    def wait_for_namespaced_pods(self, namespace: str, until="ready"):
        out = client.CoreV1Api(self.apigetter()).list_namespaced_pod(namespace)

        pods = []
        for x in out.items:
            pods.append(x.metadata.name)

        for name in pods:
            self._wait_for(
                name,
                lambda x: client.CoreV1Api(x).list_namespaced_pod,
                namespace,
                until=until,
            )

    def wait_for_namespaced_service(self, name: str, namespace: str, until="delete", resource_version=None):
        self._wait_for(
            name,
            lambda x: client.CoreV1Api(x).list_namespaced_service,
            namespace,
            until=until,
            resource_version=resource_version,
        )

    def wait_for_namespaced_stateful_set(self, name: str, namespace: str, until="ready"):
        self._wait_for(
            name,
            lambda x: client.AppsV1Api(x).list_namespaced_stateful_set,
            namespace,
            until=until,
        )

    def wait_for_namespaced_job(
        self,
        name: str,
        namespace: str,
        until="complete",
        resource_version=None,
    ):
        self._wait_for(
            name,
            lambda x: client.BatchV1Api(x).list_namespaced_job,
            namespace,
            until=until,
            resource_version=resource_version,
        )
