from kubernetes import client
from fmperf.utils import Waiting


class Deleting:
    def __init__(self, apigetter, logger):
        self.apigetter = apigetter
        self.logger = logger
        self.waiting = Waiting(apigetter, logger)

    def delete_namespaced_service(self, name, namespace, wait=True):
        try:
            manifest = client.CoreV1Api(self.apigetter()).read_namespaced_service(name, namespace)
            exists = True
        except Exception:
            exists = False

        if not exists:
            self.logger.info("service %s does not exist in namespace %s" % (name, namespace))
            return

        self.logger.info("deleting namespaced service: %s %s" % (name, namespace))

        client.CoreV1Api(self.apigetter()).delete_namespaced_service(name, namespace)

        if wait:
            self.waiting.wait_for_namespaced_service(
                name,
                namespace,
                until="delete",
                resource_version=manifest.metadata.resource_version,
            )

    def delete_namespaced_deployment(self, name, namespace, wait=True):
        try:
            manifest = client.AppsV1Api(self.apigetter()).read_namespaced_deployment(name, namespace)
            exists = True
        except Exception:
            exists = False

        if not exists:
            self.logger.info("deployment %s does not exist in namespace %s" % (name, namespace))
            return

        self.logger.info("deleting namespaced deployment: %s %s" % (name, namespace))

        client.AppsV1Api(self.apigetter()).delete_namespaced_deployment(name, namespace, propagation_policy="Foreground")

        if wait:
            self.waiting.wait_for_namespaced_deployment(
                name,
                namespace,
                until="delete",
                resource_version=manifest.metadata.resource_version,
            )

    def delete_custom_object(self, name, api, plural, wait=True):
        group, version = api.split("/")

        try:
            client.CustomObjectsApi(self.apigetter()).get_cluster_custom_object(
                group,
                version,
                plural,
                name,
            )
            exists = True
        except Exception:
            exists = False

        if not exists:
            return

        self.logger.info("deleting custom object: %s %s %s" % (name, api, plural))

        client.CustomObjectsApi(self.apigetter()).delete_cluster_custom_object(
            group,
            version,
            plural,
            name,
            propagation_policy="Foreground",
        )

        if wait:
            self.waiting.wait_for_cluster_custom_object(
                name,
                group,
                version,
                plural,
                until="delete",
            )

    def delete_namespaced_custom_object(self, name, api, namespace, plural, wait=True):
        group, version = api.split("/")

        try:
            client.CustomObjectsApi(self.apigetter()).get_namespaced_custom_object(
                group,
                version,
                namespace,
                plural,
                name,
            )
            exists = True
        except Exception:
            exists = False

        if not exists:
            return

        self.logger.info("deleting namespaced custom object: %s %s %s %s" % (name, api, namespace, plural))

        client.CustomObjectsApi(self.apigetter()).delete_namespaced_custom_object(
            group,
            version,
            namespace,
            plural,
            name,
            propagation_policy="Foreground",
        )

        if wait:
            self.waiting.wait_for_namespaced_custom_object(
                name,
                group,
                version,
                namespace,
                plural,
                until="delete",
            )

    def delete_namespaced_job(self, name, namespace, wait=True):
        try:
            manifest = client.BatchV1Api(self.apigetter()).read_namespaced_job(name, namespace)
            exists = True
        except Exception:
            exists = False

        if not exists:
            return

        self.logger.info("deleting namespaced job: %s %s" % (name, namespace))

        client.BatchV1Api(self.apigetter()).delete_namespaced_job(
            name,
            namespace,
            propagation_policy="Foreground",
        )

        if wait:
            self.waiting.wait_for_namespaced_job(
                name,
                namespace,
                until="delete",
                resource_version=manifest.metadata.resource_version,
            )
