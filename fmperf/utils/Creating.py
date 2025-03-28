from kubernetes import client
from fmperf.utils import Waiting
import pkg_resources
import yaml


class Creating:
    def __init__(self, apigetter, logger, ignore_exists=False):
        self.apigetter = apigetter
        self.logger = logger
        self.waiting = Waiting(apigetter, logger)
        self.ignore_exists = ignore_exists

    def create_custom_object(self, name, api, plural, payload, wait_until="ready"):
        self.logger.info("creating custom object: %s %s %s" % (name, api, plural))

        group, version = api.split("/")

        try:
            client.CustomObjectsApi(self.apigetter()).create_cluster_custom_object(
                group,
                version,
                plural,
                payload,
            )
        except Exception as e:
            if not (isinstance(e, client.exceptions.ApiException) and e.reason == "Conflict" and self.ignore_exists):
                raise

        if wait_until is None:
            return

        self.waiting.wait_for_cluster_custom_object(name, group, version, plural, until=wait_until)

    def create_namespaced_custom_object(self, name, api, namespace, plural, payload, wait_until="ready"):
        self.logger.info("creating namespaced custom object: %s %s %s %s" % (name, api, namespace, plural))

        group, version = api.split("/")

        try:
            client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                group,
                version,
                namespace,
                plural,
                payload,
            )
        except Exception as e:
            if not (isinstance(e, client.exceptions.ApiException) and e.reason == "Conflict" and self.ignore_exists):
                raise

        if wait_until is None:
            return

        self.waiting.wait_for_namespaced_custom_object(name, group, version, namespace, plural, until=wait_until)

    def create_namespaced_deployment(self, name, namespace, payload, wait_until="available"):
        self.logger.info("creating namespaced deployment: %s %s" % (name, namespace))

        try:
            client.AppsV1Api(self.apigetter()).create_namespaced_deployment(namespace=namespace, body=payload)
        except Exception as e:
            if not (isinstance(e, client.exceptions.ApiException) and e.reason == "Conflict" and self.ignore_exists):
                raise

        if wait_until is None:
            return

        self.waiting.wait_for_namespaced_deployment(name, namespace, until=wait_until)

    def create_namespaced_service(self, name, namespace, payload):
        self.logger.info("creating namespaced service: %s %s" % (name, namespace))

        try:
            client.CoreV1Api(self.apigetter()).create_namespaced_service(namespace=namespace, body=payload)
        except Exception as e:
            if not (isinstance(e, client.exceptions.ApiException) and e.reason == "Conflict" and self.ignore_exists):
                raise

    def create_from_resource(self, resource):
        data = []
        with pkg_resources.resource_stream("fmperf.resources", resource) as f:
            for x in yaml.load_all(f, Loader=yaml.FullLoader):
                if x is not None:
                    data.append(x)

        for x in data:
            self.create_from_yaml(x)

    def create_from_yaml(self, x):
        # self.logger.info("Creating resource %s %s %s" % (x["apiVersion"], x["kind"], x["metadata"]["name"]))

        try:
            if x["kind"] == "CustomResourceDefinition":
                client.ApiextensionsV1Api(self.apigetter()).create_custom_resource_definition(body=x)
            elif x["apiVersion"] == "v1":
                v1api = client.CoreV1Api(self.apigetter())
                namespace = x["metadata"]["namespace"] if ("namespace" in x["metadata"]) else "default"
                if x["kind"] == "ServiceAccount":
                    v1api.create_namespaced_service_account(namespace, body=x)
                elif x["kind"] == "ConfigMap":
                    v1api.create_namespaced_config_map(namespace, body=x)
                elif x["kind"] == "Service":
                    v1api.create_namespaced_service(namespace, body=x)
                elif x["kind"] == "Namespace":
                    v1api.create_namespace(x)
                elif x["kind"] == "Secret":
                    v1api.create_namespaced_secret(namespace, body=x)
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "rbac.authorization.k8s.io/v1":
                rbacapi = client.RbacAuthorizationV1Api(self.apigetter())
                if x["kind"] == "ClusterRole":
                    rbacapi.create_cluster_role(body=x)
                elif x["kind"] == "ClusterRoleBinding":
                    rbacapi.create_cluster_role_binding(body=x)
                elif x["kind"] == "Role":
                    rbacapi.create_namespaced_role(x["metadata"]["namespace"], body=x)
                elif x["kind"] == "RoleBinding":
                    rbacapi.create_namespaced_role_binding(x["metadata"]["namespace"], body=x)
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "admissionregistration.k8s.io/v1":
                if x["kind"] == "ValidatingWebhookConfiguration":
                    client.AdmissionregistrationV1Api(self.apigetter()).create_validating_webhook_configuration(x)
                elif x["kind"] == "MutatingWebhookConfiguration":
                    client.AdmissionregistrationV1Api(self.apigetter()).create_mutating_webhook_configuration(x)
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "networking.istio.io/v1alpha3":
                if x["kind"] == "EnvoyFilter":
                    client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                        "networking.istio.io",
                        "v1alpha3",
                        x["metadata"]["namespace"],
                        "envoyfilters",
                        x,
                    )
                elif x["kind"] == "Gateway":
                    client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                        "networking.istio.io",
                        "v1alpha3",
                        x["metadata"]["namespace"],
                        "gateways",
                        x,
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "networking.k8s.io/v1":
                if x["kind"] == "IngressClass":
                    client.NetworkingV1Api(self.apigetter()).create_ingress_class(x)
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "security.istio.io/v1beta1":
                if x["kind"] == "PeerAuthentication":
                    client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                        "security.istio.io",
                        "v1beta1",
                        x["metadata"]["namespace"],
                        "peerauthentications",
                        x,
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "cert-manager.io/v1":
                if x["kind"] == "Certificate":
                    client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                        "cert-manager.io",
                        "v1",
                        x["metadata"]["namespace"],
                        "certificates",
                        x,
                    )
                elif x["kind"] == "Issuer":
                    client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                        "cert-manager.io",
                        "v1",
                        x["metadata"]["namespace"],
                        "issuers",
                        x,
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "cert-manager.io/v1alpha2":
                if x["kind"] == "Certificate":
                    client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                        "cert-manager.io",
                        "v1alpha2",
                        x["metadata"]["namespace"],
                        "certificates",
                        x,
                    )
                elif x["kind"] == "Issuer":
                    client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                        "cert-manager.io",
                        "v1alpha2",
                        x["metadata"]["namespace"],
                        "issuers",
                        x,
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "apps/v1":
                if x["kind"] == "Deployment":
                    client.AppsV1Api(self.apigetter()).create_namespaced_deployment(x["metadata"]["namespace"], body=x)
                elif x["kind"] == "StatefulSet":
                    client.AppsV1Api(self.apigetter()).create_namespaced_stateful_set(x["metadata"]["namespace"], body=x)
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "policy/v1":
                if x["kind"] == "PodDisruptionBudget":
                    client.PolicyV1Api(self.apigetter()).create_namespaced_pod_disruption_budget(
                        x["metadata"]["namespace"], body=x
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "policy/v1beta1":
                if x["kind"] == "PodDisruptionBudget":
                    client.PolicyV1beta1Api(self.apigetter()).create_namespaced_pod_disruption_budget(
                        x["metadata"]["namespace"], body=x
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "autoscaling/v2beta1":
                if x["kind"] == "HorizontalPodAutoscaler":
                    client.AutoscalingV2beta1Api(self.apigetter()).create_namespaced_horizontal_pod_autoscaler(
                        x["metadata"]["namespace"], body=x
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "autoscaling/v2beta2":
                if x["kind"] == "HorizontalPodAutoscaler":
                    client.AutoscalingV2beta2Api(self.apigetter()).create_namespaced_horizontal_pod_autoscaler(
                        x["metadata"]["namespace"], body=x
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "caching.internal.knative.dev/v1alpha1":
                if x["kind"] == "Image":
                    client.CustomObjectsApi(self.apigetter()).create_namespaced_custom_object(
                        "caching.internal.knative.dev",
                        "v1alpha1",
                        x["metadata"]["namespace"],
                        "images",
                        x,
                    )
                else:
                    raise NotImplementedError()
            elif x["apiVersion"] == "serving.kserve.io/v1alpha1":
                if x["kind"] == "ClusterServingRuntime":
                    client.CustomObjectsApi(self.apigetter()).create_cluster_custom_object(
                        "serving.kserve.io",
                        "v1alpha1",
                        "clusterservingruntimes",
                        x,
                    )
                else:
                    raise NotImplementedError()

            else:
                raise NotImplementedError()
        except Exception as e:
            if not (isinstance(e, client.exceptions.ApiException) and e.reason == "Conflict" and self.ignore_exists):
                raise e
