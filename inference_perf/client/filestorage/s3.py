# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
from typing import Any, List, Optional
import boto3
from botocore.config import Config as BotoConfig
from inference_perf.client.filestorage import StorageClient
from inference_perf.config import SimpleStorageServiceConfig
from inference_perf.utils import ReportFile

logger = logging.getLogger(__name__)


def _build_boto_config(addressing_style: Optional[str]) -> Optional[BotoConfig]:
    """Build a botocore Config that honors the configured S3 addressing style.

    Some S3-compatible object stores only accept virtual-hosted style requests
    (`bucket.host/key`) and reject path-style (`host/bucket/key`) with a
    `PathStyleRequestNotAllowed` error. Exposing this knob lets users target
    those backends without relying on environment-specific AWS config files.
    """
    if addressing_style is None:
        return None
    return BotoConfig(s3={"addressing_style": addressing_style})


class SimpleStorageServiceClient(StorageClient):
    def __init__(self, config: SimpleStorageServiceConfig) -> None:
        super().__init__(config=config)
        logger.debug("Created new S3 client")
        self.output_bucket = config.bucket_name
        client_kwargs: dict[str, Any] = {}
        if config.endpoint_url is not None:
            client_kwargs["endpoint_url"] = config.endpoint_url
        if config.region_name is not None:
            client_kwargs["region_name"] = config.region_name
        boto_config = _build_boto_config(config.addressing_style)
        if boto_config is not None:
            client_kwargs["config"] = boto_config
        self.client = boto3.client("s3", **client_kwargs)

    def save_report(self, reports: List[ReportFile]) -> None:
        filenames = [report.get_filename() for report in reports]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Duplicate filenames detected", filenames)

        for _, report in enumerate(reports):
            filename = report.get_filename()
            blob_path = f"{self.config.path if self.config.path else ''}/{self.config.report_file_prefix if self.config.report_file_prefix else ''}{filename}".lstrip(
                "/"
            )  # remove any leading slahes
            try:
                try:
                    self.client.head_object(Bucket=self.output_bucket, Key=blob_path)
                    logger.info(f"Skipping upload: s3://{self.output_bucket}/{blob_path} already exists")
                    continue
                except self.client.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        pass

                # Upload the files
                self.client.put_object(
                    Bucket=self.output_bucket,
                    Key=blob_path,
                    Body=json.dumps(report.get_contents()),
                    ContentType="application/json",
                )
                logger.info(f"Uploaded s3://{self.output_bucket}/{blob_path}")
            except Exception as e:
                logger.error(f"Failed to upload {blob_path}: {e}")
