# Copyright 2025 The Kubernetes Authors.
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
from typing import List
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from inference_perf.client.filestorage import StorageClient
from inference_perf.config import GoogleCloudStorageConfig
from inference_perf.utils import ReportFile

logger = logging.getLogger(__name__)


class GoogleCloudStorageClient(StorageClient):
    def __init__(self, config: GoogleCloudStorageConfig) -> None:
        super().__init__(config=config)
        logger.debug("Created new GCS client")
        self.output_bucket = config.bucket_name
        self.client = storage.Client()

        self.bucket = self.client.lookup_bucket(config.bucket_name)
        if self.bucket is None:
            raise ValueError(f"GCS bucket '{config.bucket_name}' does not exist or is inaccessible.")

    def save_report(self, reports: List[ReportFile]) -> None:
        filenames = [report.get_filename() for report in reports]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Duplicate filenames detected", filenames)

        for _, report in enumerate(reports):
            filename = report.get_filename()
            blob_path = f"{self.config.path if self.config.path else ''}/{self.config.report_file_prefix if self.config.report_file_prefix else ''}{filename}"
            blob = self.bucket.blob(blob_path)

            if blob.exists():
                logger.info(f"Skipping upload: gs://{self.output_bucket}/{blob_path} already exists")
                continue

            try:
                blob.upload_from_string(json.dumps(report.get_contents()), content_type="application/json")
                logger.info(f"Uploaded gs://{self.output_bucket}/{blob_path}")
            except GoogleCloudError as e:
                logger.error(f"Failed to upload {blob_path}: {e}")
