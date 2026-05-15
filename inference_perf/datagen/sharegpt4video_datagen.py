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
"""ShareGPT4Video real-dataset loader for multimodal benchmarking.

The hot request path NEVER blocks on the network. A background thread pulls
source MP4 zips from the gated HuggingFace dataset repo into the local cache;
``load_lazy_data`` only ever samples from videos that are already extracted
on disk. Init blocks only until the first zip has been extracted so there's
always a non-empty pool when load generation starts.

Emits requests in the Frames wire format from PR #450 (N ``image_url`` blocks
at one insertion point), compatible with VLMs that don't accept ``video_url``
natively.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import zipfile
from typing import Any, Generator, List, Optional

from filelock import FileLock
from PIL import Image as PILImage

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType, DataConfig, ShareGPT4VideoConfig
from inference_perf.datagen.multimodal_sampling import resolution_to_wh
from inference_perf.payloads import (
    ImageRepresentation,
    MultimodalSpec,
    PreEncodedFramesVideoSpec,
    VideoRepresentation,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer

from .base import DataGenerator, LazyLoadDataMixin
from .gated_hf_dataset import GatedHFDataset

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIRNAME = "sharegpt4video_cache"
ZIP_FOLDER_PREFIX = "zip_folder/"


class ShareGPT4VideoDataGenerator(DataGenerator, LazyLoadDataMixin, GatedHFDataset):
    """Emits ShareGPT4Video requests using the Frames wire format.

    Source zips download asynchronously in the background; the on-disk video
    pool grows over time. ``load_lazy_data`` samples from whatever is currently
    available and never blocks on the network.
    """

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        DataGenerator.__init__(self, api_config, config, tokenizer)

        if config.sharegpt4video is None:
            raise ValueError("sharegpt4video config is required for ShareGPT4VideoDataGenerator")
        self.sg4v_config: ShareGPT4VideoConfig = config.sharegpt4video

        GatedHFDataset.__init__(self, token=self.sg4v_config.token)

        self.cache_dir = os.path.abspath(self.sg4v_config.cache_dir or os.path.join(os.getcwd(), DEFAULT_CACHE_DIRNAME))
        self.videos_dir = os.path.join(self.cache_dir, "videos")
        self.hf_cache_dir = os.path.join(self.cache_dir, "hf")
        self._extracted_marker_dir = os.path.join(self.cache_dir, ".extracted")
        self._lock_dir = os.path.join(self.cache_dir, ".locks")
        for d in (self.videos_dir, self.hf_cache_dir, self._extracted_marker_dir, self._lock_dir):
            os.makedirs(d, exist_ok=True)

        self._available_zips: List[str] = self._list_remote_zips()
        self._target_w, self._target_h = resolution_to_wh(self.sg4v_config.target_resolution)
        self._pil_format = "JPEG" if self.sg4v_config.representation == VideoRepresentation.JPEG_FRAMES else "PNG"

        # Stream the dataset once and index rows by video_path so the hot path
        # can do an O(1) lookup once a video appears on disk.
        self._rows_by_path: dict[str, dict[str, Any]] = self._build_dataset_index()

        # On-disk pool, kept current by the background downloader. Initialized
        # from whatever's already extracted (skip download work if the user
        # pre-populated the cache).
        self._available_lock = threading.Lock()
        self._available_paths: List[str] = self._scan_available_paths()

        self._log_cache_banner()

        # Background downloader. Daemon thread so it doesn't block process exit.
        self._stop_event = threading.Event()
        self._first_zip_ready = threading.Event()
        if self._available_paths:
            # Pre-populated cache → no need to wait at bootstrap.
            self._first_zip_ready.set()
        self._downloader_thread = threading.Thread(
            target=self._background_download_loop, name="sharegpt4video-downloader", daemon=True
        )
        self._downloader_thread.start()

        # Bootstrap: block until at least one zip is extracted (or downloader
        # exits without producing one, in which case we surface the failure).
        self._first_zip_ready.wait()
        with self._available_lock:
            if not self._available_paths:
                raise RuntimeError(
                    "ShareGPT4Video bootstrap failed: no usable videos available after first download. "
                    f"Check cache_dir '{self.cache_dir}' and HF repo connectivity."
                )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        # Snapshot the available pool under the lock so the background thread
        # can keep updating it while we work.
        with self._available_lock:
            snapshot = list(self._available_paths)
        if not snapshot:
            raise RuntimeError("ShareGPT4Video pool is empty after bootstrap; this should never happen.")

        # Deterministic cycle for reproducibility across workers with the same
        # data_index sequence. Pool grows over time, so later indices may pick
        # from a larger set — that's fine for a benchmark workload.
        idx = data.data_index % len(snapshot)
        video_path = snapshot[idx]
        row = self._rows_by_path[video_path]

        local_video = os.path.join(self.videos_dir, video_path)
        keyframes = self._select_keyframes(row)
        frames_bytes = self._extract_frames(local_video, keyframes)
        prompt = self._build_prompt(row)

        spec = MultimodalSpec(
            videos=[
                PreEncodedFramesVideoSpec(
                    width=self._target_w,
                    height=self._target_h,
                    frames=len(frames_bytes),
                    insertion_point=self.sg4v_config.insertion_point,
                    frame_representation=(
                        ImageRepresentation.JPEG
                        if self.sg4v_config.representation == VideoRepresentation.JPEG_FRAMES
                        else ImageRepresentation.PNG
                    ),
                    frames_bytes=frames_bytes,
                )
            ]
        )
        return ChatCompletionAPIData(
            messages=[ChatMessage(role="user", content=prompt)],
            multimodal_spec=spec,
        )

    def stop(self) -> None:
        """Signal the background downloader to exit. Daemon thread; best-effort."""
        self._stop_event.set()

    # -------------------------- dataset index --------------------------

    def _build_dataset_index(self) -> dict[str, dict[str, Any]]:
        load_kwargs: dict[str, Any] = {
            "streaming": True,
            "split": self.sg4v_config.hf_split,
            "cache_dir": self.hf_cache_dir,
        }
        if self.sg4v_config.hf_data_files is not None:
            load_kwargs["data_files"] = self.sg4v_config.hf_data_files

        logger.info("Indexing ShareGPT4Video dataset rows ...")
        idx: dict[str, dict[str, Any]] = {}
        for row in self._load_dataset(self.sg4v_config.hf_dataset_name, **load_kwargs):
            if not isinstance(row, dict):
                continue
            vp = row.get("video_path")
            if not isinstance(vp, str) or not vp:
                continue
            if not row.get("keyframe") or not row.get("captions"):
                continue
            # Keep only fields we use to minimize memory footprint.
            idx[vp] = {
                "video_path": vp,
                "keyframe": row["keyframe"],
                "captions": row["captions"],
            }
        logger.info("Indexed %d ShareGPT4Video rows", len(idx))
        return idx

    # -------------------------- HF zip discovery / download --------------------------

    def _list_remote_zips(self) -> List[str]:
        """List zips under ``zip_folder/`` on the HF dataset repo."""
        from huggingface_hub import HfApi

        api = HfApi(token=self._hf_token)
        files = api.list_repo_files(repo_id=self.sg4v_config.hf_dataset_name, repo_type="dataset")
        zips = [f for f in files if f.startswith(ZIP_FOLDER_PREFIX) and f.endswith(".zip")]
        return sorted(zips)

    def _log_cache_banner(self) -> None:
        n_zips = len(self._available_zips)
        n_available = len(self._available_paths)
        banner = "=" * 78
        logger.warning(banner)
        logger.warning("ShareGPT4Video cache directory: %s", self.cache_dir)
        logger.warning("  videos/  -> extracted MP4s (sampled by load_lazy_data; never blocks)")
        logger.warning("  hf/      -> HuggingFace datasets streaming cache (captions + indices)")
        logger.warning("Pre-populate ``videos/`` to skip downloads. Discovered %d source zips on", n_zips)
        logger.warning("the HF repo ``%s`` (each typically 15-21 GB).", self.sg4v_config.hf_dataset_name)
        logger.warning("Starting on-disk pool: %d videos. A background thread will keep pulling zips", n_available)
        logger.warning("from HF as the benchmark runs; the pool grows over time.")
        logger.warning("Set HF_HUB_ENABLE_HF_TRANSFER=1 for faster parallel downloads.")
        logger.warning(banner)

    def _background_download_loop(self) -> None:
        """Daemon loop: download + extract every remaining zip, one at a time."""
        try:
            for repo_path in self._available_zips:
                if self._stop_event.is_set():
                    return
                marker = self._marker_path(repo_path)
                if os.path.exists(marker):
                    # Already extracted by a previous run or worker — just ensure
                    # any new paths are in the pool, then move on.
                    self._refresh_available_paths()
                    self._first_zip_ready.set()
                    continue
                try:
                    self._download_and_extract_zip(repo_path)
                except Exception as e:
                    logger.warning("ShareGPT4Video: failed to download/extract %s: %s", repo_path, e)
                    continue
                self._refresh_available_paths()
                self._first_zip_ready.set()
        finally:
            # Even on failure, unblock bootstrap so init can raise rather than hang.
            self._first_zip_ready.set()
            logger.info("ShareGPT4Video background downloader finished.")

    def _marker_path(self, repo_path: str) -> str:
        return os.path.join(self._extracted_marker_dir, os.path.basename(repo_path) + ".done")

    def _lock_path(self, repo_path: str) -> str:
        return os.path.join(self._lock_dir, os.path.basename(repo_path) + ".lock")

    def _download_and_extract_zip(self, repo_path: str) -> None:
        """Download a zip from HF and extract its videos into ``videos_dir``.

        Cross-process safe via ``filelock``: the first worker to acquire the
        per-zip lock does the work; others wait and find the marker.
        """
        from huggingface_hub import hf_hub_download

        marker = self._marker_path(repo_path)
        with FileLock(self._lock_path(repo_path)):
            if os.path.exists(marker):
                return
            logger.warning("ShareGPT4Video: downloading %s ...", repo_path)
            local_zip = hf_hub_download(
                repo_id=self.sg4v_config.hf_dataset_name,
                filename=repo_path,
                repo_type="dataset",
                token=self._hf_token,
                cache_dir=self.hf_cache_dir,
            )
            videos_real = os.path.realpath(self.videos_dir)
            with zipfile.ZipFile(local_zip) as zf:
                for member in zf.infolist():
                    target = os.path.normpath(os.path.join(self.videos_dir, member.filename))
                    if not (target == videos_real or target.startswith(videos_real + os.sep)):
                        logger.warning("Skipping zip entry with unsafe path: %s", member.filename)
                        continue
                    if member.is_dir():
                        os.makedirs(target, exist_ok=True)
                        continue
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
            # Atomic-ish marker write: tmp + rename so partial writes can't be
            # mistaken for completion by a concurrent scanner.
            tmp = marker + ".tmp"
            with open(tmp, "w") as f:
                f.write(repo_path)
            os.replace(tmp, marker)
            logger.info("ShareGPT4Video: extracted %s -> %s", repo_path, self.videos_dir)

    # -------------------------- on-disk pool --------------------------

    def _scan_available_paths(self) -> List[str]:
        """Return sorted list of dataset video_paths that exist on disk."""
        available: List[str] = []
        for vp in self._rows_by_path.keys():
            if os.path.isfile(os.path.join(self.videos_dir, vp)):
                available.append(vp)
        available.sort()
        return available

    def _refresh_available_paths(self) -> None:
        new_list = self._scan_available_paths()
        with self._available_lock:
            self._available_paths = new_list

    # -------------------------- per-row helpers --------------------------

    def _select_keyframes(self, row: dict[str, Any]) -> List[int]:
        raw = row.get("keyframe") or []
        indices = [int(x) for x in raw]
        cap = self.sg4v_config.max_frames_per_request
        if len(indices) > cap:
            step = len(indices) / cap
            indices = [indices[int(i * step)] for i in range(cap)]
        return indices

    def _build_prompt(self, row: dict[str, Any]) -> str:
        captions = row.get("captions") or []
        parts: list[str] = []
        for c in captions:
            if isinstance(c, dict):
                text = c.get("content") or c.get("caption") or ""
            else:
                text = str(c)
            text = text.strip()
            if text:
                parts.append(text)
        return " ".join(parts) if parts else "Describe this video."

    def _extract_frames(self, video_path: str, keyframe_indices: List[int]) -> List[bytes]:
        import av

        encoded: List[bytes] = []
        wanted = sorted(set(int(i) for i in keyframe_indices if i >= 0))
        if not wanted:
            raise ValueError(f"No usable keyframe indices for {video_path}")
        wanted_set = set(wanted)
        max_idx = wanted[-1]

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted_set:
                    img = frame.to_image()  # type: ignore[no-untyped-call]
                    encoded.append(self._encode_pil(img))
                if i >= max_idx:
                    break

        if not encoded:
            raise RuntimeError(
                f"PyAV decoded no frames at indices {wanted} from {video_path}; "
                "the video may be shorter than the recorded keyframe range."
            )
        return encoded

    def _encode_pil(self, img: PILImage.Image) -> bytes:
        resized = img.resize((self._target_w, self._target_h))
        buf = io.BytesIO()
        if self._pil_format == "JPEG":
            if resized.mode != "RGB":
                resized = resized.convert("RGB")
            resized.save(buf, format="JPEG", quality=75)
        else:
            resized.save(buf, format="PNG")
        return buf.getvalue()
