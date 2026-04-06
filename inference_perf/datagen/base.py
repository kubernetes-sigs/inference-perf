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
from inference_perf.apis import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution, SharedPrefix, TraceConfig
from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Dict, Any


class BaseGenerator(ABC):
    """Base class for all data/trace generators with common functionality."""

    api_config: APIConfig
    tokenizer: Optional[CustomTokenizer]

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        """Initialize base generator with common attributes.

        Args:
            api_config: API configuration (Chat, Completion, etc.)
            config: Data configuration
            tokenizer: Optional tokenizer for token counting
        """
        # Validate API type
        if api_config.type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {api_config}")

        self.api_config = api_config
        self.tokenizer = tokenizer

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        """Return list of supported API types (Chat, Completion, etc.)."""
        raise NotImplementedError

    def is_preferred_worker_requested(self) -> bool:
        """Whether this generator requests preferred worker routing.

        Returns False by default. Override to enable worker affinity.
        """
        return False


class DataGenerator(BaseGenerator):
    """Request-based data generation for standard load types (CONSTANT, POISSON, CONCURRENT, TRACE_REPLAY)."""

    input_distribution: Optional[Distribution]
    output_distribution: Optional[Distribution]
    shared_prefix: Optional[SharedPrefix]
    trace: Optional[TraceConfig]

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        """Initialize data generator with distribution and prefix support.

        Args:
            api_config: API configuration
            config: Data configuration including distributions and shared prefix
            tokenizer: Optional tokenizer
        """
        super().__init__(api_config, config, tokenizer)

        # DataGenerator-specific validation
        if (
            config.input_distribution is not None or config.output_distribution is not None
        ) and not self.is_io_distribution_supported():
            raise Exception("IO distribution not supported for this data generator")

        if config.shared_prefix is not None and not self.is_shared_prefix_supported():
            raise Exception("Shared prefix not supported for this data generator")

        self.input_distribution = config.input_distribution
        self.output_distribution = config.output_distribution
        self.shared_prefix = config.shared_prefix
        self.trace = config.trace

    @abstractmethod
    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        """Generate stream of individual requests.

        Yields:
            InferenceAPIData objects for each request
        """
        raise NotImplementedError

    @abstractmethod
    def is_io_distribution_supported(self) -> bool:
        """Whether this generator supports input/output distributions."""
        raise NotImplementedError

    @abstractmethod
    def is_shared_prefix_supported(self) -> bool:
        """Whether this generator supports shared prefix patterns."""
        raise NotImplementedError


class SessionGenerator(BaseGenerator):
    """Session-based trace replay for agentic workloads (TRACE_SESSION_REPLAY load type).

    Unlike DataGenerator which streams individual requests, SessionGenerator manages
    sessions with dependencies between requests. Used for replaying complex multi-turn
    conversations and agentic workflows.
    """

    @abstractmethod
    def get_session_count(self) -> int:
        """Return total number of sessions available for replay."""
        raise NotImplementedError

    @abstractmethod
    def get_session_info(self, session_index: int) -> Dict[str, Any]:
        """Get metadata about a specific session.

        Args:
            session_index: Index of the session (0-based)

        Returns:
            Dictionary with session metadata (session_id, file_path, num_events, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def get_session_event_indices(self, session_index: int) -> List[int]:
        """Get event indices for a specific session.

        Args:
            session_index: Index of the session (0-based)

        Returns:
            List of event indices that belong to this session
        """
        raise NotImplementedError

    @abstractmethod
    def get_session_events(self, session_index: int) -> List[LazyLoadInferenceAPIData]:
        """Get all events for a session as lazy-loadable data.

        This is used by LoadGen to dispatch a session's events.

        Args:
            session_index: Index of the session (0-based)

        Returns:
            List of LazyLoadInferenceAPIData for this session's events
        """
        raise NotImplementedError

    @abstractmethod
    def activate_session(self, session_id: str) -> None:
        """Activate a session (called by LoadGen when starting a session).

        Args:
            session_id: The session ID to activate
        """
        raise NotImplementedError

    @abstractmethod
    def check_session_completed(self, session_id: str) -> bool:
        """Check if a session has completed all its events.

        Args:
            session_id: The session ID to check

        Returns:
            True if all events in the session have completed
        """
        raise NotImplementedError

    @abstractmethod
    def build_session_metric(
        self,
        session_id: str,
        stage_id: int,
        start_time: float,
        end_time: float,
    ) -> Any:  # Returns SessionLifecycleMetric but avoiding circular import
        """Build session-level lifecycle metric.

        Args:
            session_id: The session ID
            stage_id: The stage ID this session ran in
            start_time: Session start time (epoch)
            end_time: Session end time (epoch)

        Returns:
            SessionLifecycleMetric object
        """
        raise NotImplementedError

    @abstractmethod
    def cleanup_session(self, session_id: str) -> None:
        """Clean up memory for a completed session.

        Removes all event outputs, messages, and completion tracking data
        for the specified session to prevent memory leaks.

        Args:
            session_id: The session ID to clean up
        """
        raise NotImplementedError

    # notify load gen whether request has preferred worker
    def is_preferred_worker_requested(self) -> bool:
        return False


class LazyLoadDataMixin(ABC):
    """
    Mixin for generators that support lazy loading (works with both DataGenerator and SessionGenerator).

    This is a capability marker - generators inherit from it to signal lazy-load support.
    The static get_request() method is a utility that checks for this capability at runtime.

    Useful for multiprocessing where the actual InferenceAPIData objects might be large
    or unpickleable, or need to be initialized in the worker process.
    """

    @abstractmethod
    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        """Load the actual data for a lazy placeholder.

        This method is called by worker processes to materialize lazy data.

        Args:
            data: LazyLoadInferenceAPIData placeholder

        Returns:
            Materialized InferenceAPIData object
        """
        raise NotImplementedError

    @staticmethod
    def get_request(data_generator: BaseGenerator, data: InferenceAPIData) -> InferenceAPIData:
        """Static utility method to handle lazy loading.

        Usage: LazyLoadDataMixin.get_request(datagen, data)

        Checks if datagen supports lazy loading and materializes data if needed.
        Works with both DataGenerator and SessionGenerator.

        Args:
            data_generator: The generator (DataGenerator or SessionGenerator)
            data: The data (may be lazy or already materialized)

        Returns:
            Materialized InferenceAPIData object
        """
        if isinstance(data, LazyLoadInferenceAPIData):
            if isinstance(data_generator, LazyLoadDataMixin):
                result = data_generator.load_lazy_data(data)
                # Propagate session_id from the lazy wrapper to the materialized object
                if data.session_id is not None and hasattr(result, "session_id"):
                    result.session_id = data.session_id
                return result
            else:
                raise NotImplementedError("Generator doesn't support lazy loading of requested InferenceAPIData")
        else:
            return data
