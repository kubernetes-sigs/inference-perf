from abc import ABC, abstractmethod
from typing import Any, List
import asyncio
import aiohttp


class ErrorsReport:
    ClientConnectorErrors: int
    TimeoutErrors: int
    ContentTypeErrors: int
    ClientOSErrors: int
    ServerDisconnectedErrors: int
    unknown_errors: int

    def __init__(self) -> None:
        self.ClientConnectorErrors = 0
        self.TimeoutErrors = 0
        self.ContentTypeErrors = 0
        self.ClientOSErrors = 0
        self.ServerDisconnectedErrors = 0
        self.unknown_errors = 0

    def to_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, int)}

    def record_error(self, error: Exception) -> None:
        if isinstance(error, aiohttp.client_exceptions.ClientConnectorError):
            self.ClientConnectorErrors += 1
            print(f"ClientConnectorError: {error}")
        elif isinstance(error, asyncio.TimeoutError):
            self.TimeoutErrors += 1
            print(f"TimeoutError: {error}")
        elif isinstance(error, aiohttp.client_exceptions.ContentTypeError):
            self.ContentTypeErrors += 1
            print(f"ContentTypeError: {error}")
        elif isinstance(error, aiohttp.client_exceptions.ClientOSError):
            self.ClientOSErrors += 1
            print(f"ClientOSError: {error}")
        elif isinstance(error, aiohttp.client_exceptions.ServerDisconnectedError):
            self.ServerDisconnectedErrors += 1
            print(f"ServerDisconnectedError: {error}")
        else:
            self.unknown_errors += 1
            print(f"Unknown error: {error}")

    def append_report(self, report: "ErrorsReport") -> None:
        self.ClientConnectorErrors += report.ClientConnectorErrors
        self.TimeoutErrors += report.TimeoutErrors
        self.ContentTypeErrors += report.ContentTypeErrors
        self.ClientOSErrors += report.ClientOSErrors
        self.ServerDisconnectedErrors += report.ServerDisconnectedErrors
        self.unknown_errors += report.unknown_errors


class Client(ABC):
    # The client will collect a summary of all observed errors
    Errors: ErrorsReport

    @abstractmethod
    def summary(self) -> Any:
        """
        Represents summary data derived from the inputs and outputs which depends on their specific data types.
        Subclasses should implement this at the client data type level (e.g., text-to-text, text-to-image).
        """
        pass

    @abstractmethod
    def request(self, *args: Any, **kwargs: Any) -> Any:
        """
        This is the method loadgen should use to make requests to a model server
        """
        pass

    @abstractmethod
    def describe_server_metrics(self) -> list[str]:
        """
        Returns list of model server metrics of interest.
        """
        pass
