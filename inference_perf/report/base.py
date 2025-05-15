import json
from typing import Any


class ReportFile:
    name: str
    contents: Any

    def __init__(self, name: str, contents: Any):
        self.name = f"{name}.json"
        self.contents = contents
        self._store_locally()

    def _store_locally(self) -> None:
        filename = self.get_filename()
        contents = self.get_contents()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(contents, indent=2))

    def get_filename(self) -> str:
        return self.name

    def get_contents(self) -> Any:
        return self.contents
