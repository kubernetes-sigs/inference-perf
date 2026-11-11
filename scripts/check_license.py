#!/usr/bin/env python3
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

import re
import sys

# The header as it should be written in new files. YEAR stands in for the
# copyright year and is turned into a pattern below, so a file keeps the year
# it was written in rather than having to match the year the check runs in.
LICENSE_TEMPLATE = """# Copyright YEAR The Kubernetes Authors.
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
"""


def _build_pattern() -> "re.Pattern[str]":
    # The template matched literally, except for the year, which is any four
    # digits, and the indentation of the license URL. Both URL indentations
    # appear in the tree and both are the Apache boilerplate, so neither is
    # worth rewriting files over.
    escaped = re.escape(LICENSE_TEMPLATE)
    escaped = escaped.replace("YEAR", r"\d{4}")
    escaped = escaped.replace(re.escape("# http://"), r"#[ ]+http://")
    return re.compile(escaped)


LICENSE_PATTERN = _build_pattern()


def has_license_header(content: str) -> bool:
    # A shebang has to come first in the file, so the header follows it rather
    # than opening the file. Some scripts also leave a blank line in between.
    if content.startswith("#!"):
        _, _, content = content.partition("\n")
        content = content.lstrip("\n")
    return LICENSE_PATTERN.match(content) is not None


def check_file(filename: str) -> bool:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return False

    # An empty file carries no code to license.
    if not content.strip():
        return True

    if not has_license_header(content):
        print(f"File missing or incorrect license header: {filename}")
        return False
    return True


def main() -> None:
    failed = False
    for filename in sys.argv[1:]:
        if not check_file(filename):
            failed = True

    if failed:
        sys.exit(1)
    print("All files passed license check.")


if __name__ == "__main__":
    main()
