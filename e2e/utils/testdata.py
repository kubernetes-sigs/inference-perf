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
import os
import pathlib
import subprocess

TEST_E2E_DIR = pathlib.Path(__file__).parent.parent
TEST_E2E_TESTDATA = TEST_E2E_DIR.joinpath("testdata")


def extract_tarball(name: str | pathlib.Path) -> pathlib.Path:
    """
    Extract tarball with the given path to the directory that that tarball is
    in.

    The returned path is the folder containing the content of the tarball, named
    after the tarball name itself without the extension.
    """
    name = pathlib.Path(name).resolve()

    dest = name
    while dest.suffix:
        dest = dest.with_suffix("")

    if not dest.is_dir():
        if not name.is_file():
            raise FileNotFoundError(f"Tarball {name} not found!")

        os.makedirs(dest)
        subprocess.run(["tar", "-xzvf", name, "-C", dest], check=True)

    return dest
