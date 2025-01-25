# Copyright 2025
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
from jinja2 import Template
import os


class LoadGenerator:
    def __init__(self) -> None:
        templates_path = os.path.dirname(os.path.realpath(__file__))
        with open(templates_path + "/templates/script.js.j2") as tmpfile:
            self.template = Template(tmpfile.read())

    def generate(self, **kwargs: str) -> str:
        return self.template.render(kwargs)
