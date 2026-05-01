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
from typing import List, Optional
from pydantic import BaseModel


class Text(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Image(BaseModel):
    pixels: int = 0
    bytes: int = 0
    aspect_ratio: float = 0.0


class Images(BaseModel):
    count: int = 0
    instances: List[Image] = []


class Video(BaseModel):
    pixels: int = 0
    bytes: int = 0
    aspect_ratio: float = 0.0
    frames: int = 0


class Videos(BaseModel):
    count: int = 0
    instances: List[Video] = []


class Audio(BaseModel):
    bytes: int = 0
    seconds: float = 0.0


class Audios(BaseModel):
    count: int = 0
    instances: List[Audio] = []


class Payload(BaseModel):
    text: Text
    image: Optional[Images] = None
    video: Optional[Videos] = None
    audio: Optional[Audios] = None
