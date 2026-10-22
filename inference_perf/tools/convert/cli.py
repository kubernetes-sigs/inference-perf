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
"""The ``inference-perf-convert`` console entry point.

Usage::

    inference-perf-convert vllm-bench --peer-version v0.10.0 -o config.yaml -- \\
        --dataset-name random --num-prompts 600 --request-rate 40 ...
    inference-perf-convert aiperf-profile --peer-version v0.12.0 -- \\
        --model m --endpoint-type completions --streaming --osl 64 ...

Exit codes: 0 converted; 2 refused, with every ``REFUSE <flag>: <reason>``
line on stderr and no config written; 1 usage or internal error.
"""

import argparse
import logging
import sys
from typing import List, Optional

from inference_perf.tools.convert.aiperf_profile import convert_aiperf_profile
from inference_perf.tools.convert.emit import emit_yaml
from inference_perf.tools.convert.model import Conversion, PeerUsageError
from inference_perf.tools.convert.vllm_bench import convert_vllm_bench

logger = logging.getLogger(__name__)

EXIT_CONVERTED = 0
EXIT_USAGE = 1
EXIT_REFUSED = 2

_FRONTENDS = {
    "vllm-bench": convert_vllm_bench,
    "aiperf-profile": convert_aiperf_profile,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-perf-convert",
        description="Convert a peer benchmarking tool's arguments into an inference-perf config."
        " Refuses rather than guesses where the mapping is lossy (#755).",
    )
    parser.add_argument("tool", choices=sorted(_FRONTENDS), help="Peer tool whose arguments follow after --")
    parser.add_argument(
        "--peer-version",
        required=True,
        help="Peer tool version the arguments are written for. Only verified versions convert.",
    )
    parser.add_argument("-o", "--output", default=None, help="Write the config here instead of stdout")
    parser.add_argument(
        "--log-level", help="Logging level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    return parser


def run(argv: Optional[List[str]] = None) -> int:
    # Everything after the first `--` belongs to the peer tool verbatim;
    # argparse must never see it, since peer flags collide with our own.
    own_argv = list(sys.argv[1:]) if argv is None else list(argv)
    peer_argv: List[str] = []
    if "--" in own_argv:
        split = own_argv.index("--")
        own_argv, peer_argv = own_argv[:split], own_argv[split + 1 :]
    args = _build_parser().parse_args(own_argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    try:
        conversion: Conversion = _FRONTENDS[args.tool](peer_argv, args.peer_version)
    except PeerUsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE

    if conversion.refusals:
        for refusal in conversion.refusals:
            print(f"REFUSE {refusal}", file=sys.stderr)
        print(
            f"refused: {len(conversion.refusals)} reason(s); no config written"
            " (a config that looks converted but changes the workload is the failure mode this tool exists"
            " to prevent)",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    text = emit_yaml(conversion)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(text)
        logger.info("wrote %s", args.output)
    else:
        sys.stdout.write(text)
    return EXIT_CONVERTED


def main_cli() -> None:
    sys.exit(run())
