from dataclasses import dataclass
import logging
from pathlib import Path
import random
from typing import List
from pydantic import BaseModel
import yaml
from inference_perf.utils.shared_prefix_trace_reader import SharedPrefixTraceEntry


class ToTConfig(BaseModel):
    num_layers: int = 3
    branching_factor: int = 3
    expand_top: int = 2
    system_prompt_len: int = 100
    thought_prompt_len: int = 20
    thought_len: int = 100
    start_timestamp: float = 0.0
    graph_output: str | None = None


@dataclass
class ToTNode:
    id: int
    len: int
    timestamp: float
    level: int
    parent_id: int | None = None
    selected: bool = False
    is_final: bool = False


class ToTTraceGenerator:
    def __init__(self, config: ToTConfig):
        self.cfg = config
        self.next_prefix_id = 1
        self.nodes: List[ToTNode] = []

    def generate(self) -> List[SharedPrefixTraceEntry]:
        # Start with the root context
        root_node = ToTNode(
            id=self.next_prefix_id,
            len=self.cfg.system_prompt_len,
            timestamp=self.cfg.start_timestamp,
            level=0,
            selected=True,
        )
        current_layer_nodes = [root_node]
        self.nodes.append(root_node)
        self.next_prefix_id += 1

        traces = []

        for layer in range(self.cfg.num_layers):
            next_layer_nodes = []

            # For each node in the current layer (beam), generate branches
            for node in current_layer_nodes:
                # Make 'branching_factor' requests
                for i in range(self.cfg.branching_factor):
                    # Stagger requests slightly
                    req_timestamp = node.timestamp + (i * 0.01)

                    # Trace entry
                    # shared_prefix_id = node.id refers to the parent context
                    entry = SharedPrefixTraceEntry(
                        timestamp=req_timestamp,
                        shared_prefix_length=int(node.len),
                        tail_input_length=self.cfg.thought_prompt_len,
                        output_length=self.cfg.thought_len,
                        shared_prefix_id=int(node.id),
                    )
                    traces.append(entry)

                    # Calculate new length for the next node
                    new_node_len = node.len + self.cfg.thought_prompt_len + self.cfg.thought_len

                    # Estimate completion time (e.g. 100 tokens/sec) for next timestamp
                    # This determines when the NEXT layer can start from this node
                    completion_time = req_timestamp + (self.cfg.thought_len / 100.0)

                    new_node = ToTNode(
                        id=self.next_prefix_id,
                        len=new_node_len,
                        timestamp=completion_time,
                        level=layer + 1,
                        parent_id=node.id,
                    )
                    self.nodes.append(new_node)
                    self.next_prefix_id += 1
                    next_layer_nodes.append(new_node)

            # Prune / Beam Search for next layer
            if not next_layer_nodes:
                break

            if self.cfg.expand_top < len(next_layer_nodes):
                # Pick 'expand_top' random nodes to continue
                current_layer_nodes = random.sample(next_layer_nodes, self.cfg.expand_top)
            else:
                current_layer_nodes = next_layer_nodes

            for n in current_layer_nodes:
                n.selected = True

        if current_layer_nodes:
            current_layer_nodes[0].is_final = True

        # Sort by timestamp to ensure chronological order
        traces.sort(key=lambda x: x.timestamp)
        return traces

    def write_graph(self, filename: str) -> None:
        """Generates a Graphviz DOT file for the Tree of Thoughts."""
        logger = logging.getLogger(__name__)
        try:
            with open(filename, "w") as f:
                f.write("digraph TreeOfThought {\n")
                f.write('    rankdir="TB";\n')
                f.write('    node [shape=box, style="filled,rounded"];\n')

                for node in self.nodes:
                    # Style based on selection
                    fillcolor = "lightblue" if node.selected else "white"
                    style = "filled,rounded"
                    if node.selected:
                        style += ",bold"

                    # Label with small text for tokens
                    label_text = f"<{node.id}<br/>"
                    if node.is_final:
                        label_text += "&#9733; "  # Star symbol
                    label_text += f'<font point-size="10">shared_prefix_id={node.parent_id if node.parent_id is not None else "N/A"}</font>>'

                    f.write(f'    {node.id} [label={label_text}, fillcolor="{fillcolor}", style="{style}"];\n')

                    if node.parent_id is not None:
                        f.write(f"    {node.parent_id} -> {node.id};\n")

                f.write("}\n")
            logger.info(f"Graphviz file written to {filename}")
        except Exception as e:
            logger.error(f"Failed to write graph file: {e}")


def main(config_file: str) -> None:
    logger = logging.getLogger(__name__)

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Extract tracegen config if nested under a 'tracegen' key, otherwise assume root
    tot_config_data = config_data.get("tracegen", config_data)

    # Filter for relevant keys to avoid pydantic errors with full benchmark configs
    # Or just use the keys that match the model
    # For now, let's assume the config file provided IS the ToT config or contains it
    try:
        config = ToTConfig(**tot_config_data)
    except Exception:
        # Fallback: maybe it's a full benchmark config with a 'data.shared_prefix' section?
        # But the requirement says "devutils/configs/tracegen_tot.yaml", implying a specific config.
        # Let's keep it simple for now.
        config = ToTConfig(**tot_config_data)

    output_path = Path(config_file).parent / "trace_tot.jsonl"
    if "output_file" in tot_config_data:
        output_path = Path(tot_config_data["output_file"])

    generator = ToTTraceGenerator(config)
    traces = generator.generate()

    if config.graph_output:
        # Assuming the graph_output is an absolute or relative path from current working directory
        generator.write_graph(config.graph_output)

    logger.info(f"Generated {len(traces)} trace entries.")

    with open(output_path, "w") as f:
        for t in traces:
            f.write(t.model_dump_json() + "\n")

    logger.info(f"Saved traces to {output_path}")
