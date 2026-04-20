import argparse
import json
import typing
from enum import Enum
from pydantic import BaseModel


def unwrap_type(annotation: typing.Any) -> typing.Tuple[typing.Any, bool]:
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = typing.get_args(annotation)
        # Handle Optional (Union[X, NoneType])
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0], True
    return annotation, False


def add_pydantic_args(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    model_cls: type[BaseModel],
    prefix: str = "",
    docs: typing.Optional[typing.List[str]] = None,
) -> typing.List[str]:
    """
    Recursively adds argparse arguments corresponding to a Pydantic model's fields.
    Returns a list of markdown doc strings for each argument.
    """
    if docs is None:
        docs = []

    for name, field in model_cls.model_fields.items():
        arg_name = f"--{prefix}{name}"
        help_text = field.description or f"Matches {prefix}{name} in config"

        annotation, is_optional = unwrap_type(field.annotation)
        origin = typing.get_origin(annotation)

        # Recurse into nested BaseModels
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            # To avoid "Nesting argument groups is deprecated" warnings, just keep using the same parser parent when possible
            # or add an argument group on the top level parser if `parser` is ArgumentParser
            target_parser = parser
            if isinstance(parser, argparse.ArgumentParser):
                target_parser = parser.add_argument_group(f"{prefix}{name} configuration")
            add_pydantic_args(target_parser, annotation, prefix=f"{prefix}{name}.", docs=docs)
        elif isinstance(annotation, type) and issubclass(annotation, Enum):
            choices = [e.value for e in annotation]
            parser.add_argument(arg_name, type=str, choices=choices, help=help_text, default=argparse.SUPPRESS)
            docs.append(f"| `{arg_name}` | Enum ({', '.join(choices)}) | {help_text} |")
        elif annotation is bool:
            # We accept "true", "1", "yes" as true, anything else as false
            parser.add_argument(
                arg_name,
                type=lambda x: str(x).lower() in ["true", "1", "yes"],
                help=f"{help_text} (true/false)",
                default=argparse.SUPPRESS,
            )
            docs.append(f"| `{arg_name}` | boolean | {help_text} |")
        elif annotation in (int, float, str):
            parser.add_argument(arg_name, type=annotation, help=help_text, default=argparse.SUPPRESS)
            docs.append(f"| `{arg_name}` | {annotation.__name__} | {help_text} |")
        else:
            is_json = False
            if origin in (list, dict, typing.List, typing.Dict) or (
                isinstance(annotation, type) and issubclass(annotation, (list, dict))
            ):
                is_json = True
            elif origin is typing.Union:
                args = typing.get_args(annotation)
                if any(typing.get_origin(arg) in (list, dict, typing.List, typing.Dict) for arg in args):
                    is_json = True

            if is_json:
                parser.add_argument(arg_name, type=json.loads, help=f"{help_text} (JSON string)", default=argparse.SUPPRESS)
                docs.append(f"| `{arg_name}` | JSON | {help_text} |")
            else:
                parser.add_argument(arg_name, type=str, help=f"{help_text} (handled as string)", default=argparse.SUPPRESS)
                docs.append(f"| `{arg_name}` | string | {help_text} |")

    return docs


def unflatten_dict(flat_dict: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """
    Converts a flat dictionary with dot-separated keys to a nested dictionary.
    """
    result: dict[str, typing.Any] = {}
    for key, value in flat_dict.items():
        if value is None:
            continue
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result
