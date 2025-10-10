from __future__ import annotations
from dataclasses import dataclass
from typing import Type
from datetime import datetime
from .config import TriggerSpec

trigger_implementations: dict[Type[TriggerSpec], Type[Trigger]] = {}

class TriggerMeta(type):
    def __new__(mcs, name, bases, dct, spec_cls: Type[TriggerSpec] = None):
        cls = super().__new__(mcs, name, bases, dct)
        if spec_cls:
            if spec_cls in trigger_implementations:
                raise ValueError(f'Trigger spec "{spec_cls}" already registered')
            trigger_implementations[spec_cls] = cls
        elif name != 'Trigger':
            raise TypeError(f'Trigger class {name} must have a spec_cls to register')
        return cls


@dataclass
class HitSample:
    ts: datetime
    hit: int


class Trigger(metaclass=TriggerMeta):
    def update(self, s: HitSample) -> None: ...
    def fired(self) -> bool: ...
    def reset(self) -> None: ...

def _init_trigger(trigger_class: type[Trigger], type: str, **kargs) -> Trigger:
    return trigger_class(**kargs)

def build_trigger(spec: TriggerSpec) -> Trigger:
    if type(spec) in trigger_implementations:
        return _init_trigger(trigger_implementations[type(spec)], **spec.model_dump())
    raise ValueError(f"Unknown trigger spec: {spec}")

