from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Type, Any, Optional, Dict, Tuple
from datetime import datetime
from .config import TriggerSpec

trigger_implementations: dict[Type[TriggerSpec], type] = {}


class TriggerMeta(ABCMeta):
    def __new__(
        mcs: Type[TriggerMeta],
        name: str,
        bases: Tuple[type, ...],
        dct: Dict[str, Any],
        spec_cls: Optional[Type[TriggerSpec]] = None,
    ) -> type:
        cls = super().__new__(mcs, name, bases, dct)
        if spec_cls:
            if spec_cls in trigger_implementations:
                raise ValueError(f'Trigger spec "{spec_cls}" already registered')
            trigger_implementations[spec_cls] = cls
        elif name != "Trigger":
            raise TypeError(f"Trigger class {name} must have a spec_cls to register")
        return cls


@dataclass
class HitSample:
    ts: datetime
    hit: int


class Trigger(metaclass=TriggerMeta):
    @abstractmethod
    def update(self, s: HitSample) -> None: ...

    @abstractmethod
    def fired(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...


def _init_trigger(trigger_class: type[Trigger], type: str, **kargs: Any) -> Trigger:
    return trigger_class(**kargs)


def build_trigger(spec: TriggerSpec) -> Trigger:
    if type(spec) in trigger_implementations:
        return _init_trigger(trigger_implementations[type(spec)], **spec.model_dump())
    raise ValueError(f"Unknown trigger spec: {spec}")
