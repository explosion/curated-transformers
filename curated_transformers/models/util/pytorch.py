from typing import Callable, List, Optional
from dataclasses import dataclass
from torch.nn import Module


@dataclass
class ModuleIterator:
    # Current module.
    module: Module
    # Module name.
    name: str
    # Current dot path of the module. Includes the name.
    prefix: str
    # Parent module.
    parent: Optional[Module]


def apply_to_module(module: Module, func: Callable[[ModuleIterator], None]):
    """Apply a function the module and its submodules in a breadth-first
    fashion.

    :param module:
        Root module.
    :param func:
        A callable that takes a module iterator as its argument.
    """
    queue: List[ModuleIterator] = [ModuleIterator(module, "", "", None)]

    while queue:
        itr = queue.pop(0)
        func(itr)

        for name, child in itr.module._modules.items():
            if child is not None:
                child_prefix = f"{itr.prefix}.{name}" if itr.prefix else name
                queue.append(ModuleIterator(child, name, child_prefix, itr.module))
