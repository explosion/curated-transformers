from dataclasses import dataclass
from typing import Callable, List, Optional

from torch.nn import Module


@dataclass
class ModuleIterator:
    """
    Represents the details of a module when travesing a PyTorch module graph.

    :param module:
        Current module.
    :param name:
        Name of the module.
    :param prefix:
        Current dot path of the module. Includes the name.
    :param parent:
        Parent module. Will be `None` for the root module.
    """

    module: Module
    name: str
    prefix: str
    parent: Optional[Module]


def apply_to_module(module: Module, func: Callable[[ModuleIterator], None]):
    """
    Apply a function the module and its submodules in a breadth-first
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

        for name, child in itr.module.named_children():
            child_prefix = f"{itr.prefix}.{name}" if itr.prefix else name
            queue.append(ModuleIterator(child, name, child_prefix, itr.module))
