from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

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
        Current dot path of the module. Does not include the name.
    :param full_path:
        Current dot path of the module. Includes the name.
    :param parent:
        Parent module. Will be `None` for the root module.
    """

    module: Module
    name: str
    prefix: str
    full_path: str
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
    queue: List[ModuleIterator] = [ModuleIterator(module, "", "", "", None)]

    while queue:
        itr = queue.pop(0)
        func(itr)

        for name, child in itr.module.named_children():
            child_path = f"{itr.full_path}.{name}" if itr.full_path else name
            queue.append(
                ModuleIterator(
                    child,
                    name,
                    prefix=itr.full_path,
                    full_path=child_path,
                    parent=itr.module,
                )
            )


def split_full_name(full_name: str) -> Tuple[str, str]:
    """
    Split a fully-qualified name into its prefix and final
    component.

    :param full_name:
        Fully-qualified name.
    :returns:
        The final component and the prefix.
    """
    splits = full_name.split(".")
    if len(splits) < 2:
        raise ValueError(
            f"Fully-qualified name '{full_name}' must contain at least one dot"
        )
    name = splits[-1]
    prefix = ".".join(splits[:-1])
    return name, prefix
