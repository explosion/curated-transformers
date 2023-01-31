from typing import Optional, Tuple
from thinc.api import Schedule


# TODO: most schedules, except the transformer_discriminative should
# be moved to Thinc at some point.


def warmup_exponential_decay(
    initial_rate: float,
    decay_rate: float,
    decay_steps: int,
    staircase: bool,
    warmup_steps: int,
) -> Schedule:
    return Schedule(
        "warmup_exponential_decay",
        _warmup_exponential_decay_schedule,
        attrs={
            "initial_rate": initial_rate,
            "decay_rate": decay_rate,
            "decay_steps": decay_steps,
            "staircase": staircase,
            "warmup_steps": warmup_steps,
        },
    )


def _warmup_exponential_decay_schedule(
    schedule: Schedule, step: int, **kwargs
) -> float:
    initial_rate: float = schedule.attrs["initial_rate"]
    decay_rate: float = schedule.attrs["decay_rate"]
    decay_steps: int = schedule.attrs["decay_steps"]
    staircase: bool = schedule.attrs["staircase"]
    warmup_steps: int = schedule.attrs["warmup_steps"]

    if step < warmup_steps:
        return (initial_rate / warmup_steps) * step

    step = step - warmup_steps
    exponent = step // decay_steps if staircase else step / decay_steps

    return initial_rate * (decay_rate**exponent)


def transformer_discriminative(
    default_schedule: Schedule,
    transformer_schedule: Schedule,
) -> Schedule:
    return Schedule(
        "transfomer",
        _transformer_discriminative_schedule,
        attrs={
            "default_schedule": default_schedule,
            "transformer_schedule": transformer_schedule,
        },
    )


def _transformer_discriminative_schedule(
    schedule: Schedule, step: int, *, key: Optional[Tuple[int, str]] = None, **kwargs
) -> float:
    default_schedule: Schedule = schedule.attrs["default_schedule"]
    transformer_schedule: Schedule = schedule.attrs["transformer_schedule"]

    if key is None:
        return default_schedule(step=step, key=key, **kwargs)

    key_str = key[1]
    if "layers." in key_str or "embeddings." in key_str:
        return transformer_schedule(step=step, key=key, **kwargs)

    return default_schedule(step=step, key=key, **kwargs)
