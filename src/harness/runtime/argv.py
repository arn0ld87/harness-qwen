"""Build the ``llama-server`` command line from typed configuration.

Kept apart from the supervisor so the command can be asserted without
starting anything, and printed by ``doctor`` without a process. What goes on
this line decides throughput on this hardware, so it is worth being able to
read it back exactly as it will be run.
"""

from __future__ import annotations

from harness.config.schema import HarnessConfig

# Typed field -> flag. Order is fixed so two runs with the same config produce
# the same line, which is what makes a benchmark comparable to an earlier one.
_MODEL_FLAGS: tuple[tuple[str, str], ...] = (
    ("path", "--model"),
    ("alias", "--alias"),
    ("n_ctx", "--ctx-size"),
    ("n_gpu_layers", "--n-gpu-layers"),
    ("threads", "--threads"),
    ("batch_size", "--batch-size"),
    ("cache_type_k", "--cache-type-k"),
    ("cache_type_v", "--cache-type-v"),
)


def build_argv(config: HarnessConfig) -> list[str]:
    """Return the full command line, typed flags first, extras last.

    An unset field contributes nothing at all rather than an empty value:
    absent means "whatever the server defaults to", and passing ``--threads``
    with nothing after it would be a different, worse statement.

    ``extra_flags`` goes last so it can override anything above it — an escape
    hatch for a flag this config does not model yet, which is preferable to
    growing a field for every flag llama.cpp has.
    """
    if config.runtime.server_binary is None:
        raise ValueError(
            "cannot build a start command without runtime.server_binary; "
            "set it, or use attach mode for a server that is already running"
        )

    argv: list[str] = [str(config.runtime.server_binary)]
    for field, flag in _MODEL_FLAGS:
        value = getattr(config.model, field)
        if value is not None:
            argv += [flag, str(value)]

    argv += ["--host", config.runtime.host, "--port", str(config.runtime.port)]
    argv += [str(flag) for flag in config.model.extra_flags]
    return argv


__all__ = ["build_argv"]
