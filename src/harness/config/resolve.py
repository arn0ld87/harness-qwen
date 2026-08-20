"""Resolve configuration from four sources and remember which one won.

Defaults < config file < environment < CLI flags. The order is the easy part;
the part worth writing down is the provenance. "Why is the context window
8192?" otherwise means reading four sources and guessing, and the run that
gets measured against a setting nobody chose is the expensive kind of bug on
a machine where one cold prompt costs 25 s.
"""

from __future__ import annotations

import enum
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from harness.config.schema import (
    SECRET_FIELDS,
    SECRET_FLAG_SUFFIXES,
    HarnessConfig,
)
from harness.telemetry.redact import redact

DEFAULT_CONFIG_NAMES: tuple[str, ...] = ("harness.json", ".harness.json")
DEFAULT_PROFILE_PATH = Path("config/hardware-profile.json")
ENV_PREFIX = "HARNESS_"
REDACTED = "[redacted:api_key]"


class ConfigError(RuntimeError):
    """Configuration could not be read, parsed or validated.

    One exception type for all four sources: the caller wants to print what is
    wrong and stop, and which layer produced it is already in the message.
    """


class Origin(enum.StrEnum):
    """Which layer a field's effective value came from.

    Listed in increasing priority. ``PROFILE`` sits between the built-in
    defaults and anything a human wrote down: it is a measurement of *this*
    machine, which beats a number chosen for no machine in particular, and
    loses to someone who states what they want.
    """

    DEFAULT = "default"
    PROFILE = "profile"
    FILE = "file"
    ENV = "env"
    CLI = "cli"


class ResolvedConfig:
    """An effective configuration plus where each field came from."""

    def __init__(
        self,
        config: HarnessConfig,
        origins: Mapping[str, tuple[Origin, str]],
        hardware_profile: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self._origins = dict(origins)
        self.hardware_profile = hardware_profile
        self.warnings: list[str] = list(config.context.adjustments)
        """Values corrected during validation. Reported rather than raised, so
        a measurement of a smaller machine adjusts the run instead of blocking
        it — but never silently."""

    def origin_of(self, path: str) -> Origin:
        """Layer that set ``path`` (dotted), or DEFAULT if nothing did."""
        return self._origins.get(path, (Origin.DEFAULT, "built-in default"))[0]

    def source_of(self, path: str) -> str:
        """Human-readable source: the file, the variable, or the flag."""
        return self._origins.get(path, (Origin.DEFAULT, "built-in default"))[1]

    def as_dict(self, *, reveal_secrets: bool = False) -> dict[str, Any]:
        """The effective config as plain data, redacted unless asked otherwise.

        Redaction is the default because every caller that prints, logs or
        serialises this would otherwise have to remember to ask for it, and
        the one that forgets writes a key to disk.
        """
        data = json.loads(self.config.model_dump_json())
        if not reveal_secrets:
            _redact_in_place(data)
        return data

    def provenance(self) -> dict[str, tuple[Origin, str]]:
        """Every non-default field, with the layer and source that set it."""
        return dict(self._origins)

    def render(self) -> str:
        """Redacted, human-readable rendering used by ``config show``.

        Two passes with different jobs. Declared secret fields are replaced by
        name in :meth:`as_dict`, which is exact. The text scrubber then runs
        over each *value* — never the whole line — to catch a credential that
        ended up somewhere nobody declared, such as ``extra_flags``. Scrubbing
        the line instead would redact ``max_output_tokens`` for containing the
        word TOKEN, and print a ``[redacted]`` for an ``api_key`` nobody set.
        """
        lines: list[str] = []
        for path, value in sorted(_flatten(self.as_dict()).items()):
            origin = self.origin_of(path)
            marker = "" if origin is Origin.DEFAULT else f"  [{origin}]"
            lines.append(f"{path} = {redact(str(value))}{marker}")
        return "\n".join(lines)


def resolve_config(
    *,
    config_file: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    profile_file: Path | None = None,
) -> ResolvedConfig:
    """Merge the four layers into one validated configuration.

    ``config_file`` and ``profile_file`` are read only if given; a named file
    that does not exist is an error, because falling back to defaults after
    being told which file to use is the failure nobody notices.
    """
    merged: dict[str, Any] = {}
    origins: dict[str, tuple[Origin, str]] = {}
    profile = _read_profile(profile_file)

    # Applied lowest-priority first, each layer overwriting the last. The
    # environment is handled separately only because it carries a different
    # source label per field — the variable name, not one file path.
    if profile is not None:
        values = _from_profile(profile)
        _merge(merged, values)
        for path in _flatten(values):
            origins[path] = (Origin.PROFILE, f"{profile_file} (recommended)")

    if config_file is not None:
        values = _read_file(config_file)
        _merge(merged, values)
        for path in _flatten(values):
            origins[path] = (Origin.FILE, str(config_file))

    if env is not None:
        for path, (value, variable) in _from_env(env).items():
            _assign(merged, path, value)
            origins[path] = (Origin.ENV, variable)

    if overrides:
        values = _nest(dict(overrides))
        _merge(merged, values)
        for path in _flatten(values):
            origins[path] = (Origin.CLI, "command line")

    try:
        config = HarnessConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(_explain(exc, origins)) from exc

    _inherit_derived_origins(config, merged, origins)
    return ResolvedConfig(config, origins, profile)


def load_config(
    *,
    config_file: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    directory: Path | None = None,
) -> ResolvedConfig:
    """Resolve using the process environment and the conventional file names."""
    base = directory or Path.cwd()
    if config_file is None:
        config_file = next(
            (base / name for name in DEFAULT_CONFIG_NAMES if (base / name).exists()),
            None,
        )
    profile = base / DEFAULT_PROFILE_PATH
    return resolve_config(
        config_file=config_file,
        env=os.environ,
        overrides=overrides,
        profile_file=profile if profile.exists() else None,
    )


# -- sources ---------------------------------------------------------------


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain an object")
    return data


def _read_profile(path: Path | None) -> dict[str, Any] | None:
    """Read the hardware profile if it is there.

    Absent is fine and stays fine: the profile describes the machine this was
    tuned on, and refusing to run elsewhere would turn a measurement into a
    requirement.
    """
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read hardware profile {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"hardware profile {path} must contain an object")
    return data


PROFILE_FIELDS: dict[str, str] = {
    "threads": "model.threads",
    "gpu_layers": "model.n_gpu_layers",
    "batch_size": "model.batch_size",
    "context_length": "context.context_window",
    "max_output_tokens": "budget.max_output_tokens",
}
"""``recommended`` keys that map onto configuration, and where they land."""


def _from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Turn measured recommendations into configuration values.

    ``null`` is skipped rather than applied: in this profile it means "the
    benchmark has not answered this yet" (see the ``rationale`` block beside
    it), and turning an open question into a setting is how an unmeasured
    number ends up looking like a decision.
    """
    recommended = profile.get("recommended")
    if not isinstance(recommended, Mapping):
        return {}
    values: dict[str, Any] = {}
    for key, path in PROFILE_FIELDS.items():
        value = recommended.get(key)
        if value is not None:
            _assign(values, path, value)
    return values


def _from_env(env: Mapping[str, str]) -> dict[str, tuple[Any, str]]:
    """Map ``HARNESS_SECTION_FIELD`` onto dotted paths, keeping the variable.

    Values stay strings here. Pydantic does the coercion, so the types live in
    exactly one place and an unparseable value fails with the field's own
    rules rather than a second, subtly different parser.
    """
    fields = _known_paths(HarnessConfig)
    found: dict[str, tuple[Any, str]] = {}
    unknown: list[str] = []
    for variable, raw in env.items():
        if not variable.startswith(ENV_PREFIX):
            continue
        tail = variable[len(ENV_PREFIX) :].lower()
        match = next((path for path in fields if _env_name(path) == tail), None)
        if match is None:
            unknown.append(variable)
            continue
        found[match] = (raw, variable)

    if unknown:
        # The same failure class ``extra="forbid"`` catches in files: a
        # misspelled variable that is skipped looks exactly like one that
        # applied, and the run it fails to configure is the one nobody debugs.
        names = ", ".join(sorted(unknown))
        raise ConfigError(
            f"unknown configuration variable(s): {names}. Valid names are "
            f"{ENV_PREFIX}<SECTION>_<FIELD>, e.g. "
            f"{ENV_PREFIX}{_env_name('runtime.port').upper()}."
        )
    return found


# -- dotted-path plumbing --------------------------------------------------


def _env_name(path: str) -> str:
    return path.replace(".", "_")


def _known_paths(model: type[Any], prefix: str = "") -> list[str]:
    """Every leaf path in the model, so env lookup never guesses."""
    paths: list[str] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        nested = getattr(annotation, "model_fields", None)
        full = f"{prefix}{name}"
        if nested is not None:
            paths.extend(_known_paths(annotation, prefix=f"{full}."))
        else:
            paths.append(full)
    return paths


def _nest(flat: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path, value in flat.items():
        _assign(out, path, value)
    return out


def _assign(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    walked: list[str] = []
    for part in parts[:-1]:
        walked.append(part)
        nxt = cursor.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ConfigError(
                f"cannot set {path}: {'.'.join(walked)} is "
                f"{type(nxt).__name__}, not a section"
            )
        cursor = nxt
    cursor[parts[-1]] = value


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def _redact_in_place(data: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            _redact_in_place(value)
        elif key in SECRET_FIELDS and value is not None:
            data[key] = REDACTED
        elif isinstance(value, list):
            data[key] = _redact_flags(value)


def _redact_flags(argv: list[Any]) -> list[Any]:
    """Hide the value that follows a credential-carrying flag.

    Both spellings: ``--api-key VALUE`` (the one llama-server actually uses,
    and the one a pattern matching ``name=value`` never saw) and
    ``--api-key=VALUE``. The flag itself stays visible — that a key is being
    passed is exactly what someone reading ``config show`` needs to know.
    """
    out: list[Any] = []
    hide_next = False
    for item in argv:
        if hide_next:
            out.append("[redacted:flag_value]")
            hide_next = False
            continue
        if isinstance(item, str) and item.startswith("-"):
            name, sep, _value = item.partition("=")
            if _is_secret_flag(name):
                out.append(f"{name}=[redacted:flag_value]" if sep else item)
                hide_next = not sep
                continue
        out.append(item)
    return out


def _is_secret_flag(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in SECRET_FLAG_SUFFIXES)


def _explain(exc: ValidationError, origins: Mapping[str, tuple[Origin, str]]) -> str:
    """Name the source that produced each rejected value.

    "port must be <= 65535" is half an answer; the other half is which file or
    variable to go and fix.
    """
    lines: list[str] = []
    for error in exc.errors():
        path = ".".join(str(part) for part in error["loc"])
        where = origins.get(path) or _origin_under(path, origins)
        suffix = f" (from {where[1]})" if where else ""
        lines.append(f"{path}: {error['msg']}{suffix}")
    return "invalid configuration:\n  " + "\n  ".join(lines)


def _origin_under(
    path: str, origins: Mapping[str, tuple[Origin, str]]
) -> tuple[Origin, str] | None:
    """Source of any field inside ``path``.

    A whole-model validator reports the section it guards, not the field that
    broke it, so an exact lookup finds nothing — on exactly the error class
    that is hardest to trace back to its source.
    """
    prefix = f"{path}."
    inner = sorted(key for key in origins if key.startswith(prefix))
    return origins[inner[0]] if inner else None


def _inherit_derived_origins(
    config: HarnessConfig,
    merged: Mapping[str, Any],
    origins: dict[str, tuple[Origin, str]],
) -> None:
    """Attribute a derived value to whatever it was derived from.

    ``base_url`` follows host and port unless it was set explicitly. Leaving
    it marked as a built-in default would present a URL that came from a
    config file as one nobody chose — the exact question provenance exists to
    answer.
    """
    if "runtime.base_url" in _flatten(merged):
        return
    for source in ("runtime.port", "runtime.host"):
        if source in origins:
            layer, where = origins[source]
            origins["runtime.base_url"] = (layer, f"derived from {source} ({where})")
            return


__all__ = [
    "ConfigError",
    "Origin",
    "ResolvedConfig",
    "load_config",
    "resolve_config",
]
