"""Configuration resolution and its provenance (issue #12).

The order is Defaults < file < environment < CLI, and every field remembers
which of those four it came from. Provenance is not a nicety: "why is the
context window 8192?" is otherwise answered by reading four sources and
guessing, which is how a run ends up measured against a config nobody set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.config import (
    ConfigError,
    HarnessConfig,
    Origin,
    load_config,
    resolve_config,
)
from harness.context.budget import DEFAULT_CONTEXT_WINDOW, DEFAULT_SOFT_CEILING
from harness.core import Budget, NetworkMode
from harness.models.llamacpp import DEFAULT_BASE_URL


def _write(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# -- the chain ------------------------------------------------------------


def test_defaults_are_the_values_the_subsystems_already_own() -> None:
    """Config points at the existing definitions; it does not restate them.

    A second copy of the context window would be a second owner of the same
    number, and the two would drift the first time one of them was tuned.
    """
    resolved = resolve_config()

    assert resolved.config.runtime.base_url == DEFAULT_BASE_URL
    assert resolved.config.context.context_window == DEFAULT_CONTEXT_WINDOW
    assert resolved.config.context.soft_ceiling == DEFAULT_SOFT_CEILING
    assert resolved.config.budget.max_steps == Budget().max_steps


def test_file_overrides_defaults(tmp_path: Path) -> None:
    path = _write(tmp_path / "harness.json", {"runtime": {"port": 9999}})
    resolved = resolve_config(config_file=path)

    assert resolved.config.runtime.port == 9999
    assert resolved.origin_of("runtime.port") is Origin.FILE


def test_environment_overrides_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "harness.json", {"runtime": {"port": 9999}})
    resolved = resolve_config(config_file=path, env={"HARNESS_RUNTIME_PORT": "7777"})

    assert resolved.config.runtime.port == 7777
    assert resolved.origin_of("runtime.port") is Origin.ENV


def test_cli_overrides_environment(tmp_path: Path) -> None:
    path = _write(tmp_path / "harness.json", {"runtime": {"port": 9999}})
    resolved = resolve_config(
        config_file=path,
        env={"HARNESS_RUNTIME_PORT": "7777"},
        overrides={"runtime.port": 5555},
    )

    assert resolved.config.runtime.port == 5555
    assert resolved.origin_of("runtime.port") is Origin.CLI


def test_untouched_fields_keep_the_default_origin(tmp_path: Path) -> None:
    path = _write(tmp_path / "harness.json", {"runtime": {"port": 9999}})
    resolved = resolve_config(config_file=path)

    assert resolved.origin_of("budget.max_steps") is Origin.DEFAULT
    assert resolved.origin_of("sandbox.network") is Origin.DEFAULT


def test_a_derived_value_inherits_the_origin_it_was_derived_from(
    tmp_path: Path,
) -> None:
    """base_url follows the port, so it must not read as a built-in default.

    Presenting a URL that came from a config file as one nobody chose is the
    exact question this layer exists to answer.
    """
    path = _write(tmp_path / "harness.json", {"runtime": {"port": 9999}})
    resolved = resolve_config(config_file=path)

    assert resolved.config.runtime.base_url == "http://127.0.0.1:9999"
    assert resolved.origin_of("runtime.base_url") is Origin.FILE
    assert "runtime.port" in resolved.source_of("runtime.base_url")


def test_an_explicit_url_keeps_its_own_origin(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "harness.json",
        {"runtime": {"base_url": "http://10.0.0.5:8080", "port": 9999}},
    )
    resolved = resolve_config(config_file=path)

    assert "derived" not in resolved.source_of("runtime.base_url")


def test_origin_records_which_file_a_value_came_from(tmp_path: Path) -> None:
    path = _write(tmp_path / "harness.json", {"runtime": {"port": 9999}})
    resolved = resolve_config(config_file=path)

    assert str(path) in resolved.source_of("runtime.port")


def test_origin_records_which_variable_a_value_came_from() -> None:
    resolved = resolve_config(env={"HARNESS_RUNTIME_PORT": "7777"})
    assert "HARNESS_RUNTIME_PORT" in resolved.source_of("runtime.port")


# -- typing and validation -------------------------------------------------


def test_environment_values_are_typed_not_left_as_strings() -> None:
    resolved = resolve_config(
        env={
            "HARNESS_RUNTIME_PORT": "7777",
            "HARNESS_CONTEXT_SOFT_CEILING": "4096",
            "HARNESS_SANDBOX_NETWORK": "allowed",
        }
    )

    assert resolved.config.runtime.port == 7777
    assert resolved.config.context.soft_ceiling == 4096
    assert resolved.config.sandbox.network is NetworkMode.ALLOWED


def test_a_bad_environment_value_names_the_variable() -> None:
    with pytest.raises(ConfigError) as exc:
        resolve_config(env={"HARNESS_RUNTIME_PORT": "not-a-port"})

    assert "HARNESS_RUNTIME_PORT" in str(exc.value)


def test_an_out_of_range_port_is_rejected() -> None:
    with pytest.raises(ConfigError):
        resolve_config(overrides={"runtime.port": 70000})


def test_a_soft_ceiling_above_the_window_is_clamped_and_reported() -> None:
    """Clamped, not refused — and never silently.

    Refusing looked principled until a hardware profile measuring a smaller
    context blocked every command on the weaker machine it was measured on:
    the layer meant to be advisory became a gate. The advisory limit is
    lowered to the hard one, and the adjustment is reported.
    """
    resolved = resolve_config(
        overrides={"context.context_window": 8192, "context.soft_ceiling": 16000}
    )

    assert resolved.config.context.soft_ceiling == 8192
    assert any("soft_ceiling" in warning for warning in resolved.warnings)


def test_a_profile_measuring_a_smaller_context_does_not_block_the_run(
    tmp_path: Path,
) -> None:
    """The case that turned a measurement into a requirement."""
    profile = _write(
        tmp_path / "hw.json",
        {"schema_version": 1, "recommended": {"context_length": 8192}},
    )

    resolved = resolve_config(profile_file=profile)

    assert resolved.config.context.context_window == 8192
    assert resolved.config.context.soft_ceiling <= 8192


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """A typo in a config file must not look like a setting that took effect."""
    path = _write(tmp_path / "harness.json", {"runtime": {"prot": 9999}})
    with pytest.raises(ConfigError) as exc:
        resolve_config(config_file=path)

    assert "prot" in str(exc.value)


def test_unparseable_file_is_reported_with_its_path(tmp_path: Path) -> None:
    path = tmp_path / "harness.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        resolve_config(config_file=path)

    assert str(path) in str(exc.value)


def test_a_named_config_file_that_does_not_exist_is_an_error(tmp_path: Path) -> None:
    """Asking for a file and silently getting defaults is the worst outcome."""
    with pytest.raises(ConfigError):
        resolve_config(config_file=tmp_path / "absent.json")


def test_no_config_file_at_all_is_fine() -> None:
    assert resolve_config().config.runtime.port == HarnessConfig().runtime.port


# -- hardware profile ------------------------------------------------------


def test_hardware_profile_is_read_when_present(tmp_path: Path) -> None:
    profile = _write(
        tmp_path / "hardware-profile.json",
        {"schema_version": 1, "hostname": "asus", "cpu": {"physical_cores": 4}},
    )
    resolved = resolve_config(profile_file=profile)

    assert resolved.hardware_profile is not None
    assert resolved.hardware_profile["hostname"] == "asus"


def test_a_missing_hardware_profile_does_not_block_the_run(tmp_path: Path) -> None:
    """Other machines must still work; the profile is information, not a gate."""
    resolved = resolve_config(profile_file=tmp_path / "absent.json")

    assert resolved.hardware_profile is None
    assert resolved.config.runtime.base_url == DEFAULT_BASE_URL


def test_a_corrupt_hardware_profile_is_reported_not_swallowed(tmp_path: Path) -> None:
    profile = tmp_path / "hardware-profile.json"
    profile.write_text("{broken", encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        resolve_config(profile_file=profile)

    assert str(profile) in str(exc.value)


# -- redaction -------------------------------------------------------------


def test_secrets_are_redacted_when_the_config_is_rendered() -> None:
    resolved = resolve_config(overrides={"runtime.api_key": "sk-abcdef1234567890"})

    rendered = resolved.render()

    assert "sk-abcdef1234567890" not in rendered
    assert "redacted" in rendered


def test_redacted_rendering_keeps_the_field_visible() -> None:
    """You must still be able to see *that* a key is set."""
    resolved = resolve_config(overrides={"runtime.api_key": "sk-abcdef1234567890"})
    assert "api_key" in resolved.render()


def test_as_dict_redacts_by_default() -> None:
    resolved = resolve_config(overrides={"runtime.api_key": "sk-abcdef1234567890"})
    assert "sk-abcdef1234567890" not in json.dumps(resolved.as_dict())


def test_the_unredacted_value_is_still_reachable_for_the_client() -> None:
    """The provider needs the real key; only the *output* is redacted."""
    resolved = resolve_config(overrides={"runtime.api_key": "sk-abcdef1234567890"})
    assert resolved.config.runtime.api_key == "sk-abcdef1234567890"


# -- loading ---------------------------------------------------------------


def test_load_config_finds_the_default_file(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / "harness.json", {"runtime": {"port": 4242}})
    monkeypatch.chdir(tmp_path)

    assert load_config().config.runtime.port == 4242


def test_load_config_without_a_file_uses_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HARNESS_RUNTIME_PORT", raising=False)

    assert load_config().config.runtime.port == HarnessConfig().runtime.port


# -- host, port and url stay consistent ------------------------------------


def test_setting_the_port_moves_the_url_too() -> None:
    """Otherwise the client talks to 18080 while the server binds 9999."""
    resolved = resolve_config(overrides={"runtime.port": 9999})

    assert resolved.config.runtime.base_url == "http://127.0.0.1:9999"
    assert resolved.config.runtime.port == 9999


def test_an_explicit_url_wins_over_host_and_port() -> None:
    """The attach case: the URL describes a server we did not start."""
    resolved = resolve_config(
        overrides={"runtime.base_url": "http://10.0.0.5:8080", "runtime.port": 9999}
    )

    assert resolved.config.runtime.base_url == "http://10.0.0.5:8080"


def test_the_default_url_agrees_with_the_default_host_and_port() -> None:
    runtime = resolve_config().config.runtime
    assert runtime.base_url == f"http://{runtime.host}:{runtime.port}"


# -- the hardware profile as a layer ---------------------------------------


def _profile(path: Path, recommended: dict[str, object]) -> Path:
    return _write(path, {"schema_version": 1, "recommended": recommended})


def test_profile_recommendations_beat_built_in_defaults(tmp_path: Path) -> None:
    """A measurement of this machine outranks a number chosen for no machine."""
    profile = _profile(tmp_path / "hw.json", {"threads": 4, "gpu_layers": 20})
    resolved = resolve_config(profile_file=profile)

    assert resolved.config.model.threads == 4
    assert resolved.config.model.n_gpu_layers == 20
    assert resolved.origin_of("model.threads") is Origin.PROFILE


def test_explicit_configuration_beats_the_profile(tmp_path: Path) -> None:
    profile = _profile(tmp_path / "hw.json", {"threads": 4})
    resolved = resolve_config(profile_file=profile, overrides={"model.threads": 8})

    assert resolved.config.model.threads == 8
    assert resolved.origin_of("model.threads") is Origin.CLI


def test_unmeasured_recommendations_are_left_alone(tmp_path: Path) -> None:
    """null in the profile means "the benchmark has not answered this yet".

    Treating it as a value would turn an open question into a setting.
    """
    profile = _profile(
        tmp_path / "hw.json", {"threads": 4, "gpu_layers": None, "batch_size": None}
    )
    resolved = resolve_config(profile_file=profile)

    assert resolved.config.model.n_gpu_layers is None
    assert resolved.origin_of("model.n_gpu_layers") is Origin.DEFAULT


_REAL_PROFILE = Path(__file__).resolve().parents[1] / "config" / "hardware-profile.json"


@pytest.mark.skipif(
    not _REAL_PROFILE.exists(),
    reason="hardware-profile.json is machine-specific and not versioned; "
    "run `harness doctor` to generate one",
)
def test_the_real_repository_profile_resolves() -> None:
    """A profile written by `doctor` on this machine must actually apply.

    Skipped where none has been generated: the file describes one host and is
    gitignored for that reason, so requiring it would make a fresh clone fail
    a test about configuration.
    """
    resolved = resolve_config(profile_file=_REAL_PROFILE)

    assert resolved.hardware_profile is not None
    assert resolved.config.model.threads is not None


def test_a_field_whose_name_merely_contains_token_is_not_redacted() -> None:
    """`max_output_tokens` is a budget, not a credential.

    The text scrubber matches any NAME=value where the name contains TOKEN,
    which is right for free-form logs and wrong for a structured field whose
    name we already know.
    """
    rendered = resolve_config().render()

    assert "max_output_tokens = 2048" in rendered
    assert "redacted:max_output_tokens" not in rendered


def test_an_unset_secret_reads_as_unset() -> None:
    """Showing `[redacted]` for a key nobody set invents a credential."""
    rendered = resolve_config().render()

    assert "api_key = None" in rendered


def test_a_secret_passed_as_a_server_flag_is_scrubbed() -> None:
    """The real shape: ``--api-key VALUE``, two list entries, no `=`.

    The text scrubber only matches ``name=value``, so it never saw this one.
    An earlier version of this test used an AWS-shaped key, which the scrubber
    catches by its own prefix pattern — it passed while the case that actually
    occurs, and reaches the terminal through ``config show``, leaked.
    """
    resolved = resolve_config(
        overrides={
            "model.extra_flags": ["--api-key", "sk-abcdef0123456789abcdef", "--jinja"]
        }
    )

    rendered = resolved.render()

    assert "sk-abcdef0123456789abcdef" not in rendered
    assert "--api-key" in rendered   # that a key is passed stays visible
    assert "--jinja" in rendered     # ordinary flags are untouched
    assert "sk-abcdef0123456789abcdef" not in json.dumps(resolved.as_dict())


def test_a_secret_matching_a_known_pattern_is_still_scrubbed() -> None:
    resolved = resolve_config(overrides={"model.alias": "AKIAIOSFODNN7EXAMPLE"})
    assert "AKIAIOSFODNN7EXAMPLE" not in resolved.render()


def test_the_flag_value_is_intact_for_the_server() -> None:
    """Only the *output* is redacted; llama-server still gets the real key."""
    resolved = resolve_config(
        overrides={"model.extra_flags": ["--api-key", "sk-abcdef0123456789abcdef"]}
    )
    assert resolved.config.model.extra_flags[1] == "sk-abcdef0123456789abcdef"


# -- typos must not look like settings -------------------------------------


def test_a_typo_in_the_budget_section_is_refused(tmp_path: Path) -> None:
    """`Budget` itself is lenient; configuration must not be.

    `max_stpes` was accepted and dropped, and the run then used the default
    while its author believed otherwise — the one failure mode this whole
    layer is meant to remove.
    """
    path = _write(tmp_path / "harness.json", {"budget": {"max_stpes": 999}})

    with pytest.raises(ConfigError) as exc:
        resolve_config(config_file=path)

    assert "max_stpes" in str(exc.value)


def test_an_unknown_environment_variable_is_refused() -> None:
    with pytest.raises(ConfigError) as exc:
        resolve_config(env={"HARNESS_MODEL_NGPU_LAYERS": "99"})

    assert "HARNESS_MODEL_NGPU_LAYERS" in str(exc.value)


def test_unrelated_environment_variables_are_left_alone() -> None:
    """Only the HARNESS_ namespace is ours to police."""
    resolved = resolve_config(env={"PATH": "/usr/bin", "EDITOR": "vim"})
    assert resolved.config.runtime.port == HarnessConfig().runtime.port


def test_a_scalar_where_a_section_belongs_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path / "harness.json", {"runtime": 5})

    with pytest.raises(ConfigError) as exc:
        resolve_config(config_file=path, env={"HARNESS_RUNTIME_PORT": "9000"})

    assert "runtime" in str(exc.value)


# -- one window, not two ---------------------------------------------------


def test_the_budget_window_follows_the_window_the_server_is_started_with() -> None:
    """`n_ctx` launches the server; `context_window` is what the budget fills.

    Two numbers for one window let the budget grow a prompt past what the
    server accepts, and the overflow then arrives as a runtime error instead
    of a configuration one.
    """
    resolved = resolve_config(overrides={"model.n_ctx": 8192})

    assert resolved.config.context.context_window == 8192
    assert resolved.config.context.soft_ceiling <= 8192
    assert any("context_window" in warning for warning in resolved.warnings)


def test_without_n_ctx_the_configured_window_stands() -> None:
    resolved = resolve_config(overrides={"context.context_window": 32768})

    assert resolved.config.context.context_window == 32768
    assert resolved.warnings == []


# -- error messages name their source --------------------------------------


def test_a_whole_section_validator_still_names_the_source(tmp_path: Path) -> None:
    """The hardest errors to trace are the ones a model validator raises.

    Pydantic reports the section it guards, not the field that broke it, so an
    exact provenance lookup finds nothing precisely where it is needed most.
    """
    path = _write(tmp_path / "harness.json", {"runtime": {"port": 70000}})

    with pytest.raises(ConfigError) as exc:
        resolve_config(config_file=path)

    assert str(path) in str(exc.value)
