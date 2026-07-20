from inference_perf.datagen.synthetic_themes import load_theme, Theme, GENERIC_THEME, DEFAULT_SYSTEM_PROMPT  # noqa: F401


def test_load_bundled_theme():
    t = load_theme("db2_latency_incident")
    assert isinstance(t, Theme)
    assert t.name == "db2_latency_incident"
    assert t.objective_template  # non-empty
    assert len(t.verbs) >= 3
    assert t.tool_names  # at least one tool


def test_generic_theme_is_valid():
    assert isinstance(GENERIC_THEME, Theme)
    assert GENERIC_THEME.objective_template


def test_load_unknown_theme_raises():
    import pytest
    with pytest.raises(ValueError):
        load_theme("nonexistent_theme_xyz")
