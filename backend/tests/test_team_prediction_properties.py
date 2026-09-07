"""
Property-based tests for the team-prediction feature.
Uses the Hypothesis library (already in requirements.txt).

Run from the backend/ directory:
    pytest tests/test_team_prediction_properties.py

Tasks implemented here: 2.1, 2.2, 2.3, 10
"""
import os
import sys

# Ensure backend/ is on the path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services import ensamblar_vector_prediccion

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

# All stat types recognised by preparar_datos_para_modelo (used when building
# synthetic stats entries for property tests).
STAT_FIELDS = [
    "Ball Possession",
    "Total Shots",
    "Fouls",
    "Yellow Cards",
    "Goals",
]


def _make_stats_entry(team_label: str, stat_overrides: dict) -> dict:
    """
    Build a single stats entry in the shape returned by
    obtener_estadisticas_partido, with ``team.name`` set to *team_label*.
    *stat_overrides* maps stat-type strings to their values.
    """
    statistics = [
        {"type": stat_type, "value": stat_overrides.get(stat_type, 0)}
        for stat_type in STAT_FIELDS
    ]
    return {
        "team": {"name": team_label},
        "statistics": statistics,
    }


# Hypothesis strategy that produces a single numeric stat value
numeric_stat = st.one_of(
    st.integers(min_value=0, max_value=100),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)

# Strategy that generates a dict of stat-type → value for one team
stat_overrides_strategy = st.fixed_dictionaries(
    {field: numeric_stat for field in STAT_FIELDS}
)


# ---------------------------------------------------------------------------
# Property 1: Stats mapping produces a complete, ordered 9-value numeric vector
# Feature: team-prediction, Property 1
# Validates: Requirements 2.5, 3.1
# ---------------------------------------------------------------------------

@given(
    local_stats=stat_overrides_strategy,
    visitante_stats=stat_overrides_strategy,
)
@settings(max_examples=100)
def test_property_1_stats_mapping_complete_vector(local_stats, visitante_stats):
    """
    **Validates: Requirements 2.5, 3.1**

    Property 1: Stats mapping produces a complete, ordered 9-value numeric vector.

    For any raw statistics response (or mocked equivalent), ensamblar_vector_prediccion
    must return a list of exactly 9 elements, all numeric (int or float).
    """
    stats_local = [_make_stats_entry("local", local_stats)]
    stats_visitante = [_make_stats_entry("visitante", visitante_stats)]

    vector = ensamblar_vector_prediccion(stats_local, stats_visitante)

    # Must be a list (or list-compatible sequence)
    assert isinstance(vector, list), f"Expected list, got {type(vector)}"

    # Must have exactly 9 elements
    assert len(vector) == 9, f"Expected 9 elements, got {len(vector)}: {vector}"

    # Every element must be numeric
    for i, val in enumerate(vector):
        assert isinstance(val, (int, float)), (
            f"Element {i} is not numeric: {val!r} (type {type(val).__name__})"
        )


# ---------------------------------------------------------------------------
# Property 2: Goal difference is always local goals minus visitante goals
# Feature: team-prediction, Property 2
# Validates: Requirements 2.6
# ---------------------------------------------------------------------------

@given(
    goles_local=st.integers(min_value=0, max_value=20),
    goles_visitante=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100)
def test_property_2_goal_difference(goles_local, goles_visitante):
    """
    **Validates: Requirements 2.6**

    Property 2: Goal difference is always local goals minus visitante goals.

    The last element of the vector (index 8, diferencia_goles) must equal
    goles_local - goles_visitante for any integer inputs.
    """
    # Build entries with explicit goals values
    local_overrides = {f: 0 for f in STAT_FIELDS}
    local_overrides["Goals"] = goles_local

    visitante_overrides = {f: 0 for f in STAT_FIELDS}
    visitante_overrides["Goals"] = goles_visitante

    stats_local = [_make_stats_entry("local", local_overrides)]
    stats_visitante = [_make_stats_entry("visitante", visitante_overrides)]

    vector = ensamblar_vector_prediccion(stats_local, stats_visitante)

    diferencia_goles = vector[8]
    expected = goles_local - goles_visitante

    assert diferencia_goles == expected, (
        f"diferencia_goles={diferencia_goles} != "
        f"goles_local({goles_local}) - goles_visitante({goles_visitante}) = {expected}"
    )


# ---------------------------------------------------------------------------
# Property 3: Missing or null stat fields default to zero
# Feature: team-prediction, Property 3
# Validates: Requirements 2.7
# ---------------------------------------------------------------------------

@given(
    missing_local=st.frozensets(st.sampled_from(STAT_FIELDS)),
    missing_visitante=st.frozensets(st.sampled_from(STAT_FIELDS)),
)
@settings(max_examples=100)
def test_property_3_missing_stats_default_zero(missing_local, missing_visitante):
    """
    **Validates: Requirements 2.7**

    Property 3: Missing or null stat fields default to zero.

    For any combination of absent or null stat fields, the assembled vector must
    substitute 0 and still have exactly 9 numeric elements.
    """
    def _make_entry_with_missing(label: str, missing_fields: frozenset) -> dict:
        """Build a stats entry where *missing_fields* are either absent or null."""
        statistics = []
        for field in STAT_FIELDS:
            if field in missing_fields:
                # Randomly use null value (None maps to 0 in preparar_datos_para_modelo)
                statistics.append({"type": field, "value": None})
            else:
                statistics.append({"type": field, "value": 5})
        return {"team": {"name": label}, "statistics": statistics}

    stats_local = [_make_entry_with_missing("local", missing_local)]
    stats_visitante = [_make_entry_with_missing("visitante", missing_visitante)]

    vector = ensamblar_vector_prediccion(stats_local, stats_visitante)

    # Vector must still have 9 elements
    assert len(vector) == 9, (
        f"Expected 9 elements with missing fields {missing_local | missing_visitante}, "
        f"got {len(vector)}: {vector}"
    )

    # Every element must be numeric (None must have been replaced by 0)
    for i, val in enumerate(vector):
        assert isinstance(val, (int, float)), (
            f"Element {i} is not numeric after null substitution: {val!r}"
        )

    # Fields that were null or absent must contribute 0 to their respective slots.
    # We verify by checking: if ALL non-goal fields are missing for both teams,
    # the first 8 non-goal elements should all be 0.
    all_non_goal_missing_local = {f for f in missing_local if f != "Goals"}
    all_non_goal_missing_visitante = {f for f in missing_visitante if f != "Goals"}

    slot_map = [
        ("Ball Possession", 0, "local"),
        ("Ball Possession", 1, "visitante"),
        ("Total Shots",     2, "local"),
        ("Total Shots",     3, "visitante"),
        ("Fouls",           4, "local"),
        ("Fouls",           5, "visitante"),
        ("Yellow Cards",    6, "local"),
        ("Yellow Cards",    7, "visitante"),
    ]
    for field, slot_idx, team in slot_map:
        missing_set = all_non_goal_missing_local if team == "local" else all_non_goal_missing_visitante
        if field in missing_set:
            assert vector[slot_idx] == 0, (
                f"Slot {slot_idx} ({field}, {team}) should be 0 when field is missing, "
                f"got {vector[slot_idx]}"
            )


# ---------------------------------------------------------------------------
# Property 4: The ML model always returns one of three valid outcome strings
# Feature: team-prediction, Property 4
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------

from ml_model import cargar_modelo, predecir as ml_predecir


@given(st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    min_size=9,
    max_size=9,
))
@settings(max_examples=100)
def test_property_4_prediction_outcome_is_valid(entrada):
    """
    **Validates: Requirements 3.2**

    Property 4: The ML model always returns one of three valid outcome strings.

    For any list of exactly 9 numeric values passed to predecir(modelo, scaler, entrada),
    the return value must be one of "Gana equipo local", "Gana equipo visitante", or "Empate".
    """
    valid_outcomes = {"Gana equipo local", "Gana equipo visitante", "Empate"}

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(backend_dir, "modelo.pkl")
    modelo, scaler = cargar_modelo(model_path)

    resultado = ml_predecir(modelo, scaler, entrada)

    assert resultado in valid_outcomes, (
        f"Expected one of {valid_outcomes}, but got: {resultado!r}"
    )


# ---------------------------------------------------------------------------
# Property 5: The result page always includes both team names
# Feature: team-prediction, Property 5
# Validates: Requirements 3.5
# ---------------------------------------------------------------------------

import sys
import os

# Ensure the backend directory is importable as the Flask app root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app
from markupsafe import escape as jinja_escape


@given(
    equipo_local=st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters=" .-_'",
        ),
        min_size=1,
        max_size=50,
    ),
    equipo_visitante=st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters=" .-_'",
        ),
        min_size=1,
        max_size=50,
    ),
)
@settings(max_examples=100)
def test_property_5_result_page_includes_team_names(equipo_local, equipo_visitante):
    """
    **Validates: Requirements 3.5**

    Property 5: The result page always includes both team names.

    For any pair of non-empty team name strings, the HTML rendered by resultado.html
    must contain both team name strings in the response body.

    Note: Jinja2 auto-escapes special characters (e.g. ' → &#39;), so the assertion
    compares against the HTML-escaped form of each team name, which is exactly what
    a browser would receive and display.
    """
    assume(equipo_local.strip() != "")
    assume(equipo_visitante.strip() != "")

    with flask_app.test_request_context():
        from flask import render_template
        html = render_template(
            "resultado.html",
            resultado="Empate",
            equipo_local=equipo_local,
            equipo_visitante=equipo_visitante,
        )

    # Jinja2 HTML-escapes values in {{ ... }} blocks via markupsafe.escape
    # (e.g. ' → &#39;, & → &amp;).  We compare against the escaped form so
    # the assertion matches what's actually present in the rendered HTML.
    escaped_local = str(jinja_escape(equipo_local))
    escaped_visitante = str(jinja_escape(equipo_visitante))

    assert escaped_local in html, (
        f"equipo_local {equipo_local!r} (escaped: {escaped_local!r}) not found in rendered HTML"
    )
    assert escaped_visitante in html, (
        f"equipo_visitante {equipo_visitante!r} (escaped: {escaped_visitante!r}) not found in rendered HTML"
    )


# ---------------------------------------------------------------------------
# Property 6: Identical team IDs are always rejected with HTTP 400
# Feature: team-prediction, Property 6
# Validates: Requirements 4.5
# ---------------------------------------------------------------------------

import app as app_module


@given(team_id=st.integers(min_value=1, max_value=999_999))
@settings(max_examples=100)
def test_property_6_same_team_rejected(team_id):
    """
    **Validates: Requirements 4.5**

    Property 6: Identical team IDs are always rejected with HTTP 400.

    For any team ID value ``t``, a POST request to ``/predecir/equipos`` with
    ``local_team_id=t`` and ``visitante_team_id=t`` SHALL return HTTP 400.
    """
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        response = client.post(
            "/predecir/equipos",
            data={
                "local_team_id": str(team_id),
                "visitante_team_id": str(team_id),
            },
        )
    assert response.status_code == 400, (
        f"Expected HTTP 400 for team_id={team_id}, got {response.status_code}"
    )
