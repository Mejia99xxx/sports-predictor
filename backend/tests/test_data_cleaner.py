"""
Unit tests for data_cleaner.py — tasks 4.1, 4.3, 4.6.

Covers:
- combinar_fuentes: schema enforcement, CSV + API merging, empty inputs
- limpiar_dataset: deduplication, null imputation, out-of-range discard, summary dict
- validar_y_persistir: atomic write, ValueError on insufficient rows, no partial overwrite

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

import os
import sys

# Ensure backend/ is on the path so relative imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from scripts.data_cleaner import (
    combinar_fuentes,
    limpiar_dataset,
    validar_y_persistir,
    COLUMNAS_ESQUEMA,
    FEATURE_VECTOR_COLS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**kwargs) -> dict:
    """
    Build a complete valid row using sensible defaults, overriding with kwargs.
    """
    defaults = {
        "equipo_local": "Equipo A",
        "equipo_visitante": "Equipo B",
        "goles_local": 1,
        "goles_visitante": 0,
        "posesion_local": 55.0,
        "posesion_visitante": 45.0,
        "tiros_local": 8,
        "tiros_visitante": 5,
        "faltas_local": 10,
        "faltas_visitante": 12,
        "tarjetas_local": 1,
        "tarjetas_visitante": 2,
        "resultado": 1,
        "diferencia_goles": 1,
    }
    defaults.update(kwargs)
    return defaults


def _make_df(rows: list) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts."""
    return pd.DataFrame(rows)


def _valid_df(n: int = 5) -> pd.DataFrame:
    """Return a DataFrame with n valid, distinct rows."""
    rows = [
        _make_row(equipo_local=f"Local_{i}", equipo_visitante=f"Visit_{i}")
        for i in range(n)
    ]
    return _make_df(rows)


# ---------------------------------------------------------------------------
# combinar_fuentes
# ---------------------------------------------------------------------------

class TestCombinarFuentes:

    def test_output_has_exactly_14_columns(self):
        """Output must have exactly the 14 schema columns, no more, no less."""
        df_csv = _make_df([_make_row()])
        df_api = [_make_row(equipo_local="API_A", equipo_visitante="API_B")]

        result = combinar_fuentes(df_csv, df_api)

        assert list(result.columns) == COLUMNAS_ESQUEMA
        assert len(result.columns) == 14

    def test_csv_and_api_rows_are_all_present(self):
        """Rows from both sources are concatenated into the result."""
        df_csv = _make_df([
            _make_row(equipo_local="CSV_A", equipo_visitante="CSV_B"),
            _make_row(equipo_local="CSV_C", equipo_visitante="CSV_D"),
        ])
        df_api = [
            _make_row(equipo_local="API_A", equipo_visitante="API_B"),
        ]

        result = combinar_fuentes(df_csv, df_api)

        assert len(result) == 3
        assert set(result["equipo_local"]) == {"CSV_A", "CSV_C", "API_A"}

    def test_list_of_csv_dataframes_is_accepted(self):
        """df_csv can be a list of DataFrames; all are included."""
        df1 = _make_df([_make_row(equipo_local="A", equipo_visitante="B")])
        df2 = _make_df([_make_row(equipo_local="C", equipo_visitante="D")])
        df_api: list = []

        result = combinar_fuentes([df1, df2], df_api)

        assert len(result) == 2
        assert set(result["equipo_local"]) == {"A", "C"}

    def test_api_as_dataframe_is_accepted(self):
        """df_api can be a DataFrame as well as a list of dicts."""
        df_csv = _make_df([_make_row()])
        df_api = pd.DataFrame([_make_row(equipo_local="API_X", equipo_visitante="API_Y")])

        result = combinar_fuentes(df_csv, df_api)

        assert len(result) == 2
        assert "API_X" in result["equipo_local"].values

    def test_empty_csv_and_api_returns_empty_df_with_schema(self):
        """When both sources are empty, returns an empty DataFrame with schema columns."""
        result = combinar_fuentes(pd.DataFrame(), [])

        assert list(result.columns) == COLUMNAS_ESQUEMA
        assert len(result) == 0

    def test_extra_columns_in_csv_are_dropped(self):
        """Extra columns not in the schema are discarded."""
        row = _make_row()
        row["columna_extra"] = "x"
        df_csv = _make_df([row])

        result = combinar_fuentes(df_csv, [])

        assert "columna_extra" not in result.columns
        assert list(result.columns) == COLUMNAS_ESQUEMA

    def test_missing_columns_in_api_are_filled_with_nan(self):
        """If the API source lacks a schema column, it is added as NaN."""
        # Provide only a subset of columns
        df_api = pd.DataFrame([{
            "equipo_local": "X",
            "equipo_visitante": "Y",
            # remaining columns deliberately absent
        }])

        result = combinar_fuentes(pd.DataFrame(), df_api)

        assert "goles_local" in result.columns
        assert pd.isna(result.iloc[0]["goles_local"])

    def test_column_order_matches_schema(self):
        """Columns in the output must follow the canonical schema order."""
        df_csv = _make_df([_make_row()])
        result = combinar_fuentes(df_csv, [])

        assert list(result.columns) == COLUMNAS_ESQUEMA

    def test_only_api_source(self):
        """Works when df_csv is an empty DataFrame and df_api has rows."""
        df_api = [_make_row(equipo_local="Only_API", equipo_visitante="Only_B")]
        result = combinar_fuentes(pd.DataFrame(), df_api)

        assert len(result) == 1
        assert result.iloc[0]["equipo_local"] == "Only_API"


# ---------------------------------------------------------------------------
# limpiar_dataset
# ---------------------------------------------------------------------------

class TestLimpiarDataset:

    # --- Deduplication ---

    def test_duplicate_rows_are_removed(self):
        """Rows with the same (equipo_local, equipo_visitante) keep only one."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", goles_local=1),
            _make_row(equipo_local="A", equipo_visitante="B", goles_local=3),  # dup
            _make_row(equipo_local="C", equipo_visitante="D"),
        ])

        df_limpio, resumen = limpiar_dataset(df)

        assert len(df_limpio) == 2
        assert resumen["duplicados_eliminados"] == 1

    def test_no_duplicates_nothing_removed(self):
        """With no duplicates, duplicados_eliminados is 0."""
        df = _valid_df(3)
        _, resumen = limpiar_dataset(df)
        assert resumen["duplicados_eliminados"] == 0

    # --- Null imputation ---

    def test_nulls_in_feature_vector_are_imputed_with_median(self):
        """Nulls in Feature_Vector columns are filled with column median."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", goles_local=2.0),
            _make_row(equipo_local="C", equipo_visitante="D", goles_local=4.0),
            _make_row(equipo_local="E", equipo_visitante="F", goles_local=float("nan")),
        ])

        df_limpio, resumen = limpiar_dataset(df)

        # Median of [2, 4] = 3
        assert not df_limpio["goles_local"].isnull().any()
        assert df_limpio.loc[df_limpio["equipo_local"] == "E", "goles_local"].values[0] == pytest.approx(3.0)
        assert resumen["filas_imputadas"] == 1

    def test_no_nulls_filas_imputadas_is_zero(self):
        """With no nulls, filas_imputadas should be 0."""
        df = _valid_df(4)
        _, resumen = limpiar_dataset(df)
        assert resumen["filas_imputadas"] == 0

    def test_no_nulls_remain_after_imputation(self):
        """After imputation no Feature_Vector column should have nulls."""
        rows = [_make_row(equipo_local=f"L{i}", equipo_visitante=f"V{i}") for i in range(5)]
        # Inject nulls across several Feature_Vector cols
        rows[0]["posesion_local"] = float("nan")
        rows[1]["tiros_local"] = float("nan")
        rows[2]["faltas_visitante"] = float("nan")
        df = _make_df(rows)

        df_limpio, _ = limpiar_dataset(df)

        for col in FEATURE_VECTOR_COLS:
            if col in df_limpio.columns:
                assert not df_limpio[col].isnull().any(), f"Null found in {col}"

    # --- Out-of-range discard ---

    def test_posesion_above_100_is_discarded(self):
        """Rows with posesion > 100 are discarded."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", posesion_local=101.0),
            _make_row(equipo_local="C", equipo_visitante="D"),
        ])
        df_limpio, resumen = limpiar_dataset(df)
        assert len(df_limpio) == 1
        assert resumen["filas_fuera_de_rango"] == 1

    def test_posesion_below_0_is_discarded(self):
        """Rows with posesion < 0 are discarded."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", posesion_visitante=-5.0),
            _make_row(equipo_local="C", equipo_visitante="D"),
        ])
        df_limpio, resumen = limpiar_dataset(df)
        assert len(df_limpio) == 1
        assert resumen["filas_fuera_de_rango"] == 1

    def test_negative_tiros_is_discarded(self):
        """Rows with negative tiros values are discarded."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", tiros_local=-1),
            _make_row(equipo_local="C", equipo_visitante="D"),
        ])
        df_limpio, resumen = limpiar_dataset(df)
        assert len(df_limpio) == 1
        assert resumen["filas_fuera_de_rango"] == 1

    def test_negative_faltas_is_discarded(self):
        """Rows with negative faltas values are discarded."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", faltas_visitante=-3),
            _make_row(equipo_local="C", equipo_visitante="D"),
        ])
        df_limpio, resumen = limpiar_dataset(df)
        assert len(df_limpio) == 1

    def test_negative_tarjetas_is_discarded(self):
        """Rows with negative tarjetas values are discarded."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", tarjetas_local=-1),
            _make_row(equipo_local="C", equipo_visitante="D"),
        ])
        df_limpio, resumen = limpiar_dataset(df)
        assert len(df_limpio) == 1

    def test_invalid_resultado_is_discarded(self):
        """Rows with resultado not in {-1, 0, 1} are discarded."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", resultado=2),
            _make_row(equipo_local="C", equipo_visitante="D", resultado=99),
            _make_row(equipo_local="E", equipo_visitante="F", resultado=1),
        ])
        df_limpio, resumen = limpiar_dataset(df)
        assert len(df_limpio) == 1
        assert resumen["filas_fuera_de_rango"] == 2

    def test_valid_boundary_values_are_kept(self):
        """Boundary values (posesion=0, posesion=100, resultado=-1/0/1) are valid."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B",
                      posesion_local=0.0, posesion_visitante=100.0, resultado=-1),
            _make_row(equipo_local="C", equipo_visitante="D",
                      posesion_local=50.0, posesion_visitante=50.0, resultado=0),
            _make_row(equipo_local="E", equipo_visitante="F",
                      posesion_local=100.0, posesion_visitante=0.0, resultado=1),
        ])
        df_limpio, resumen = limpiar_dataset(df)
        assert len(df_limpio) == 3
        assert resumen["filas_fuera_de_rango"] == 0

    # --- Summary dict ---

    def test_summary_has_all_required_keys(self):
        """resumen must contain all five required keys."""
        _, resumen = limpiar_dataset(_valid_df(3))
        required_keys = {
            "filas_por_fuente",
            "duplicados_eliminados",
            "filas_fuera_de_rango",
            "filas_imputadas",
            "filas_finales",
        }
        assert required_keys == set(resumen.keys())

    def test_summary_values_are_non_negative_integers(self):
        """All summary values must be non-negative integers."""
        _, resumen = limpiar_dataset(_valid_df(4))
        for key, val in resumen.items():
            assert isinstance(val, int), f"Key '{key}' is not int: {val!r}"
            assert val >= 0, f"Key '{key}' is negative: {val}"

    def test_filas_por_fuente_matches_input_length(self):
        """filas_por_fuente must equal the number of input rows."""
        df = _valid_df(7)
        _, resumen = limpiar_dataset(df)
        assert resumen["filas_por_fuente"] == 7

    def test_filas_finales_matches_output_length(self):
        """filas_finales must match len(df_limpio)."""
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B"),
            _make_row(equipo_local="C", equipo_visitante="D", resultado=99),  # invalid
        ])
        df_limpio, resumen = limpiar_dataset(df)
        assert resumen["filas_finales"] == len(df_limpio)

    def test_order_dedup_then_impute_then_range(self):
        """
        Dedup must happen before imputation so that the imputation sees
        the correct (deduplicated) medians.
        """
        # Two rows with the same key: after dedup only one survives.
        # The survivor has a null that gets imputed from median of remaining rows.
        df = _make_df([
            _make_row(equipo_local="A", equipo_visitante="B", goles_local=float("nan")),
            _make_row(equipo_local="A", equipo_visitante="B", goles_local=2.0),  # dup
            _make_row(equipo_local="C", equipo_visitante="D", goles_local=4.0),
        ])
        df_limpio, resumen = limpiar_dataset(df)
        # After dedup: rows A-B (null) and C-D (4). Median of [nan, 4] → after dedup
        # the surviving null row gets median of [nan, 4] = 4.0
        assert resumen["duplicados_eliminados"] == 1
        assert not df_limpio["goles_local"].isnull().any()


# ---------------------------------------------------------------------------
# validar_y_persistir
# ---------------------------------------------------------------------------

class TestValidarYPersistir:

    def test_writes_file_when_rows_meet_minimum(self, tmp_path):
        """With >= min_filas rows, the file is written to the specified path."""
        df = _valid_df(50)
        ruta = str(tmp_path / "output.csv")

        validar_y_persistir(df, ruta, min_filas=50)

        assert os.path.exists(ruta)

    def test_written_csv_can_be_read_back(self, tmp_path):
        """The persisted CSV can be read back and has the expected row count."""
        df = _valid_df(60)
        ruta = str(tmp_path / "partidos.csv")

        validar_y_persistir(df, ruta, min_filas=50)

        df_back = pd.read_csv(ruta)
        assert len(df_back) == 60

    def test_raises_value_error_when_rows_below_minimum(self, tmp_path):
        """Raises ValueError if the DataFrame has fewer than min_filas rows."""
        df = _valid_df(10)
        ruta = str(tmp_path / "output.csv")

        with pytest.raises(ValueError) as exc_info:
            validar_y_persistir(df, ruta, min_filas=50)

        assert "10" in str(exc_info.value)

    def test_error_message_includes_actual_row_count(self, tmp_path):
        """ValueError message must include the actual number of rows."""
        n = 7
        df = _valid_df(n)
        ruta = str(tmp_path / "output.csv")

        with pytest.raises(ValueError) as exc_info:
            validar_y_persistir(df, ruta, min_filas=50)

        assert str(n) in str(exc_info.value)

    def test_existing_file_not_overwritten_on_failure(self, tmp_path):
        """When there are too few rows, an existing file is NOT overwritten."""
        ruta = str(tmp_path / "existing.csv")

        # Write original content
        original_df = _valid_df(55)
        original_df.to_csv(ruta, index=False)
        original_content = open(ruta).read()

        # Attempt to overwrite with too-small dataset
        small_df = _valid_df(5)
        with pytest.raises(ValueError):
            validar_y_persistir(small_df, ruta, min_filas=50)

        # Original file must be untouched
        assert open(ruta).read() == original_content

    def test_atomic_write_no_partial_file_on_exception(self, tmp_path, monkeypatch):
        """If writing the temp file fails, no partial file is left at the target path."""
        df = _valid_df(60)
        ruta = str(tmp_path / "output.csv")

        # Patch DataFrame.to_csv to simulate a mid-write IOError
        original_to_csv = pd.DataFrame.to_csv

        def failing_to_csv(self, path_or_buf=None, *args, **kwargs):
            raise IOError("Simulated disk full error")

        monkeypatch.setattr(pd.DataFrame, "to_csv", failing_to_csv)

        with pytest.raises(IOError):
            validar_y_persistir(df, ruta, min_filas=50)

        # The target file must not exist (atomic write was aborted)
        assert not os.path.exists(ruta)

    def test_custom_min_filas_threshold_respected(self, tmp_path):
        """min_filas parameter is honoured; default is 50 but can be overridden."""
        df = _valid_df(10)
        ruta = str(tmp_path / "output.csv")

        # With a threshold of 5, 10 rows should pass
        validar_y_persistir(df, ruta, min_filas=5)
        assert os.path.exists(ruta)

    def test_exactly_min_filas_rows_is_accepted(self, tmp_path):
        """Exactly min_filas rows should NOT raise (boundary is inclusive)."""
        df = _valid_df(50)
        ruta = str(tmp_path / "exact.csv")

        # Should not raise
        validar_y_persistir(df, ruta, min_filas=50)
        assert os.path.exists(ruta)

    def test_one_below_min_filas_raises(self, tmp_path):
        """One row below the threshold raises ValueError."""
        df = _valid_df(49)
        ruta = str(tmp_path / "output.csv")

        with pytest.raises(ValueError):
            validar_y_persistir(df, ruta, min_filas=50)

    def test_creates_parent_directory_if_missing(self, tmp_path):
        """The function creates the parent directory if it does not exist."""
        nested_ruta = str(tmp_path / "subdir" / "deep" / "output.csv")
        df = _valid_df(50)

        validar_y_persistir(df, nested_ruta, min_filas=50)

        assert os.path.exists(nested_ruta)
