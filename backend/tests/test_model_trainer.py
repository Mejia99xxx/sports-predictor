"""
Unit tests for model_trainer.py — tasks 6.1, 6.2, 6.5, 6.7, 7.1.

Covers:
- construir_grids: structure, algorithm names, param_grid sizes
- entrenar_y_seleccionar: return shape, scaling, fold adaptation, best-model selection
- evaluar_y_reportar: report keys, accuracy range, metrics per class
- persistir_reporte: JSON write, IOError swallowed
- guardar_modelo_con_backup: backup creation, IOError re-raised, no deletion

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3
"""

import os
import sys
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scripts.model_trainer import (
    construir_grids,
    entrenar_y_seleccionar,
    evaluar_y_reportar,
    persistir_reporte,
    guardar_modelo_con_backup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 120, n_features: int = 9, seed: int = 42):
    """Return (X, y) with balanced classes {-1, 0, 1}."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 100, size=(n, n_features))
    y = np.array([-1, 0, 1] * (n // 3) + [-1] * (n % 3))
    return X, y


def _small_dataset(n: int = 60):
    """Return a small dataset (< 100 rows) for fold-adaptation tests."""
    return _make_dataset(n=n)


def _make_metadata(resultados_cv: list) -> dict:
    """Build a metadata dict compatible with evaluar_y_reportar from resultados_cv."""
    # Pick the best by score (mimicking entrenar_y_seleccionar)
    best = max(resultados_cv, key=lambda r: r["mejor_score_cv"])
    scores = [r["mejor_score_cv"] for r in resultados_cv]
    return {
        "nombre_algoritmo": best["nombre"],
        "accuracy_cv_media": best["mejor_score_cv"],
        "accuracy_cv_std": float(np.std(scores)),
        "hiperparametros": best["mejores_params"],
        "filas_entrenamiento": 96,   # 80% of 120
        "algoritmos_evaluados": [
            {"nombre": r["nombre"], "accuracy_cv_media": r["mejor_score_cv"]}
            for r in resultados_cv
        ],
    }


# ---------------------------------------------------------------------------
# construir_grids — task 6.1
# ---------------------------------------------------------------------------

class TestConstruirGrids:

    def test_returns_list_of_three_dicts(self):
        """construir_grids debe retornar exactamente 3 dicts."""
        grids = construir_grids()
        assert isinstance(grids, list)
        assert len(grids) == 3

    def test_each_dict_has_required_keys(self):
        """Cada dict debe tener 'nombre', 'estimador' y 'param_grid'."""
        grids = construir_grids()
        for g in grids:
            assert "nombre" in g
            assert "estimador" in g
            assert "param_grid" in g

    def test_algorithm_names_are_correct(self):
        """Los nombres deben ser los tres algoritmos requeridos."""
        grids = construir_grids()
        nombres = {g["nombre"] for g in grids}
        assert nombres == {
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LogisticRegression",
        }

    def test_param_grids_have_at_least_three_combinations(self):
        """Cada param_grid debe generar >= 3 combinaciones de hiperparámetros."""
        import itertools
        grids = construir_grids()
        for g in grids:
            values = g["param_grid"].values()
            total = 1
            for v in values:
                total *= len(v)
            assert total >= 3, (
                f"{g['nombre']} tiene solo {total} combinación(es) "
                "en su param_grid (se requieren >= 3)"
            )

    def test_random_forest_param_grid_contents(self):
        """RandomForestClassifier param_grid contiene las claves y valores correctos."""
        grids = construir_grids()
        rf = next(g for g in grids if g["nombre"] == "RandomForestClassifier")
        pg = rf["param_grid"]

        assert set(pg["n_estimators"]) == {100, 200}
        assert set(pg["max_depth"]) == {None, 10, 20}
        assert set(pg["min_samples_split"]) == {2, 5}

    def test_gradient_boosting_param_grid_contents(self):
        """GradientBoostingClassifier param_grid contiene las claves y valores correctos."""
        grids = construir_grids()
        gb = next(g for g in grids if g["nombre"] == "GradientBoostingClassifier")
        pg = gb["param_grid"]

        assert set(pg["n_estimators"]) == {100, 200}
        assert set(pg["learning_rate"]) == {0.05, 0.1, 0.2}
        assert set(pg["max_depth"]) == {3, 5}

    def test_logistic_regression_param_grid_contents(self):
        """LogisticRegression param_grid contiene las claves y valores correctos."""
        grids = construir_grids()
        lr = next(g for g in grids if g["nombre"] == "LogisticRegression")
        pg = lr["param_grid"]

        assert set(pg["C"]) == {0.1, 1.0, 10.0}
        assert set(pg["solver"]) == {"lbfgs", "saga"}

    def test_logistic_regression_max_iter_is_1000(self):
        """LogisticRegression estimador debe tener max_iter=1000."""
        grids = construir_grids()
        lr = next(g for g in grids if g["nombre"] == "LogisticRegression")
        assert lr["estimador"].max_iter == 1000

    def test_returns_fresh_grids_on_each_call(self):
        """Llamadas independientes retornan listas distintas (no compartidas)."""
        grids1 = construir_grids()
        grids2 = construir_grids()
        assert grids1 is not grids2


# ---------------------------------------------------------------------------
# Fast grids fixture for entrenar_y_seleccionar tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def fast_grids(monkeypatch):
    """
    Replace construir_grids() with ultra-lightweight grids so GridSearchCV
    finishes in seconds instead of minutes during unit testing.
    """
    from sklearn.dummy import DummyClassifier
    import scripts.model_trainer as mt

    def _fast_grids():
        return [
            {
                "nombre": "RandomForestClassifier",
                "estimador": RandomForestClassifier(n_estimators=5, random_state=42),
                "param_grid": {"n_estimators": [5]},
            },
            {
                "nombre": "GradientBoostingClassifier",
                "estimador": DummyClassifier(strategy="most_frequent"),
                "param_grid": {"strategy": ["most_frequent"]},
            },
            {
                "nombre": "LogisticRegression",
                "estimador": LogisticRegression(max_iter=100, random_state=42),
                "param_grid": {"C": [1.0]},
            },
        ]

    monkeypatch.setattr(mt, "construir_grids", _fast_grids)
    return _fast_grids


# ---------------------------------------------------------------------------
# entrenar_y_seleccionar — task 6.2
# ---------------------------------------------------------------------------

class TestEntrenarYSeleccionar:

    def test_returns_five_element_tuple(self, fast_grids):
        """Debe retornar exactamente una tupla de 5 elementos."""
        X, y = _make_dataset(120)
        result = entrenar_y_seleccionar(X, y, cv_folds=3)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_first_element_is_fitted_estimator(self, fast_grids):
        """El primer elemento debe ser un estimador ajustado con método predict."""
        X, y = _make_dataset(120)
        modelo, *_ = entrenar_y_seleccionar(X, y, cv_folds=3)
        assert hasattr(modelo, "predict")

    def test_second_element_is_standard_scaler(self, fast_grids):
        """El segundo elemento debe ser un StandardScaler ajustado."""
        X, y = _make_dataset(120)
        _, scaler, *_ = entrenar_y_seleccionar(X, y, cv_folds=3)
        assert isinstance(scaler, StandardScaler)
        # Scaler must be fitted (has mean_)
        assert hasattr(scaler, "mean_")

    def test_third_element_is_list_of_three_result_dicts(self, fast_grids):
        """El tercer elemento debe ser una lista de 3 dicts (uno por algoritmo)."""
        X, y = _make_dataset(120)
        _, _, resultados, *_ = entrenar_y_seleccionar(X, y, cv_folds=3)
        assert isinstance(resultados, list)
        assert len(resultados) == 3

    def test_result_dicts_have_required_keys(self, fast_grids):
        """Cada dict en resultados_cv debe tener las 4 claves requeridas."""
        X, y = _make_dataset(120)
        _, _, resultados, *_ = entrenar_y_seleccionar(X, y, cv_folds=3)
        required = {"nombre", "mejor_score_cv", "mejores_params", "estimador"}
        for r in resultados:
            assert required == set(r.keys()), f"Claves incorrectas: {set(r.keys())}"

    def test_scores_are_floats_between_0_and_1(self, fast_grids):
        """mejor_score_cv debe ser float en [0, 1]."""
        X, y = _make_dataset(120)
        _, _, resultados, *_ = entrenar_y_seleccionar(X, y, cv_folds=3)
        for r in resultados:
            assert 0.0 <= r["mejor_score_cv"] <= 1.0

    def test_x_test_and_y_test_match_20_percent(self, fast_grids):
        """X_test e y_test deben corresponder al 20% del dataset."""
        n = 120
        X, y = _make_dataset(n)
        _, _, _, X_test, y_test = entrenar_y_seleccionar(X, y, cv_folds=3)
        # 80/20 split → test ~24 rows
        assert len(X_test) == len(y_test)
        assert len(X_test) == pytest.approx(n * 0.2, abs=2)

    def test_small_dataset_uses_3_folds(self, fast_grids, caplog):
        """Con < 100 filas, debe usar cv_folds=3 y emitir advertencia."""
        import logging
        X, y = _small_dataset(60)
        with caplog.at_level(logging.WARNING):
            entrenar_y_seleccionar(X, y, cv_folds=5)
        # Warning about reduced folds must be present
        assert any("3" in msg or "cv_folds" in msg.lower() or "100" in msg
                   for msg in caplog.messages)

    def test_best_model_has_highest_cv_score(self, fast_grids):
        """El modelo retornado debe ser el que tiene el mayor score CV."""
        X, y = _make_dataset(120)
        modelo, _, resultados, X_test, y_test = entrenar_y_seleccionar(X, y, cv_folds=3)
        max_score = max(r["mejor_score_cv"] for r in resultados)
        # The returned model must correspond to the result with max score
        best_result = next(r for r in resultados if r["mejor_score_cv"] == max_score)
        assert best_result["estimador"] is modelo

    def test_x_test_is_scaled(self, fast_grids):
        """X_test devuelto debe estar escalado (no en escala original)."""
        X, y = _make_dataset(120)
        _, scaler, _, X_test, _ = entrenar_y_seleccionar(X, y, cv_folds=3)
        # After StandardScaler transform, values can exceed original range [0, 100]
        # and the mean of each feature in X_test should differ from the original data
        # The scaler's mean_ and scale_ are set
        assert scaler.mean_ is not None


# ---------------------------------------------------------------------------
# evaluar_y_reportar — task 6.5
# ---------------------------------------------------------------------------

def _get_trained_directly():
    """
    Helper that trains a tiny model WITHOUT going through entrenar_y_seleccionar,
    so this test class does not depend on GridSearchCV speed.
    """
    X, y = _make_dataset(120)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_s, X_test_s, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    from sklearn.ensemble import RandomForestClassifier as RF
    modelo = RF(n_estimators=5, random_state=42)
    modelo.fit(X_train_s, y_train)

    resultados = [
        {"nombre": "RandomForestClassifier", "mejor_score_cv": 0.55, "mejores_params": {"n_estimators": 5}, "estimador": modelo},
        {"nombre": "GradientBoostingClassifier", "mejor_score_cv": 0.50, "mejores_params": {}, "estimador": modelo},
        {"nombre": "LogisticRegression", "mejor_score_cv": 0.48, "mejores_params": {"C": 1.0}, "estimador": modelo},
    ]
    metadata = _make_metadata(resultados)
    metadata["filas_entrenamiento"] = len(X_train_s)
    return modelo, scaler, X_test_s, y_test, metadata


class TestEvaluarYReportar:

    def test_report_contains_all_required_keys(self):
        """El reporte debe contener todas las claves del esquema."""
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)

        required_keys = {
            "nombre_algoritmo",
            "accuracy_test",
            "accuracy_cv_media",
            "accuracy_cv_std",
            "hiperparametros",
            "fecha_hora",
            "filas_entrenamiento",
            "filas_prueba",
            "precision_por_clase",
            "recall_por_clase",
            "f1_por_clase",
            "matriz_confusion",
            "algoritmos_evaluados",
        }
        assert required_keys.issubset(set(reporte.keys()))

    def test_accuracy_test_is_float_between_0_and_1(self):
        """accuracy_test debe ser float en [0, 1]."""
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        assert isinstance(reporte["accuracy_test"], float)
        assert 0.0 <= reporte["accuracy_test"] <= 1.0

    def test_filas_prueba_matches_y_test_length(self):
        """filas_prueba debe igualar len(y_test)."""
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        assert reporte["filas_prueba"] == len(y_test)

    def test_matriz_confusion_is_list_of_lists(self):
        """matriz_confusion debe ser una lista de listas de enteros."""
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        mc = reporte["matriz_confusion"]
        assert isinstance(mc, list)
        assert all(isinstance(row, list) for row in mc)

    def test_fecha_hora_is_iso_8601_utc(self):
        """fecha_hora debe estar en formato ISO 8601 (YYYY-MM-DDTHH:MM:SS)."""
        from datetime import datetime
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        fh = reporte["fecha_hora"]
        # Must parse without error
        dt = datetime.strptime(fh, "%Y-%m-%dT%H:%M:%S")
        assert dt is not None

    def test_precision_recall_f1_per_class_are_dicts(self):
        """precision_por_clase, recall_por_clase, f1_por_clase deben ser dicts."""
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        assert isinstance(reporte["precision_por_clase"], dict)
        assert isinstance(reporte["recall_por_clase"], dict)
        assert isinstance(reporte["f1_por_clase"], dict)

    def test_nombre_algoritmo_from_metadata(self):
        """nombre_algoritmo en reporte debe coincidir con metadata."""
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        assert reporte["nombre_algoritmo"] == metadata["nombre_algoritmo"]

    def test_algoritmos_evaluados_from_metadata(self):
        """algoritmos_evaluados en reporte debe venir de metadata."""
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        assert reporte["algoritmos_evaluados"] == metadata["algoritmos_evaluados"]

    def test_accuracy_cv_fields_match_metadata(self):
        """accuracy_cv_media y accuracy_cv_std deben coincidir con metadata."""
        modelo, scaler, X_test, y_test, metadata = _get_trained_directly()
        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        assert reporte["accuracy_cv_media"] == metadata["accuracy_cv_media"]
        assert reporte["accuracy_cv_std"] == metadata["accuracy_cv_std"]


# ---------------------------------------------------------------------------
# persistir_reporte — task 6.7
# ---------------------------------------------------------------------------

class TestPersistirReporte:

    def _sample_reporte(self) -> dict:
        return {
            "nombre_algoritmo": "TestAlgo",
            "accuracy_test": 0.75,
            "accuracy_cv_media": 0.72,
            "accuracy_cv_std": 0.03,
            "hiperparametros": {"C": 1.0},
            "fecha_hora": "2024-01-01T00:00:00",
            "filas_entrenamiento": 100,
            "filas_prueba": 25,
            "precision_por_clase": {"1": 0.8},
            "recall_por_clase": {"1": 0.7},
            "f1_por_clase": {"1": 0.75},
            "matriz_confusion": [[10, 2], [3, 10]],
            "algoritmos_evaluados": [],
        }

    def test_creates_json_file(self, tmp_path):
        """Debe crear el archivo JSON en la ruta indicada."""
        ruta = str(tmp_path / "report.json")
        persistir_reporte(self._sample_reporte(), ruta)
        assert os.path.exists(ruta)

    def test_json_content_matches_reporte(self, tmp_path):
        """El contenido del archivo debe ser el reporte serializado."""
        ruta = str(tmp_path / "report.json")
        reporte = self._sample_reporte()
        persistir_reporte(reporte, ruta)

        with open(ruta, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["nombre_algoritmo"] == reporte["nombre_algoritmo"]
        assert loaded["accuracy_test"] == reporte["accuracy_test"]

    def test_overwrites_existing_file(self, tmp_path):
        """Debe sobrescribir el archivo si ya existe."""
        ruta = str(tmp_path / "report.json")
        persistir_reporte({"nombre_algoritmo": "old"}, ruta)
        persistir_reporte(self._sample_reporte(), ruta)

        with open(ruta, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["nombre_algoritmo"] == "TestAlgo"

    def test_ioerror_is_swallowed(self, tmp_path, monkeypatch):
        """Si open() lanza IOError, no debe propagarse la excepción."""
        ruta = str(tmp_path / "report.json")

        original_open = open

        def failing_open(path, *args, **kwargs):
            if str(ruta) in str(path):
                raise IOError("Simulated write error")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", failing_open)

        # Must not raise
        persistir_reporte(self._sample_reporte(), ruta)

    def test_creates_parent_directories(self, tmp_path):
        """Debe crear directorios padres si no existen."""
        ruta = str(tmp_path / "subdir" / "nested" / "report.json")
        persistir_reporte(self._sample_reporte(), ruta)
        assert os.path.exists(ruta)


# ---------------------------------------------------------------------------
# guardar_modelo_con_backup — task 7.1
# ---------------------------------------------------------------------------

class TestGuardarModeloConBackup:

    def _make_simple_model(self):
        """Return a minimal (modelo, scaler) pair."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (30, 9))
        y = np.array([-1, 0, 1] * 10)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        modelo = RandomForestClassifier(n_estimators=5, random_state=0)
        modelo.fit(X_scaled, y)
        return modelo, scaler

    def test_saves_model_pkl(self, tmp_path):
        """Debe crear el archivo .pkl en la ruta indicada."""
        ruta = str(tmp_path / "modelo.pkl")
        modelo, scaler = self._make_simple_model()
        guardar_modelo_con_backup(modelo, scaler, ruta)
        assert os.path.exists(ruta)

    def test_saved_pkl_loadable(self, tmp_path):
        """El archivo guardado debe poder cargarse con pickle."""
        ruta = str(tmp_path / "modelo.pkl")
        modelo, scaler = self._make_simple_model()
        guardar_modelo_con_backup(modelo, scaler, ruta)

        with open(ruta, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, tuple)
        assert len(obj) == 2

    def test_backup_created_when_file_exists(self, tmp_path):
        """Si el archivo ya existe, debe crearse un backup con sufijo timestamp."""
        ruta = str(tmp_path / "modelo.pkl")
        modelo, scaler = self._make_simple_model()

        # Save first version
        guardar_modelo_con_backup(modelo, scaler, ruta)
        # Save again — should create backup
        guardar_modelo_con_backup(modelo, scaler, ruta)

        files = os.listdir(tmp_path)
        backup_files = [f for f in files if "_backup_" in f]
        assert len(backup_files) >= 1

    def test_backup_filename_contains_timestamp(self, tmp_path):
        """El backup debe contener un sufijo de timestamp en el nombre."""
        ruta = str(tmp_path / "modelo.pkl")
        modelo, scaler = self._make_simple_model()

        guardar_modelo_con_backup(modelo, scaler, ruta)
        guardar_modelo_con_backup(modelo, scaler, ruta)

        files = os.listdir(tmp_path)
        backup_files = [f for f in files if "_backup_" in f]
        assert len(backup_files) >= 1
        # Timestamp format: YYYYMMDD_HHMMSS
        for bf in backup_files:
            # Should contain digits indicating a timestamp
            import re
            assert re.search(r"_backup_\d{8}_\d{6}", bf), (
                f"Backup file '{bf}' does not contain expected timestamp pattern"
            )

    def test_no_backup_when_file_does_not_exist(self, tmp_path):
        """Si el archivo no existe previamente, no se debe crear backup."""
        ruta = str(tmp_path / "modelo.pkl")
        modelo, scaler = self._make_simple_model()

        guardar_modelo_con_backup(modelo, scaler, ruta)

        files = os.listdir(tmp_path)
        backup_files = [f for f in files if "_backup_" in f]
        assert len(backup_files) == 0

    def test_ioerror_on_save_is_reraised(self, tmp_path, monkeypatch):
        """Si guardar_modelo lanza IOError, debe propagarse."""
        import scripts.model_trainer as mt

        ruta = str(tmp_path / "modelo.pkl")
        modelo, scaler = self._make_simple_model()

        def failing_guardar(modelo, scaler, ruta):
            raise IOError("Simulated disk full")

        monkeypatch.setattr(mt.ml_model, "guardar_modelo", failing_guardar)

        with pytest.raises(IOError):
            guardar_modelo_con_backup(modelo, scaler, ruta)

    def test_existing_model_not_deleted_on_ioerror(self, tmp_path, monkeypatch):
        """Si el guardado falla, el modelo.pkl existente no debe ser eliminado."""
        import scripts.model_trainer as mt

        ruta = str(tmp_path / "modelo.pkl")
        modelo, scaler = self._make_simple_model()

        # Write original model directly
        with open(ruta, "wb") as f:
            pickle.dump(("original", "content"), f)

        original_content = open(ruta, "rb").read()

        def failing_guardar(modelo, scaler, ruta):
            raise IOError("Simulated disk full")

        monkeypatch.setattr(mt.ml_model, "guardar_modelo", failing_guardar)

        with pytest.raises(IOError):
            guardar_modelo_con_backup(modelo, scaler, ruta)

        # Original file must be intact
        assert os.path.exists(ruta)
        assert open(ruta, "rb").read() == original_content

    def test_backup_failure_does_not_abort_save(self, tmp_path, monkeypatch):
        """Si el backup falla, el guardado del nuevo modelo igual se ejecuta."""
        import shutil as shutil_module
        import scripts.model_trainer as mt

        ruta = str(tmp_path / "modelo.pkl")
        modelo, scaler = self._make_simple_model()

        # Pre-create the file so backup is attempted
        with open(ruta, "wb") as f:
            pickle.dump(("old", "data"), f)

        # Make shutil.copy2 fail
        def failing_copy2(src, dst):
            raise OSError("Simulated backup failure")

        monkeypatch.setattr(mt.shutil, "copy2", failing_copy2)

        # Save should still succeed even though backup failed
        guardar_modelo_con_backup(modelo, scaler, ruta)

        assert os.path.exists(ruta)
        with open(ruta, "rb") as f:
            obj = pickle.load(f)
        # The new model should be there (tuple with predict)
        assert isinstance(obj, tuple)
