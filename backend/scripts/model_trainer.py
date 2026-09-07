"""
Model Trainer — pipeline de mejora del modelo de predicción ML.

Responsable de:
- Construir los grids de hiperparámetros para múltiples algoritmos.
- Entrenar, comparar y seleccionar el mejor modelo con validación cruzada.
- Evaluar el mejor modelo y generar un reporte de métricas.
- Persistir el reporte como JSON.
- Guardar el modelo con backup del archivo anterior.

Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3
"""

import os
import sys
import json
import shutil
from datetime import datetime, timezone

# Ensure backend/ is on the path so ml_model and scripts imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import ml_model
from scripts.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 6.1 — construir_grids
# ---------------------------------------------------------------------------

def construir_grids() -> list:
    """
    Retorna lista de dicts con 'nombre', 'estimador' y 'param_grid'
    para RandomForestClassifier, GradientBoostingClassifier y LogisticRegression.

    Cada param_grid tiene >= 3 combinaciones de hiperparámetros.

    Returns:
        list[dict]: Lista de tres dicts de configuración de algoritmos.

    Requirements: 4.1, 4.3
    """
    grids = [
        {
            "nombre": "RandomForestClassifier",
            "estimador": RandomForestClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        },
        {
            "nombre": "GradientBoostingClassifier",
            "estimador": GradientBoostingClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5],
            },
        },
        {
            "nombre": "LogisticRegression",
            "estimador": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs", "saga"],
            },
        },
    ]
    return grids


# ---------------------------------------------------------------------------
# 6.2 — entrenar_y_seleccionar
# ---------------------------------------------------------------------------

def entrenar_y_seleccionar(X, y, cv_folds: int = 5) -> tuple:
    """
    Escala features, aplica GridSearchCV con validación cruzada estratificada
    para cada algoritmo, y retorna el mejor modelo junto con metadatos.

    Args:
        X: Features (array-like de shape (n_samples, n_features)).
        y: Etiquetas (array-like de shape (n_samples,)).
        cv_folds: Número de folds para validación cruzada (default 5).
                  Si len(X) < 100 se fuerza a 3 y se emite advertencia.

    Returns:
        Tupla (mejor_estimador_fitted, scaler, resultados_cv_list, X_test, y_test)
        donde resultados_cv_list es lista de dicts:
          {'nombre', 'mejor_score_cv', 'mejores_params', 'estimador'}

    Requirements: 4.2, 4.3, 4.4, 4.5, 4.6
    """
    X = np.array(X)
    y = np.array(y)

    # Req 4.6: Reducir folds si el dataset es pequeño
    if len(X) < 100:
        logger.warning(
            f"El dataset tiene {len(X)} filas (< 100). "
            "Se reduce cv_folds a 3 para evitar errores de validación cruzada."
        )
        cv_folds = 3

    # Split 80/20 con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Req 4.5: Escalar con StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grids = construir_grids()
    resultados_cv_list = []
    mejor_score = -1.0
    mejor_estimador = None
    mejor_nombre = None
    mejor_params = None

    cv = StratifiedKFold(n_splits=cv_folds)

    for config in grids:
        nombre = config["nombre"]
        estimador = config["estimador"]
        param_grid = config["param_grid"]

        logger.info(f"Entrenando {nombre} con GridSearchCV ({cv_folds} folds)...")

        grid_search = GridSearchCV(
            estimator=estimador,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train_scaled, y_train)

        mean_score = grid_search.best_score_
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        logger.info(
            f"{nombre}: mejor accuracy CV = {mean_score:.4f}, "
            f"parámetros = {best_params}"
        )

        resultados_cv_list.append({
            "nombre": nombre,
            "mejor_score_cv": mean_score,
            "mejores_params": best_params,
            "estimador": best_estimator,
        })

        # Req 4.4: Seleccionar mayor score; en empate, el primero
        if mean_score > mejor_score:
            mejor_score = mean_score
            mejor_estimador = best_estimator
            mejor_nombre = nombre
            mejor_params = best_params

    logger.info(
        f"Mejor algoritmo seleccionado: {mejor_nombre} "
        f"con accuracy CV media = {mejor_score:.4f}"
    )

    return mejor_estimador, scaler, resultados_cv_list, X_test_scaled, y_test


# ---------------------------------------------------------------------------
# 6.5 — evaluar_y_reportar
# ---------------------------------------------------------------------------

def evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata: dict) -> dict:
    """
    Calcula métricas de evaluación del modelo sobre el conjunto de prueba
    y construye el dict del reporte completo.

    Args:
        modelo: Estimador entrenado de scikit-learn.
        scaler: StandardScaler ajustado al conjunto de entrenamiento
                (no se usa aquí; X_test ya debe estar escalado si corresponde).
        X_test: Features del conjunto de prueba (ya escaladas).
        y_test: Etiquetas del conjunto de prueba.
        metadata: dict con claves obligatorias:
            - nombre_algoritmo (str)
            - accuracy_cv_media (float)
            - accuracy_cv_std (float)
            - hiperparametros (dict)
            - filas_entrenamiento (int)
            - algoritmos_evaluados (list)

    Returns:
        dict con todas las claves del esquema model_report.json.

    Requirements: 5.1, 5.2, 5.4
    """
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_pred = modelo.predict(X_test)

    acc_test = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    # Extraer métricas por clase (excluyendo entradas de promedio)
    _exclude = {"accuracy", "macro avg", "weighted avg"}
    precision_por_clase = {
        str(k): v["precision"] for k, v in cr.items() if k not in _exclude
    }
    recall_por_clase = {
        str(k): v["recall"] for k, v in cr.items() if k not in _exclude
    }
    f1_por_clase = {
        str(k): v["f1-score"] for k, v in cr.items() if k not in _exclude
    }

    fecha_hora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    reporte = {
        "nombre_algoritmo": metadata["nombre_algoritmo"],
        "accuracy_test": acc_test,
        "accuracy_cv_media": metadata["accuracy_cv_media"],
        "accuracy_cv_std": metadata["accuracy_cv_std"],
        "hiperparametros": metadata["hiperparametros"],
        "fecha_hora": fecha_hora,
        "filas_entrenamiento": metadata["filas_entrenamiento"],
        "filas_prueba": len(y_test),
        "precision_por_clase": precision_por_clase,
        "recall_por_clase": recall_por_clase,
        "f1_por_clase": f1_por_clase,
        "matriz_confusion": cm.tolist(),
        "algoritmos_evaluados": metadata["algoritmos_evaluados"],
    }

    logger.info(
        f"Reporte generado para {metadata['nombre_algoritmo']}: "
        f"accuracy_test={acc_test:.4f}"
    )

    return reporte


# ---------------------------------------------------------------------------
# 6.7 — persistir_reporte
# ---------------------------------------------------------------------------

def persistir_reporte(reporte: dict, ruta: str) -> None:
    """
    Guarda el reporte como JSON en ruta, sobrescribiendo si ya existe.

    Si la escritura falla por IOError, registra el error con logging.error
    y continúa sin propagar la excepción.

    Args:
        reporte: dict con el contenido del reporte a persistir.
        ruta:    Ruta destino del archivo JSON.

    Requirements: 5.2, 5.3
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(reporte, f, ensure_ascii=False, indent=2)
        logger.info(f"Reporte persistido en '{ruta}'.")
    except IOError as e:
        logger.error(f"Error al escribir el reporte en '{ruta}': {e}")


# ---------------------------------------------------------------------------
# 7.1 — guardar_modelo_con_backup
# ---------------------------------------------------------------------------

def guardar_modelo_con_backup(modelo, scaler, ruta: str) -> None:
    """
    Guarda el modelo y el scaler en ruta, creando un backup del archivo
    existente antes de sobrescribir.

    - Si ruta ya existe: copia a ruta_backup = ruta.replace('.pkl', '_backup_<ts>.pkl').
      Si la copia falla, registra advertencia y continúa (best-effort).
    - Llama a ml_model.guardar_modelo(modelo, scaler, ruta) para persistir.
    - Si guardar_modelo lanza IOError: registra error y re-lanza SIN eliminar
      el modelo.pkl existente.

    Args:
        modelo:  Estimador entrenado de scikit-learn.
        scaler:  StandardScaler ajustado.
        ruta:    Ruta destino del archivo .pkl.

    Requirements: 6.1, 6.2, 6.3
    """
    # Crear backup si el archivo ya existe
    if os.path.exists(ruta):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_backup = ruta.replace(".pkl", f"_backup_{timestamp}.pkl")
        try:
            shutil.copy2(ruta, ruta_backup)
            logger.info(f"Backup creado en '{ruta_backup}'.")
        except Exception as e:
            logger.warning(
                f"No se pudo crear el backup de '{ruta}': {e}. "
                "Continuando con la sobrescritura (best-effort)."
            )

    # Guardar el nuevo modelo
    try:
        ml_model.guardar_modelo(modelo, scaler, ruta)
        logger.info(f"Modelo guardado exitosamente en '{ruta}'.")
    except IOError as e:
        logger.error(
            f"Error al guardar el modelo en '{ruta}': {e}. "
            "El archivo existente no ha sido eliminado."
        )
        raise
