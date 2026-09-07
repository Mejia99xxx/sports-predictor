"""
Run Pipeline — Orquestador del pipeline completo de mejora del modelo ML.

Responsable de:
- Orquestar en secuencia todas las etapas del pipeline.
- Manejar flags --skip-api y --skip-download desde CLI (argparse).
- Abortar etapas siguientes si una etapa falla con excepción no recuperable.
- Registrar resumen final con métricas clave al concluir exitosamente.

Uso:
    python backend/scripts/run_pipeline.py [--skip-api] [--skip-download]

Requisitos: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import argparse
import os
import sys

# Asegurar que backend/ esté en sys.path para que los imports relativos funcionen
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import numpy as np
import pandas as pd

from scripts.logger import get_logger
from scripts.data_collector import (
    descargar_csv_publicos,
    obtener_fixture_ids_por_temporada,
    recolectar_desde_api,
    FUENTES_DEFAULT,
)
from scripts.data_cleaner import combinar_fuentes, limpiar_dataset, validar_y_persistir
from scripts.model_trainer import (
    entrenar_y_seleccionar,
    evaluar_y_reportar,
    persistir_reporte,
    guardar_modelo_con_backup,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

API_KEY = "3a5f469c7cb18934b76eb4818ed42a9a"

ruta_historial = os.path.join(BACKEND_DIR, "data", "historial_partidos.csv")
ruta_reporte = os.path.join(BACKEND_DIR, "data", "model_report.json")
ruta_modelo = os.path.join(BACKEND_DIR, "modelo.pkl")

# Columnas del Feature_Vector usadas para entrenamiento
FEATURE_COLS = [
    "posesion_local",
    "posesion_visitante",
    "tiros_local",
    "tiros_visitante",
    "faltas_local",
    "faltas_visitante",
    "tarjetas_local",
    "tarjetas_visitante",
    "diferencia_goles",
]
TARGET_COL = "resultado"


# ---------------------------------------------------------------------------
# ejecutar_pipeline
# ---------------------------------------------------------------------------

def ejecutar_pipeline(skip_api: bool = False, skip_download: bool = False) -> int:
    """
    Orquesta en secuencia todas las etapas del pipeline ML.

    Si cualquier etapa lanza una excepción no recuperable, se registra la
    etapa fallida y el error, se abortan las etapas siguientes y se retorna 1.
    Si el pipeline completo finaliza exitosamente, se registra un resumen
    con las métricas clave y se retorna 0.

    Args:
        skip_api:      Si True, omite las llamadas a la API-Football.
        skip_download: Si True, omite la descarga de CSVs públicos.

    Returns:
        0 si el pipeline finaliza exitosamente, 1 si alguna etapa falla.

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    logger.info("=" * 60)
    logger.info("Iniciando pipeline de mejora del modelo ML")
    logger.info(f"  skip_api={skip_api}, skip_download={skip_download}")
    logger.info("=" * 60)

    dfs_csv = []
    dfs_api = []

    # ------------------------------------------------------------------
    # Etapa 1: Descargar CSVs públicos
    # ------------------------------------------------------------------
    if skip_download:
        logger.info("[Etapa 1] Descarga de CSVs públicos omitida (--skip-download).")
    else:
        try:
            logger.info("[Etapa 1] Descargando CSVs públicos...")
            dfs_csv = descargar_csv_publicos(FUENTES_DEFAULT)
            logger.info(f"[Etapa 1] Completada: {len(dfs_csv)} fuente(s) CSV descargadas.")
        except Exception as e:
            logger.error(f"[Etapa 1] Error en descarga de CSVs públicos: {e}")
            return 1

    # ------------------------------------------------------------------
    # Etapa 2: Recolectar desde API-Football
    # ------------------------------------------------------------------
    if skip_api:
        logger.info("[Etapa 2] Recolección de API-Football omitida (--skip-api).")
    else:
        try:
            logger.info("[Etapa 2] Obteniendo fixture IDs desde API-Football...")
            # Ligas por defecto: Premier League (39), La Liga (140)
            fixture_ids = []
            for league_id, season in [(39, 2023), (140, 2023)]:
                ids = obtener_fixture_ids_por_temporada(league_id, season, API_KEY)
                fixture_ids.extend(ids)
                logger.info(
                    f"[Etapa 2] league_id={league_id}, season={season}: "
                    f"{len(ids)} fixture IDs obtenidos."
                )

            logger.info(
                f"[Etapa 2] Total fixture IDs: {len(fixture_ids)}. "
                "Recolectando estadísticas..."
            )
            dfs_api = recolectar_desde_api(fixture_ids, API_KEY)
            logger.info(
                f"[Etapa 2] Completada: {len(dfs_api)} registros obtenidos de la API."
            )
        except Exception as e:
            logger.error(f"[Etapa 2] Error en recolección desde API-Football: {e}")
            return 1

    # ------------------------------------------------------------------
    # Etapa 3: Combinar fuentes
    # ------------------------------------------------------------------
    try:
        logger.info("[Etapa 3] Combinando fuentes de datos...")
        df_combinado = combinar_fuentes(dfs_csv, dfs_api)
        logger.info(
            f"[Etapa 3] Completada: {len(df_combinado)} filas en el dataset combinado."
        )
    except Exception as e:
        logger.error(f"[Etapa 3] Error al combinar fuentes: {e}")
        return 1

    # ------------------------------------------------------------------
    # Etapa 4: Limpiar dataset
    # ------------------------------------------------------------------
    try:
        logger.info("[Etapa 4] Limpiando dataset...")
        df_limpio, resumen_limpieza = limpiar_dataset(df_combinado)
        logger.info(
            f"[Etapa 4] Completada. Resumen de limpieza: {resumen_limpieza}"
        )
    except Exception as e:
        logger.error(f"[Etapa 4] Error al limpiar dataset: {e}")
        return 1

    # ------------------------------------------------------------------
    # Etapa 5: Validar y persistir dataset
    # ------------------------------------------------------------------
    try:
        logger.info("[Etapa 5] Validando y persistiendo dataset...")
        validar_y_persistir(df_limpio, ruta_historial)
        logger.info(
            f"[Etapa 5] Completada: {len(df_limpio)} filas guardadas en '{ruta_historial}'."
        )
    except Exception as e:
        logger.error(f"[Etapa 5] Error al validar/persistir dataset: {e}")
        return 1

    # ------------------------------------------------------------------
    # Etapa 6: Entrenar y seleccionar modelo
    # ------------------------------------------------------------------
    try:
        logger.info("[Etapa 6] Entrenando y seleccionando modelo...")
        X = df_limpio[FEATURE_COLS].values
        y = df_limpio[TARGET_COL].values

        modelo, scaler, resultados_cv, X_test, y_test = entrenar_y_seleccionar(X, y)
        logger.info("[Etapa 6] Completada: modelo seleccionado.")
    except Exception as e:
        logger.error(f"[Etapa 6] Error al entrenar/seleccionar modelo: {e}")
        return 1

    # ------------------------------------------------------------------
    # Etapa 7: Evaluar y reportar
    # ------------------------------------------------------------------
    try:
        logger.info("[Etapa 7] Evaluando modelo y generando reporte...")

        # Identificar el mejor algoritmo y sus métricas CV
        mejor_resultado = max(resultados_cv, key=lambda r: r["mejor_score_cv"])
        nombre_algoritmo = mejor_resultado["nombre"]
        accuracy_cv_media = mejor_resultado["mejor_score_cv"]

        # Calcular std de CV del mejor algoritmo
        scores_cv = [r["mejor_score_cv"] for r in resultados_cv]
        accuracy_cv_std = float(np.std(scores_cv)) if len(scores_cv) > 1 else 0.0

        # Construir lista de algoritmos evaluados para el reporte
        algoritmos_evaluados = [
            {
                "nombre": r["nombre"],
                "accuracy_cv_media": r["mejor_score_cv"],
            }
            for r in resultados_cv
        ]

        # Número de filas de entrenamiento (80% del dataset)
        filas_totales = len(df_limpio)
        filas_entrenamiento = filas_totales - len(y_test)

        metadata = {
            "nombre_algoritmo": nombre_algoritmo,
            "accuracy_cv_media": accuracy_cv_media,
            "accuracy_cv_std": accuracy_cv_std,
            "hiperparametros": mejor_resultado["mejores_params"],
            "filas_entrenamiento": filas_entrenamiento,
            "algoritmos_evaluados": algoritmos_evaluados,
        }

        reporte = evaluar_y_reportar(modelo, scaler, X_test, y_test, metadata)
        logger.info(
            f"[Etapa 7] Completada: accuracy_test={reporte['accuracy_test']:.4f}."
        )
    except Exception as e:
        logger.error(f"[Etapa 7] Error al evaluar/reportar modelo: {e}")
        return 1

    # ------------------------------------------------------------------
    # Etapa 8: Persistir reporte
    # ------------------------------------------------------------------
    try:
        logger.info(f"[Etapa 8] Persistiendo reporte en '{ruta_reporte}'...")
        persistir_reporte(reporte, ruta_reporte)
        logger.info("[Etapa 8] Completada.")
    except Exception as e:
        logger.error(f"[Etapa 8] Error al persistir reporte: {e}")
        return 1

    # ------------------------------------------------------------------
    # Etapa 9: Guardar modelo con backup
    # ------------------------------------------------------------------
    try:
        logger.info(f"[Etapa 9] Guardando modelo en '{ruta_modelo}'...")
        guardar_modelo_con_backup(modelo, scaler, ruta_modelo)
        logger.info("[Etapa 9] Completada.")
    except Exception as e:
        logger.error(f"[Etapa 9] Error al guardar modelo: {e}")
        return 1

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("✅ Pipeline completado exitosamente.")
    logger.info(f"   Filas en dataset:     {len(df_limpio)}")
    logger.info(f"   Algoritmo seleccionado: {nombre_algoritmo}")
    logger.info(f"   Accuracy en test:     {reporte['accuracy_test']:.4f}")
    logger.info(f"   Modelo guardado en:   {ruta_modelo}")
    logger.info("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# main — CLI
# ---------------------------------------------------------------------------

def main():
    """
    Punto de entrada de línea de comandos para el pipeline.

    Flags:
        --skip-api        Omite las llamadas a la API-Football.
        --skip-download   Omite la descarga de CSVs públicos.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline completo de recolección, limpieza y entrenamiento del modelo ML."
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        default=False,
        help="Omitir las llamadas a la API-Football y usar solo datos del dataset público.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=False,
        help="Omitir la descarga de CSVs públicos y usar únicamente datos crudos en disco.",
    )

    args = parser.parse_args()
    exit_code = ejecutar_pipeline(skip_api=args.skip_api, skip_download=args.skip_download)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
