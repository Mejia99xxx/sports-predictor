"""
Data Cleaner — pipeline de mejora del modelo de predicción ML.

Responsable de:
- Combinar datos de fuentes CSV y API en un único DataFrame con el esquema
  de historial_partidos.csv.
- Limpiar el dataset: deduplicación, imputación de nulos con mediana,
  descarte de filas fuera de rango.
- Validar y persistir el dataset limpio de forma atómica.

Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

import os
import tempfile

import pandas as pd

from scripts.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Columnas exactas del esquema de historial_partidos.csv (en orden canónico)
COLUMNAS_ESQUEMA = [
    "equipo_local",
    "equipo_visitante",
    "goles_local",
    "goles_visitante",
    "posesion_local",
    "posesion_visitante",
    "tiros_local",
    "tiros_visitante",
    "faltas_local",
    "faltas_visitante",
    "tarjetas_local",
    "tarjetas_visitante",
    "resultado",
    "diferencia_goles",
]

# Columnas numéricas del Feature_Vector (candidatas a imputación con mediana)
FEATURE_VECTOR_COLS = [
    "goles_local",
    "goles_visitante",
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

# Columnas sujetas a validación de rango
_RANGO_POSESION = ["posesion_local", "posesion_visitante"]
_RANGO_NO_NEGATIVOS = [
    "tiros_local",
    "tiros_visitante",
    "faltas_local",
    "faltas_visitante",
    "tarjetas_local",
    "tarjetas_visitante",
]
_VALORES_RESULTADO = {-1, 0, 1}


# ---------------------------------------------------------------------------
# combinar_fuentes
# ---------------------------------------------------------------------------

def combinar_fuentes(
    df_csv: "pd.DataFrame | list[pd.DataFrame]",
    df_api: "list[dict] | pd.DataFrame",
) -> pd.DataFrame:
    """
    Concatena datos de fuentes CSV y API al esquema de historial_partidos.csv.

    Args:
        df_csv: DataFrame (o lista de DataFrames) con datos de fuentes CSV.
                Cada DataFrame debe tener al menos las columnas del esquema.
        df_api: Lista de dicts o DataFrame con datos de la API-Football.
                Cada dict debe tener las claves del esquema.

    Returns:
        DataFrame con exactamente las 14 columnas del esquema, en orden canónico.
        Puede estar vacío si ambas fuentes son vacías.
    """
    partes: list[pd.DataFrame] = []

    # --- Procesar fuentes CSV ---
    if isinstance(df_csv, list):
        csv_list = df_csv
    elif isinstance(df_csv, pd.DataFrame):
        csv_list = [df_csv] if not df_csv.empty else []
    else:
        csv_list = []

    for i, df in enumerate(csv_list):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.debug(f"Fuente CSV #{i} vacía — omitida.")
            continue
        partes.append(_normalizar_al_esquema(pd.DataFrame(df), origen=f"csv_{i}"))

    # --- Procesar fuente API ---
    if isinstance(df_api, pd.DataFrame):
        if not df_api.empty:
            partes.append(_normalizar_al_esquema(df_api, origen="api"))
    elif isinstance(df_api, list) and df_api:
        df_from_api = pd.DataFrame(df_api)
        partes.append(_normalizar_al_esquema(df_from_api, origen="api"))
    else:
        logger.debug("Fuente API vacía o None — omitida.")

    # --- Concatenar ---
    if not partes:
        logger.warning("combinar_fuentes: ambas fuentes están vacías. Se retorna DataFrame vacío.")
        return pd.DataFrame(columns=COLUMNAS_ESQUEMA)

    df_combinado = pd.concat(partes, ignore_index=True)

    # Asegurar el orden canónico de columnas y que no haya extra
    df_combinado = df_combinado[COLUMNAS_ESQUEMA]

    logger.info(
        f"combinar_fuentes: {len(df_combinado)} filas combinadas "
        f"({len(partes)} fuentes)."
    )
    return df_combinado


def _normalizar_al_esquema(df: pd.DataFrame, origen: str = "") -> pd.DataFrame:
    """
    Asegura que un DataFrame tenga exactamente las columnas del esquema.

    - Columnas faltantes se agregan con NaN.
    - Columnas extra se descartan.
    - Orden canónico se aplica.
    """
    # Agregar columnas faltantes como NaN
    for col in COLUMNAS_ESQUEMA:
        if col not in df.columns:
            logger.debug(
                f"[{origen}] Columna '{col}' no encontrada — se agrega como NaN."
            )
            df = df.copy()
            df[col] = float("nan")

    # Descartar columnas extra y reordenar
    return df[COLUMNAS_ESQUEMA].copy()


# ---------------------------------------------------------------------------
# limpiar_dataset
# ---------------------------------------------------------------------------

def limpiar_dataset(df: pd.DataFrame) -> tuple:
    """
    Aplica en orden: deduplicación, imputación de nulos con mediana,
    descarte de filas fuera de rango.

    Args:
        df: DataFrame con el esquema de historial_partidos.csv.

    Returns:
        Tupla (df_limpio, resumen) donde resumen es un dict con:
            - filas_por_fuente (int): total de filas originales.
            - duplicados_eliminados (int): filas eliminadas por duplicación.
            - filas_fuera_de_rango (int): filas eliminadas por rango inválido.
            - filas_imputadas (int): filas donde al menos un nulo fue imputado.
            - filas_finales (int): filas en el dataset resultante.
    """
    filas_originales = len(df)
    df = df.copy()

    # --- 1. Deduplicación por (equipo_local, equipo_visitante) ---
    n_antes_dedup = len(df)
    df = df.drop_duplicates(subset=["equipo_local", "equipo_visitante"], keep="first")
    duplicados_eliminados = n_antes_dedup - len(df)
    if duplicados_eliminados > 0:
        logger.info(f"limpiar_dataset: {duplicados_eliminados} filas duplicadas eliminadas.")

    # --- 2. Imputación de nulos con mediana (columnas Feature_Vector) ---
    cols_a_imputar = [c for c in FEATURE_VECTOR_COLS if c in df.columns]

    # Detectar filas con al menos un nulo antes de imputar
    mask_con_nulos = df[cols_a_imputar].isnull().any(axis=1)
    filas_imputadas = int(mask_con_nulos.sum())

    for col in cols_a_imputar:
        if df[col].isnull().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            logger.debug(f"limpiar_dataset: columna '{col}' imputada con mediana={mediana}.")

    if filas_imputadas > 0:
        logger.info(f"limpiar_dataset: {filas_imputadas} filas imputadas con mediana.")

    # --- 3. Descarte de filas fuera de rango ---
    mask_valida = pd.Series(True, index=df.index)

    # Posesión: [0, 100]
    for col in _RANGO_POSESION:
        if col in df.columns:
            mask_valida &= df[col].between(0, 100, inclusive="both")

    # Tiros, faltas, tarjetas: >= 0
    for col in _RANGO_NO_NEGATIVOS:
        if col in df.columns:
            mask_valida &= df[col] >= 0

    # Resultado: {-1, 0, 1}
    if "resultado" in df.columns:
        mask_valida &= df["resultado"].isin(_VALORES_RESULTADO)

    filas_fuera_de_rango = int((~mask_valida).sum())
    if filas_fuera_de_rango > 0:
        logger.info(
            f"limpiar_dataset: {filas_fuera_de_rango} filas fuera de rango eliminadas."
        )

    df_limpio = df[mask_valida].reset_index(drop=True)

    resumen = {
        "filas_por_fuente": filas_originales,
        "duplicados_eliminados": duplicados_eliminados,
        "filas_fuera_de_rango": filas_fuera_de_rango,
        "filas_imputadas": filas_imputadas,
        "filas_finales": len(df_limpio),
    }

    logger.info(
        f"limpiar_dataset: {filas_originales} → {len(df_limpio)} filas finales. "
        f"Resumen: {resumen}"
    )

    return df_limpio, resumen


# ---------------------------------------------------------------------------
# validar_y_persistir
# ---------------------------------------------------------------------------

def validar_y_persistir(df: pd.DataFrame, ruta: str, min_filas: int = 50) -> None:
    """
    Persiste el dataset de forma atómica si tiene suficientes filas.

    Escribe primero a un archivo temporal en el mismo directorio y luego
    renombra (os.replace) para garantizar escritura atómica — nunca deja
    el archivo destino en estado parcialmente escrito.

    Args:
        df:        DataFrame limpio a persistir.
        ruta:      Ruta destino (p.ej. 'backend/data/historial_partidos.csv').
        min_filas: Número mínimo de filas requeridas para persistir (default 50).

    Raises:
        ValueError: Si el DataFrame tiene menos de min_filas filas.
    """
    n = len(df)

    if n < min_filas:
        msg = (
            f"El dataset tiene {n} filas válidas, pero se requieren al menos "
            f"{min_filas} para persistir en '{ruta}'. "
            "Abortando escritura para no sobrescribir datos existentes."
        )
        logger.error(f"validar_y_persistir: {msg}")
        raise ValueError(msg)

    # Escritura atómica: escribir a temporal y renombrar
    ruta_dir = os.path.dirname(os.path.abspath(ruta))
    os.makedirs(ruta_dir, exist_ok=True)

    fd, ruta_temporal = tempfile.mkstemp(dir=ruta_dir, suffix=".tmp.csv")
    try:
        os.close(fd)
        df.to_csv(ruta_temporal, index=False)
        os.replace(ruta_temporal, ruta)
        logger.info(
            f"validar_y_persistir: {n} filas persistidas exitosamente en '{ruta}'."
        )
    except Exception as exc:
        # Limpiar el temporal si algo falla antes del rename
        try:
            os.unlink(ruta_temporal)
        except OSError:
            pass
        logger.error(
            f"validar_y_persistir: error al escribir '{ruta}': {exc}"
        )
        raise
