"""
Data Collector — pipeline de mejora del modelo de predicción ML.

Responsable de:
- Descargar datos históricos de partidos desde datasets públicos en CSV.
- Recolectar estadísticas de partidos desde la API-Football (plan gratuito).

Este módulo opera de forma completamente independiente de la aplicación Flask.
"""

import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from scripts.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

API_BASE_URL = "https://v3.football.api-sports.io"
REQUEST_TIMEOUT = 30  # segundos (timeout por petición, 30s según especificación)

# Columnas mínimas requeridas para validar el CSV de football-data.co.uk
# HST/AST son opcionales (con fallback a HS/AS); HomeTeam/AwayTeam son obligatorias
_COLUMNAS_REQUERIDAS_CSV = {"FTHG", "FTAG", "HF", "AF", "HY", "AY", "HomeTeam", "AwayTeam"}

# Fuentes públicas por defecto (football-data.co.uk)
FUENTES_DEFAULT = [
    {
        "url": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "liga": "premier_league",
        "temporada": "2024",
    },
    {
        "url": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
        "liga": "la_liga",
        "temporada": "2024",
    },
    {
        "url": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
        "liga": "bundesliga",
        "temporada": "2024",
    },
    {
        "url": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
        "liga": "serie_a",
        "temporada": "2024",
    },
    {
        "url": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
        "liga": "ligue_1",
        "temporada": "2024",
    },
]

# Columnas del esquema final de historial_partidos.csv
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


# ---------------------------------------------------------------------------
# Descarga de CSVs públicos (Tarea 2.1)
# ---------------------------------------------------------------------------

def descargar_csv_publicos(fuentes: list) -> list:
    """
    Descarga CSVs desde las URLs configuradas.

    Cada fuente es un dict con claves: 'url', 'liga', 'temporada'.
    Retorna lista de DataFrames crudos con el esquema del Feature_Vector.
    Omite fuentes que fallen con error de red o columnas insuficientes.

    Args:
        fuentes: Lista de dicts con claves 'url', 'liga', 'temporada'.

    Returns:
        Lista de DataFrames con columnas del Feature_Vector extraídas.
    """
    resultados = []

    for fuente in fuentes:
        url = fuente.get("url", "")
        liga = fuente.get("liga", "desconocida")
        temporada = fuente.get("temporada", "desconocida")

        # --- Descarga con timeout de 30 segundos ---
        try:
            logger.info(f"Descargando CSV: {url} (liga={liga}, temporada={temporada})")
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"❌ Error descargando {url}: {e}")
            continue
        except ValueError as e:
            logger.error(f"❌ URL inválida {url}: {e}")
            continue

        # --- Persistir CSV crudo en backend/data/raw/ ---
        ruta_raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        os.makedirs(ruta_raw_dir, exist_ok=True)
        ruta_raw = os.path.join(ruta_raw_dir, f"csv_raw_{liga}_{temporada}.csv")
        try:
            with open(ruta_raw, "w", encoding="utf-8") as f:
                f.write(response.text)
            logger.info(f"💾 CSV crudo guardado en {ruta_raw}")
        except IOError as e:
            logger.error(f"❌ No se pudo persistir el CSV crudo {ruta_raw}: {e}")

        # --- Parsear CSV ---
        try:
            from io import StringIO
            df_crudo = pd.read_csv(StringIO(response.text))
        except ValueError as e:
            logger.error(f"❌ Error al parsear CSV de {url}: {e}")
            continue

        # --- Validar columnas mínimas requeridas ---
        columnas_presentes = set(df_crudo.columns)
        columnas_faltantes = _COLUMNAS_REQUERIDAS_CSV - columnas_presentes
        if columnas_faltantes:
            logger.warning(
                f"⚠️ CSV de {url} omitido: columnas faltantes {sorted(columnas_faltantes)}"
            )
            continue

        # --- Validar que existan filas de datos ---
        if df_crudo.empty:
            logger.warning(
                f"⚠️ CSV de {url} no tiene filas de datos. Se omite."
            )
            continue

        # --- Mapear columnas al Feature_Vector ---
        try:
            df_mapeado = _mapear_columnas_csv(df_crudo, liga, temporada)
        except (KeyError, ValueError) as e:
            logger.error(f"❌ Error al mapear columnas de {url}: {e}")
            continue

        # --- Validar filas tras el mapeo ---
        if df_mapeado.empty:
            logger.warning(
                f"⚠️ CSV de {url} resultó vacío tras el mapeo de columnas. Se omite."
            )
            continue

        resultados.append(df_mapeado)
        logger.info(
            f"✅ CSV descargado: {len(df_mapeado)} filas (liga={liga}, temporada={temporada})"
        )

    return resultados


def _mapear_columnas_csv(df: pd.DataFrame, liga: str, temporada: str) -> pd.DataFrame:
    """
    Mapea columnas de football-data.co.uk al esquema del Feature_Vector.

    Mapping:
        FTHG     → goles_local
        FTAG     → goles_visitante
        HST      → tiros_local  (fallback: HS si HST no está presente)
        AST      → tiros_visitante  (fallback: AS si AST no está presente)
        HF       → faltas_local
        AF       → faltas_visitante
        HY       → tarjetas_local
        AY       → tarjetas_visitante
        HomeTeam → equipo_local
        AwayTeam → equipo_visitante
        FTHG - FTAG → diferencia_goles
        sign(FTHG - FTAG) → resultado
        posesion_local / posesion_visitante se imputan con 50.0
    """
    import numpy as np

    resultado_df = pd.DataFrame()

    resultado_df["equipo_local"] = df["HomeTeam"].values
    resultado_df["equipo_visitante"] = df["AwayTeam"].values

    resultado_df["goles_local"] = pd.to_numeric(df["FTHG"], errors="coerce")
    resultado_df["goles_visitante"] = pd.to_numeric(df["FTAG"], errors="coerce")

    # Posesión no disponible en football-data.co.uk → imputar con 50.0
    resultado_df["posesion_local"] = 50.0
    resultado_df["posesion_visitante"] = 50.0

    # tiros: HST preferido, con fallback a HS
    if "HST" in df.columns:
        resultado_df["tiros_local"] = pd.to_numeric(df["HST"], errors="coerce")
    elif "HS" in df.columns:
        resultado_df["tiros_local"] = pd.to_numeric(df["HS"], errors="coerce")
    else:
        resultado_df["tiros_local"] = float("nan")

    # tiros visitante: AST preferido, con fallback a AS
    if "AST" in df.columns:
        resultado_df["tiros_visitante"] = pd.to_numeric(df["AST"], errors="coerce")
    elif "AS" in df.columns:
        resultado_df["tiros_visitante"] = pd.to_numeric(df["AS"], errors="coerce")
    else:
        resultado_df["tiros_visitante"] = float("nan")

    resultado_df["faltas_local"] = pd.to_numeric(df["HF"], errors="coerce")
    resultado_df["faltas_visitante"] = pd.to_numeric(df["AF"], errors="coerce")
    resultado_df["tarjetas_local"] = pd.to_numeric(df["HY"], errors="coerce")
    resultado_df["tarjetas_visitante"] = pd.to_numeric(df["AY"], errors="coerce")

    diferencia = resultado_df["goles_local"] - resultado_df["goles_visitante"]
    resultado_df["diferencia_goles"] = diferencia
    resultado_df["resultado"] = np.sign(diferencia).astype("Int64")

    return resultado_df[COLUMNAS_ESQUEMA]


def _persistir_csv_crudo(df: pd.DataFrame, liga: str, temporada: str) -> None:
    """Persiste el CSV crudo en backend/data/raw/."""
    ruta_raw = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw"
    )
    os.makedirs(ruta_raw, exist_ok=True)
    nombre_archivo = f"csv_raw_{liga}_{temporada}.csv"
    ruta_completa = os.path.join(ruta_raw, nombre_archivo)
    try:
        df.to_csv(ruta_completa, index=False)
        logger.info(f"CSV crudo persistido en {ruta_completa}")
    except IOError as e:
        logger.error(f"❌ Error persistiendo CSV crudo en {ruta_completa}: {e}")


# ---------------------------------------------------------------------------
# Obtención de fixture IDs (Tarea 2.3)
# ---------------------------------------------------------------------------

def obtener_fixture_ids_por_temporada(
    league_id: int, season: int, api_key: str
) -> list:
    """
    Retorna la lista de IDs de fixtures completados para una liga y temporada.

    Llama al endpoint:
        GET https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}&status=FT

    Nota: El plan gratuito de API-Football no soporta el parámetro &last=,
    por lo que se usa el filtro status=FT para obtener solo partidos terminados.

    Args:
        league_id: ID de la liga en API-Football.
        season:    Año de la temporada (ej. 2023).
        api_key:   Clave de autenticación de la API.

    Returns:
        Lista de IDs de fixtures (enteros). Retorna [] ante cualquier error.
    """
    url = f"{API_BASE_URL}/fixtures"
    params = {
        "league": league_id,
        "season": season,
        "status": "FT",
    }
    headers = {"x-apisports-key": api_key}

    try:
        logger.info(
            f"Obteniendo fixture IDs: league_id={league_id}, season={season}"
        )
        response = requests.get(
            url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            logger.error(
                f"❌ Error HTTP {response.status_code} al obtener fixtures "
                f"(league_id={league_id}, season={season}): {response.text[:200]}"
            )
            return []

        data = response.json()
        fixtures = data.get("response", [])
        ids = [
            int(fixture["fixture"]["id"])
            for fixture in fixtures
            if fixture.get("fixture", {}).get("id") is not None
        ]

        logger.info(
            f"✅ {len(ids)} fixture IDs obtenidos "
            f"(league_id={league_id}, season={season})"
        )
        return ids

    except requests.exceptions.ConnectionError as e:
        logger.error(
            f"❌ Error de conexión al obtener fixtures "
            f"(league_id={league_id}, season={season}): {e}"
        )
        return []
    except requests.exceptions.Timeout as e:
        logger.error(
            f"❌ Timeout al obtener fixtures "
            f"(league_id={league_id}, season={season}): {e}"
        )
        return []
    except requests.exceptions.RequestException as e:
        logger.error(
            f"❌ Error de red al obtener fixtures "
            f"(league_id={league_id}, season={season}): {e}"
        )
        return []


# ---------------------------------------------------------------------------
# Recolección desde API-Football (Tarea 2.4)
# ---------------------------------------------------------------------------

def recolectar_desde_api(
    fixture_ids: list, api_key: str, limite_diario: int = 99
) -> list:
    """
    Obtiene estadísticas de partidos desde API-Football.

    Se detiene antes de alcanzar limite_diario llamadas.
    Persiste cada respuesta cruda en backend/data/raw/api_raw_YYYYMMDD.json.
    Retorna lista de dicts con el esquema del Feature_Vector.

    Args:
        fixture_ids:   Lista de IDs de fixtures a procesar.
        api_key:       Clave de autenticación de la API.
        limite_diario: Máximo de llamadas permitidas (por defecto 99).

    Returns:
        Lista de dicts con campos del Feature_Vector.
    """
    headers = {"x-apisports-key": api_key}
    resultados = []
    respuestas_crudas = []
    llamadas_realizadas = 0

    for fixture_id in fixture_ids:
        if llamadas_realizadas >= limite_diario:
            logger.warning(
                f"⚠️ Límite de API alcanzado ({limite_diario} llamadas). "
                "Deteniendo recolección."
            )
            break

        if llamadas_realizadas >= 90:
            logger.warning(
                f"⚠️ Aproximándose al límite: {llamadas_realizadas}/{limite_diario} llamadas."
            )

        url = f"{API_BASE_URL}/fixtures/statistics"
        params = {"fixture": fixture_id}

        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
            llamadas_realizadas += 1

            if response.status_code != 200:
                logger.error(
                    f"❌ Error HTTP {response.status_code} para fixture {fixture_id}: "
                    f"{response.text[:200]}"
                )
                continue

            data = response.json()
            stats_list = data.get("response", [])

            if not stats_list:
                logger.warning(f"⚠️ Sin estadísticas para fixture {fixture_id}")
                continue

            respuestas_crudas.append({"fixture_id": fixture_id, "response": stats_list})

            vector = _mapear_stats_api_a_feature_vector(stats_list, fixture_id)
            if vector:
                resultados.append(vector)

        except requests.RequestException as e:
            llamadas_realizadas += 1
            logger.error(f"❌ Error de red para fixture {fixture_id}: {e}")
            continue

    # Persistir respuestas crudas del día
    if respuestas_crudas:
        _persistir_respuestas_crudas_api(respuestas_crudas)

    logger.info(
        f"Recolección API finalizada: {llamadas_realizadas} llamadas realizadas, "
        f"{len(resultados)} vectores obtenidos."
    )
    return resultados


def _mapear_stats_api_a_feature_vector(
    stats_list: list, fixture_id: Optional[int] = None
) -> Optional[dict]:
    """
    Mapea la respuesta de estadísticas de la API-Football al esquema del Feature_Vector.

    Equivalente a la lógica de preparar_datos_para_modelo en ml_model.py.

    Args:
        stats_list: Lista de dicts de estadísticas por equipo (formato API-Football).
        fixture_id: ID del fixture (para logging).

    Returns:
        Dict con los 9 campos del Feature_Vector o None si el mapeo falla.
    """

    def _valor_stat(stats: list, tipo: str) -> float:
        """Extrae un valor numérico de la lista de estadísticas por tipo."""
        for stat in stats:
            if stat.get("type", "").lower() == tipo.lower():
                valor = stat.get("value")
                if valor is None:
                    return 0.0
                try:
                    # Algunos valores vienen como "45%" → extraer solo el número
                    if isinstance(valor, str):
                        valor = valor.replace("%", "").strip()
                    return float(valor)
                except (ValueError, TypeError):
                    return 0.0
        return 0.0

    # Identificar equipo local y visitante
    local = next(
        (e for e in stats_list if e.get("team", {}).get("name", "").lower() == "local"),
        None,
    )
    visitante = next(
        (
            e
            for e in stats_list
            if e.get("team", {}).get("name", "").lower() == "visitante"
        ),
        None,
    )

    # Fallback: si no hay etiquetas "local"/"visitante", usar posición en la lista
    if local is None and len(stats_list) >= 1:
        local = stats_list[0]
    if visitante is None and len(stats_list) >= 2:
        visitante = stats_list[1]

    if local is None or visitante is None:
        logger.warning(f"⚠️ No se pudo identificar ambos equipos para fixture {fixture_id}")
        return None

    stats_local = local.get("statistics", [])
    stats_visitante = visitante.get("statistics", [])

    posesion_local = _valor_stat(stats_local, "Ball Possession")
    posesion_visitante = _valor_stat(stats_visitante, "Ball Possession")
    tiros_local = _valor_stat(stats_local, "Total Shots")
    tiros_visitante = _valor_stat(stats_visitante, "Total Shots")
    faltas_local = _valor_stat(stats_local, "Fouls")
    faltas_visitante = _valor_stat(stats_visitante, "Fouls")
    tarjetas_local = _valor_stat(stats_local, "Yellow Cards")
    tarjetas_visitante = _valor_stat(stats_visitante, "Yellow Cards")

    goles_local = _valor_stat(stats_local, "Goals")
    goles_visitante = _valor_stat(stats_visitante, "Goals")
    diferencia_goles = goles_local - goles_visitante

    import math
    if diferencia_goles > 0:
        resultado = 1
    elif diferencia_goles < 0:
        resultado = -1
    else:
        resultado = 0

    return {
        "equipo_local": local.get("team", {}).get("name", ""),
        "equipo_visitante": visitante.get("team", {}).get("name", ""),
        "goles_local": goles_local,
        "goles_visitante": goles_visitante,
        "posesion_local": posesion_local,
        "posesion_visitante": posesion_visitante,
        "tiros_local": tiros_local,
        "tiros_visitante": tiros_visitante,
        "faltas_local": faltas_local,
        "faltas_visitante": faltas_visitante,
        "tarjetas_local": tarjetas_local,
        "tarjetas_visitante": tarjetas_visitante,
        "resultado": resultado,
        "diferencia_goles": diferencia_goles,
    }


def _persistir_respuestas_crudas_api(respuestas: list) -> None:
    """Persiste las respuestas crudas de la API en backend/data/raw/."""
    ruta_raw = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw"
    )
    os.makedirs(ruta_raw, exist_ok=True)
    fecha_hoy = datetime.now().strftime("%Y%m%d")
    nombre_archivo = f"api_raw_{fecha_hoy}.json"
    ruta_completa = os.path.join(ruta_raw, nombre_archivo)

    # Si ya existe el archivo del día, cargar y combinar
    datos_existentes = []
    if os.path.exists(ruta_completa):
        try:
            with open(ruta_completa, "r", encoding="utf-8") as f:
                datos_existentes = json.load(f)
        except (json.JSONDecodeError, IOError):
            datos_existentes = []

    datos_combinados = datos_existentes + respuestas

    try:
        with open(ruta_completa, "w", encoding="utf-8") as f:
            json.dump(datos_combinados, f, ensure_ascii=False, indent=2)
        logger.info(f"Respuestas crudas de la API persistidas en {ruta_completa}")
    except IOError as e:
        logger.error(f"❌ Error persistiendo respuestas crudas en {ruta_completa}: {e}")
