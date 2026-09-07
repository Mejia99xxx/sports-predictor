import os
from datetime import date
import requests
from ml_model import preparar_datos_para_modelo

# API key loaded from environment variable; falls back to hardcoded value for local dev
API_KEY = os.environ.get("API_FOOTBALL_KEY", "3a5f469c7cb18934b76eb4818ed42a9a")
BASE_URL = "https://v3.football.api-sports.io"

headers = {
    "x-apisports-key": API_KEY
}

def obtener_partidos_del_dia():
    hoy = date.today().isoformat()
    url = f"{BASE_URL}/fixtures?date={hoy}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("response", [])
    return []

def obtener_ligas():
    """Return all leagues from the API."""
    url = f"{BASE_URL}/leagues"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return {}


def _temporadas_disponibles(league_id: int) -> list:
    """Return list of available season years for a league, sorted descending."""
    url = f"{BASE_URL}/leagues?id={league_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json().get("response", [])
        if not data:
            return []
        seasons = data[0].get("seasons", [])
        return sorted([int(s["year"]) for s in seasons], reverse=True)
    except Exception:
        return []


def obtener_equipos(league_id):
    """
    Fetch teams for a league using its most recent available season that
    actually has team data. Note: the free API plan does NOT support last=,
    so we fetch all fixtures for a season without that parameter.
    """
    seasons = _temporadas_disponibles(int(league_id))
    # Fallback if API call for seasons fails
    if not seasons:
        seasons = list(range(2026, 2018, -1))

    for year in seasons:
        url = f"{BASE_URL}/teams?league={league_id}&season={year}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("response"):
                    return data
        except Exception:
            continue
    return {}


def obtener_partidos_por_equipo(team_id):
    """
    Fetch completed fixtures for a team across multiple leagues and seasons.
    The free API plan blocks the &last= parameter, so we fetch all fixtures
    for a season and filter client-side for completed ones (status=FT).

    Returns a dict with 'response' key containing a list of completed fixtures
    sorted by date descending (most recent first), up to 5 entries.
    """
    # Leagues to search: club competitions + national team competitions
    # For national teams we also try Copa America (9), WC Qual CONMEBOL (34),
    # UEFA Nations League (5), EURO (4), AFCON (6), Copa America (9)
    national_leagues = [1, 4, 5, 6, 9, 10, 34, 29, 30, 31, 32, 33]
    seasons_to_try = list(range(2025, 2021, -1))

    all_completed = []

    # First try: all seasons without a league filter (broad search)
    for season in seasons_to_try:
        if len(all_completed) >= 5:
            break
        try:
            url = f"{BASE_URL}/fixtures?team={team_id}&season={season}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            fixtures = resp.json().get("response", [])
            completed = [
                f for f in fixtures
                if f.get("fixture", {}).get("status", {}).get("short") == "FT"
            ]
            all_completed.extend(completed)
        except Exception:
            continue

    # If still nothing, try national team leagues explicitly
    if not all_completed:
        for league_id in national_leagues:
            if len(all_completed) >= 5:
                break
            for season in seasons_to_try:
                try:
                    url = f"{BASE_URL}/fixtures?team={team_id}&league={league_id}&season={season}"
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code != 200:
                        continue
                    fixtures = resp.json().get("response", [])
                    completed = [
                        f for f in fixtures
                        if f.get("fixture", {}).get("status", {}).get("short") == "FT"
                    ]
                    if completed:
                        all_completed.extend(completed)
                        break
                except Exception:
                    continue

    if not all_completed:
        return {}

    # Sort by date descending and keep the 5 most recent
    all_completed.sort(key=lambda f: f.get("fixture", {}).get("date", ""), reverse=True)
    return {"response": all_completed[:5]}

def predecir_resultado(equipo1_id, equipo2_id):
    data1 = obtener_partidos_por_equipo(equipo1_id)
    data2 = obtener_partidos_por_equipo(equipo2_id)

    def promedio_goles(data, team_id):
        goles = 0
        partidos = data['response']
        for match in partidos:
            if match['teams']['home']['id'] == team_id:
                goles += match['goals']['home']
            elif match['teams']['away']['id'] == team_id:
                goles += match['goals']['away']
        return goles / len(partidos) if partidos else 0

    goles_equipo1 = promedio_goles(data1, equipo1_id)
    goles_equipo2 = promedio_goles(data2, equipo2_id)

    if goles_equipo1 > goles_equipo2:
        ganador = "Equipo 1 gana"
    elif goles_equipo1 < goles_equipo2:
        ganador = "Equipo 2 gana"
    else:
        ganador = "Empate"

    return {
        "equipo1_promedio_goles": goles_equipo1,
        "equipo2_promedio_goles": goles_equipo2,
        "pronostico": ganador
    }
    
def obtener_estadisticas_partido(fixture_id):
    url = f"{BASE_URL}/fixtures/statistics?fixture={fixture_id}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("response", [])
    return []


def ensamblar_vector_prediccion(
    stats_local: list,
    stats_visitante: list,
) -> list:
    """
    Merge two raw stats lists (one per team) into a single combined list
    compatible with ``preparar_datos_para_modelo``, normalising each entry's
    ``team.name`` to ``"local"`` or ``"visitante"``, then return the resulting
    9-value prediction vector.

    Each element in *stats_local* / *stats_visitante* is a dict in the shape
    returned by ``obtener_estadisticas_partido``:
        {
            "team": {"id": ..., "name": "...", ...},
            "statistics": [{"type": "...", "value": ...}, ...]
        }

    If either list is empty the function still delegates to
    ``preparar_datos_para_modelo``, which will fill every missing field with 0.
    """
    import copy

    def _normalise(stats: list, label: str) -> list:
        """Return a deep-copied list with every entry's team name set to *label*."""
        normalised = []
        for entry in stats:
            entry_copy = copy.deepcopy(entry)
            if "team" not in entry_copy:
                entry_copy["team"] = {}
            entry_copy["team"]["name"] = label
            normalised.append(entry_copy)
        return normalised

    combined = _normalise(stats_local, "local") + _normalise(stats_visitante, "visitante")
    return preparar_datos_para_modelo(combined)


def obtener_ultimo_partido_stats(team_id: int) -> list:
    """
    Returns the statistics list from the most recent completed fixture
    that actually has statistics available for the given team_id.
    Returns [] if no fixtures with stats are found or on any failure.

    Requirements: 2.1, 2.2, 2.3, 2.4
    """
    try:
        data = obtener_partidos_por_equipo(team_id)
        fixtures = data.get("response", []) if isinstance(data, dict) else []

        if not fixtures:
            return []

        # Sort by date descending so that the most recent fixture is tried first.
        # (obtener_partidos_por_equipo already does this, but we sort here too
        # so the function behaves correctly even when called with unsorted data.)
        fixtures_sorted = sorted(
            fixtures,
            key=lambda f: f.get("fixture", {}).get("date", ""),
            reverse=True,
        )

        for fixture in fixtures_sorted:
            fixture_id = fixture.get("fixture", {}).get("id")
            if fixture_id is None:
                continue
            stats = obtener_estadisticas_partido(fixture_id)
            if stats:
                return stats

        return []

    except Exception:
        return []

