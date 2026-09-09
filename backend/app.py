# app.py (Backend con Flask)
import os
from flask import Flask, jsonify, request, render_template, redirect, url_for
from database import init_db
from models import db
from services import (
    obtener_ligas,
    obtener_equipos,
    obtener_partidos_por_equipo,
    obtener_partidos_del_dia,
    obtener_estadisticas_partido,
    obtener_ultimo_partido_stats,
    ensamblar_vector_prediccion,
)
from ml_model import cargar_modelo, predecir as predecir_modelo, preparar_datos_para_modelo


app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# 🔧 CONFIGURACIÓN DE LA BASE DE DATOS
# En producción usa DATABASE_URL si está definida, sino SQLite local
database_url = os.environ.get('DATABASE_URL', 'sqlite:///futbol.db')
app.config['SQLALCHEMY_DATABASE_URI'] = database_url

# Inicializar base de datos
# Inicializar base de datos
db.init_app(app)
init_db(app)


# Redirigir raíz a /inicio
@app.route('/')
def index():
    return redirect(url_for('inicio'))

# Ruta principal web (frontend)
@app.route('/inicio')
def inicio():
    return render_template("index.html")

@app.route("/partidos")
def mostrar_partidos():
    # Partidos now loaded client-side — pass empty list as default
    return render_template("partidos.html", partidos=[])

@app.route("/api-config")
def api_config():
    """Expose API key to frontend so it can call API-Football directly."""
    return jsonify({
        "api_key": os.environ.get("API_FOOTBALL_KEY", "3a5f469c7cb18934b76eb4818ed42a9a"),
        "base_url": "https://v3.football.api-sports.io"
    })

modelo, scaler = None, None
try:
    modelo, scaler = cargar_modelo()
except FileNotFoundError as e:
    app.logger.error(f"No se pudo cargar el modelo: {e}")

@app.route("/predecir", methods=["POST"])
def predecir():
    if modelo is None or scaler is None:
        return "El modelo no está disponible. Ejecute el entrenamiento primero.", 503

    modo = request.form.get("modo")

    if modo == "manual":
        campos = [
            "posesion_local", "posesion_visitante",
            "tiros_local", "tiros_visitante",
            "faltas_local", "faltas_visitante",
            "tarjetas_local", "tarjetas_visitante",
        ]
        entrada_modelo = []
        for campo in campos:
            valor_str = request.form.get(campo, "").strip()
            try:
                entrada_modelo.append(float(valor_str))
            except (ValueError, TypeError):
                return f"El campo '{campo}' es inválido o está vacío.", 400
        entrada_modelo.append(0.0)  # diferencia_goles por defecto
    else:
        fixture_id = request.form.get("fixture_id")
        stats = obtener_estadisticas_partido(fixture_id)
        if not stats:
            return "No se pudieron obtener estadísticas del partido.", 400
        entrada_modelo = preparar_datos_para_modelo(stats)

    prediccion = predecir_modelo(modelo, scaler, entrada_modelo)
    return render_template("resultado.html", resultado=prediccion)



@app.route("/predecir/vector", methods=["POST"])
def predecir_vector():
    """
    Receives a pre-computed prediction vector from the frontend
    (after it fetched stats directly from API-Football) and returns prediction.
    """
    if modelo is None or scaler is None:
        return jsonify({"error": "El modelo no está disponible."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Se requiere JSON con el vector."}), 400

    vector = data.get("vector")
    equipo_local = data.get("equipo_local", "Local")
    equipo_visitante = data.get("equipo_visitante", "Visitante")

    if not vector or len(vector) != 9:
        return jsonify({"error": "El vector debe tener exactamente 9 valores."}), 400

    try:
        resultado = predecir_modelo(modelo, scaler, vector)
        return jsonify({
            "resultado": resultado,
            "equipo_local": equipo_local,
            "equipo_visitante": equipo_visitante
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
def predecir_equipos():
    # 1. Validate required parameters
    local_team_id = request.form.get("local_team_id")
    visitante_team_id = request.form.get("visitante_team_id")

    if not local_team_id:
        return "Falta el parámetro: local_team_id", 400
    if not visitante_team_id:
        return "Falta el parámetro: visitante_team_id", 400

    # 2. Reject same-team requests
    if local_team_id == visitante_team_id:
        return "Los equipos local y visitante deben ser diferentes", 400

    # 3. Model availability check
    if modelo is None or scaler is None:
        return "El modelo no está disponible. Ejecute el entrenamiento primero.", 503

    # 4. Retrieve team display names (optional — fall back to IDs)
    equipo_local = request.form.get("local_team_name") or local_team_id
    equipo_visitante = request.form.get("visitante_team_name") or visitante_team_id

    # 5. Fetch statistics for each team
    try:
        local_team_id_int = int(local_team_id)
    except (ValueError, TypeError):
        return "Falta el parámetro: local_team_id", 400

    try:
        visitante_team_id_int = int(visitante_team_id)
    except (ValueError, TypeError):
        return "Falta el parámetro: visitante_team_id", 400

    stats_local = obtener_ultimo_partido_stats(local_team_id_int)
    if not stats_local:
        return f"No hay partidos recientes disponibles para {equipo_local}", 400

    stats_visitante = obtener_ultimo_partido_stats(visitante_team_id_int)
    if not stats_visitante:
        return f"No hay partidos recientes disponibles para {equipo_visitante}", 400

    # 6. Assemble prediction vector and run the model
    try:
        vector = ensamblar_vector_prediccion(stats_local, stats_visitante)
        resultado = predecir_modelo(modelo, scaler, vector)
    except Exception as e:
        app.logger.error(f"Error en predicción por equipos: {e}")
        return f"Error al procesar la predicción: {e}", 400

    return render_template(
        "resultado.html",
        resultado=resultado,
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
    )


# =============================
# 📡 Rutas de API-Football
# =============================
@app.route('/ligas')
def ligas():
    """
    Returns a hardcoded list of popular leagues.
    The API-Football free plan blocks requests from cloud servers,
    so we use a static list to ensure the dropdown always works.
    """
    ligas_data = [
        # Mundial y selecciones
        {"league": {"id": 1,   "name": "World Cup"},             "country": {"name": "World"}},
        {"league": {"id": 4,   "name": "Euro Championship"},     "country": {"name": "Europe"}},
        {"league": {"id": 5,   "name": "UEFA Nations League"},   "country": {"name": "Europe"}},
        {"league": {"id": 9,   "name": "Copa América"},          "country": {"name": "South America"}},
        {"league": {"id": 29,  "name": "AFC Asian Cup"},         "country": {"name": "Asia"}},
        {"league": {"id": 6,   "name": "Africa Cup of Nations"}, "country": {"name": "Africa"}},
        {"league": {"id": 34,  "name": "WC Qualification CONMEBOL"}, "country": {"name": "South America"}},
        # Europa - Top 5 ligas
        {"league": {"id": 39,  "name": "Premier League"},        "country": {"name": "England"}},
        {"league": {"id": 140, "name": "La Liga"},               "country": {"name": "Spain"}},
        {"league": {"id": 78,  "name": "Bundesliga"},            "country": {"name": "Germany"}},
        {"league": {"id": 135, "name": "Serie A"},               "country": {"name": "Italy"}},
        {"league": {"id": 61,  "name": "Ligue 1"},               "country": {"name": "France"}},
        # Europa - Otras ligas
        {"league": {"id": 2,   "name": "UEFA Champions League"}, "country": {"name": "Europe"}},
        {"league": {"id": 3,   "name": "UEFA Europa League"},    "country": {"name": "Europe"}},
        {"league": {"id": 848, "name": "UEFA Conference League"},"country": {"name": "Europe"}},
        {"league": {"id": 88,  "name": "Eredivisie"},            "country": {"name": "Netherlands"}},
        {"league": {"id": 94,  "name": "Primeira Liga"},         "country": {"name": "Portugal"}},
        {"league": {"id": 144, "name": "Pro League"},            "country": {"name": "Belgium"}},
        {"league": {"id": 103, "name": "Eliteserien"},           "country": {"name": "Norway"}},
        {"league": {"id": 113, "name": "Allsvenskan"},           "country": {"name": "Sweden"}},
        {"league": {"id": 119, "name": "Superliga"},             "country": {"name": "Denmark"}},
        # América
        {"league": {"id": 71,  "name": "Série A"},               "country": {"name": "Brazil"}},
        {"league": {"id": 128, "name": "Liga Profesional"},      "country": {"name": "Argentina"}},
        {"league": {"id": 262, "name": "Liga MX"},               "country": {"name": "Mexico"}},
        {"league": {"id": 239, "name": "Primera División"},      "country": {"name": "Colombia"}},
        {"league": {"id": 265, "name": "Primera División"},      "country": {"name": "Chile"}},
        {"league": {"id": 268, "name": "Primera División"},      "country": {"name": "Peru"}},
        {"league": {"id": 255, "name": "MLS"},                   "country": {"name": "USA"}},
        # CONMEBOL competiciones
        {"league": {"id": 13,  "name": "Copa Libertadores"},     "country": {"name": "South America"}},
        {"league": {"id": 11,  "name": "Copa Sudamericana"},     "country": {"name": "South America"}},
        # Asia / Resto
        {"league": {"id": 169, "name": "K League 1"},            "country": {"name": "South Korea"}},
        {"league": {"id": 98,  "name": "J1 League"},             "country": {"name": "Japan"}},
        {"league": {"id": 307, "name": "Saudi Pro League"},      "country": {"name": "Saudi Arabia"}},
    ]
    return jsonify({"response": ligas_data})

@app.route('/equipos')
def equipos():
    league_id = request.args.get('league_id')
    if not league_id:
        return jsonify({"response": []})

    # Try live API first
    data = obtener_equipos(league_id)
    teams = data.get("response", []) if isinstance(data, dict) else []

    # Fall back to hardcoded teams if API returns nothing (blocked from cloud servers)
    if not teams:
        teams = _get_equipos_hardcoded(int(league_id))

    return jsonify({"response": teams})


def _get_equipos_hardcoded(league_id: int) -> list:
    """Hardcoded teams for popular leagues as fallback when API is blocked."""
    equipos = {
        # Premier League
        39: [
            {"team": {"id": 33, "name": "Manchester United"}},
            {"team": {"id": 34, "name": "Newcastle"}},
            {"team": {"id": 40, "name": "Liverpool"}},
            {"team": {"id": 42, "name": "Arsenal"}},
            {"team": {"id": 49, "name": "Chelsea"}},
            {"team": {"id": 50, "name": "Manchester City"}},
            {"team": {"id": 47, "name": "Tottenham"}},
            {"team": {"id": 66, "name": "Aston Villa"}},
            {"team": {"id": 51, "name": "Brighton"}},
            {"team": {"id": 55, "name": "Brentford"}},
            {"team": {"id": 45, "name": "Everton"}},
            {"team": {"id": 48, "name": "West Ham"}},
            {"team": {"id": 52, "name": "Crystal Palace"}},
            {"team": {"id": 35, "name": "Bournemouth"}},
            {"team": {"id": 36, "name": "Fulham"}},
            {"team": {"id": 65, "name": "Nottingham Forest"}},
            {"team": {"id": 46, "name": "Leicester"}},
            {"team": {"id": 44, "name": "Wolverhampton"}},
            {"team": {"id": 41, "name": "Southampton"}},
            {"team": {"id": 57, "name": "Ipswich"}},
        ],
        # La Liga
        140: [
            {"team": {"id": 541, "name": "Real Madrid"}},
            {"team": {"id": 529, "name": "Barcelona"}},
            {"team": {"id": 530, "name": "Atletico Madrid"}},
            {"team": {"id": 532, "name": "Valencia"}},
            {"team": {"id": 533, "name": "Villarreal"}},
            {"team": {"id": 543, "name": "Real Betis"}},
            {"team": {"id": 536, "name": "Sevilla"}},
            {"team": {"id": 531, "name": "Athletic Club"}},
            {"team": {"id": 548, "name": "Real Sociedad"}},
            {"team": {"id": 538, "name": "Celta Vigo"}},
            {"team": {"id": 546, "name": "Getafe"}},
            {"team": {"id": 723, "name": "Osasuna"}},
            {"team": {"id": 542, "name": "Alaves"}},
            {"team": {"id": 547, "name": "Girona"}},
            {"team": {"id": 534, "name": "Las Palmas"}},
            {"team": {"id": 545, "name": "Espanyol"}},
            {"team": {"id": 727, "name": "Rayo Vallecano"}},
            {"team": {"id": 798, "name": "Leganes"}},
            {"team": {"id": 540, "name": "Mallorca"}},
            {"team": {"id": 539, "name": "Valladolid"}},
        ],
        # Bundesliga
        78: [
            {"team": {"id": 157, "name": "Bayern Munich"}},
            {"team": {"id": 165, "name": "Borussia Dortmund"}},
            {"team": {"id": 173, "name": "RB Leipzig"}},
            {"team": {"id": 168, "name": "Bayer Leverkusen"}},
            {"team": {"id": 169, "name": "Eintracht Frankfurt"}},
            {"team": {"id": 161, "name": "VfL Wolfsburg"}},
            {"team": {"id": 163, "name": "Borussia Monchengladbach"}},
            {"team": {"id": 167, "name": "1899 Hoffenheim"}},
            {"team": {"id": 172, "name": "VfB Stuttgart"}},
            {"team": {"id": 162, "name": "Werder Bremen"}},
            {"team": {"id": 176, "name": "VfL Bochum"}},
            {"team": {"id": 182, "name": "Union Berlin"}},
            {"team": {"id": 180, "name": "FC Augsburg"}},
            {"team": {"id": 170, "name": "SC Freiburg"}},
            {"team": {"id": 164, "name": "FSV Mainz 05"}},
            {"team": {"id": 192, "name": "FC Heidenheim"}},
            {"team": {"id": 175, "name": "Werder Bremen"}},
            {"team": {"id": 185, "name": "Holstein Kiel"}},
        ],
        # Serie A
        135: [
            {"team": {"id": 489, "name": "AC Milan"}},
            {"team": {"id": 496, "name": "Juventus"}},
            {"team": {"id": 505, "name": "Inter Milan"}},
            {"team": {"id": 492, "name": "Napoli"}},
            {"team": {"id": 487, "name": "Lazio"}},
            {"team": {"id": 497, "name": "AS Roma"}},
            {"team": {"id": 488, "name": "Atalanta"}},
            {"team": {"id": 502, "name": "Fiorentina"}},
            {"team": {"id": 500, "name": "Bologna"}},
            {"team": {"id": 494, "name": "Udinese"}},
            {"team": {"id": 491, "name": "Torino"}},
            {"team": {"id": 490, "name": "Cagliari"}},
            {"team": {"id": 499, "name": "Genoa"}},
            {"team": {"id": 504, "name": "Verona"}},
            {"team": {"id": 503, "name": "Lecce"}},
            {"team": {"id": 495, "name": "Empoli"}},
            {"team": {"id": 486, "name": "Parma"}},
            {"team": {"id": 493, "name": "Como"}},
            {"team": {"id": 506, "name": "Venezia"}},
            {"team": {"id": 511, "name": "Monza"}},
        ],
        # Ligue 1
        61: [
            {"team": {"id": 85,  "name": "Paris Saint-Germain"}},
            {"team": {"id": 80,  "name": "Lyon"}},
            {"team": {"id": 81,  "name": "Marseille"}},
            {"team": {"id": 82,  "name": "Monaco"}},
            {"team": {"id": 84,  "name": "Nice"}},
            {"team": {"id": 79,  "name": "Lille"}},
            {"team": {"id": 93,  "name": "Rennes"}},
            {"team": {"id": 94,  "name": "Lens"}},
            {"team": {"id": 95,  "name": "Strasbourg"}},
            {"team": {"id": 91,  "name": "Nantes"}},
            {"team": {"id": 97,  "name": "Toulouse"}},
            {"team": {"id": 96,  "name": "Reims"}},
            {"team": {"id": 108, "name": "Brest"}},
            {"team": {"id": 111, "name": "Montpellier"}},
            {"team": {"id": 112, "name": "Angers"}},
            {"team": {"id": 116, "name": "Saint-Etienne"}},
            {"team": {"id": 106, "name": "Auxerre"}},
            {"team": {"id": 107, "name": "Le Havre"}},
        ],
        # Champions League - top clubs
        2: [
            {"team": {"id": 541, "name": "Real Madrid"}},
            {"team": {"id": 157, "name": "Bayern Munich"}},
            {"team": {"id": 505, "name": "Inter Milan"}},
            {"team": {"id": 529, "name": "Barcelona"}},
            {"team": {"id": 40,  "name": "Liverpool"}},
            {"team": {"id": 49,  "name": "Chelsea"}},
            {"team": {"id": 50,  "name": "Manchester City"}},
            {"team": {"id": 165, "name": "Borussia Dortmund"}},
            {"team": {"id": 85,  "name": "Paris Saint-Germain"}},
            {"team": {"id": 488, "name": "Atalanta"}},
            {"team": {"id": 530, "name": "Atletico Madrid"}},
            {"team": {"id": 168, "name": "Bayer Leverkusen"}},
            {"team": {"id": 42,  "name": "Arsenal"}},
            {"team": {"id": 496, "name": "Juventus"}},
            {"team": {"id": 489, "name": "AC Milan"}},
            {"team": {"id": 173, "name": "RB Leipzig"}},
        ],
        # World Cup 2022 (32 selecciones)
        1: [
            {"team": {"id": 6,   "name": "Brazil"}},
            {"team": {"id": 26,  "name": "Argentina"}},
            {"team": {"id": 2,   "name": "France"}},
            {"team": {"id": 9,   "name": "Spain"}},
            {"team": {"id": 10,  "name": "England"}},
            {"team": {"id": 3,   "name": "Croatia"}},
            {"team": {"id": 768, "name": "Morocco"}},
            {"team": {"id": 24,  "name": "Portugal"}},
            {"team": {"id": 1,   "name": "Belgium"}},
            {"team": {"id": 25,  "name": "Netherlands"}},
            {"team": {"id": 7,   "name": "Uruguay"}},
            {"team": {"id": 14,  "name": "Serbia"}},
            {"team": {"id": 12,  "name": "Japan"}},
            {"team": {"id": 21,  "name": "USA"}},
            {"team": {"id": 13,  "name": "Senegal"}},
            {"team": {"id": 22,  "name": "Mexico"}},
            {"team": {"id": 15,  "name": "Poland"}},
            {"team": {"id": 27,  "name": "Ecuador"}},
            {"team": {"id": 18,  "name": "Denmark"}},
            {"team": {"id": 17,  "name": "Germany"}},
            {"team": {"id": 800, "name": "Australia"}},
            {"team": {"id": 798, "name": "South Korea"}},
            {"team": {"id": 28,  "name": "Cameroon"}},
            {"team": {"id": 29,  "name": "Canada"}},
            {"team": {"id": 767, "name": "Ghana"}},
            {"team": {"id": 766, "name": "Tunisia"}},
            {"team": {"id": 762, "name": "Costa Rica"}},
            {"team": {"id": 764, "name": "Saudi Arabia"}},
            {"team": {"id": 31,  "name": "Switzerland"}},
            {"team": {"id": 30,  "name": "Cameroon"}},
            {"team": {"id": 763, "name": "Qatar"}},
            {"team": {"id": 765, "name": "Iran"}},
        ],
        # Liga MX
        262: [
            {"team": {"id": 2283, "name": "Club America"}},
            {"team": {"id": 2282, "name": "Chivas Guadalajara"}},
            {"team": {"id": 2286, "name": "Cruz Azul"}},
            {"team": {"id": 2288, "name": "Pumas UNAM"}},
            {"team": {"id": 2287, "name": "Tigres UANL"}},
            {"team": {"id": 2285, "name": "Monterrey"}},
            {"team": {"id": 2280, "name": "Santos Laguna"}},
            {"team": {"id": 2284, "name": "Atlas"}},
            {"team": {"id": 2281, "name": "Leon"}},
            {"team": {"id": 2290, "name": "Toluca"}},
            {"team": {"id": 2291, "name": "Tijuana"}},
            {"team": {"id": 2292, "name": "Puebla"}},
            {"team": {"id": 2293, "name": "Necaxa"}},
            {"team": {"id": 2294, "name": "Queretaro"}},
            {"team": {"id": 2295, "name": "Juarez"}},
            {"team": {"id": 2289, "name": "Pachuca"}},
        ],
        # Brasileirao Serie A
        71: [
            {"team": {"id": 127, "name": "Flamengo"}},
            {"team": {"id": 121, "name": "Palmeiras"}},
            {"team": {"id": 126, "name": "Atletico Mineiro"}},
            {"team": {"id": 128, "name": "Fluminense"}},
            {"team": {"id": 119, "name": "Corinthians"}},
            {"team": {"id": 118, "name": "Santos"}},
            {"team": {"id": 120, "name": "Sao Paulo"}},
            {"team": {"id": 130, "name": "Vasco da Gama"}},
            {"team": {"id": 131, "name": "Botafogo"}},
            {"team": {"id": 125, "name": "Internacional"}},
            {"team": {"id": 116, "name": "Gremio"}},
            {"team": {"id": 117, "name": "Cruzeiro"}},
            {"team": {"id": 133, "name": "Bahia"}},
            {"team": {"id": 135, "name": "Fortaleza"}},
            {"team": {"id": 136, "name": "Atletico Goianiense"}},
            {"team": {"id": 137, "name": "Ceara"}},
            {"team": {"id": 138, "name": "Cuiaba"}},
            {"team": {"id": 139, "name": "Sport Recife"}},
            {"team": {"id": 140, "name": "Bragantino"}},
            {"team": {"id": 141, "name": "Juventude"}},
        ],
        # Copa America
        9: [
            {"team": {"id": 26,  "name": "Argentina"}},
            {"team": {"id": 6,   "name": "Brazil"}},
            {"team": {"id": 7,   "name": "Uruguay"}},
            {"team": {"id": 27,  "name": "Ecuador"}},
            {"team": {"id": 28,  "name": "Colombia"}},
            {"team": {"id": 29,  "name": "Venezuela"}},
            {"team": {"id": 30,  "name": "Peru"}},
            {"team": {"id": 31,  "name": "Chile"}},
            {"team": {"id": 32,  "name": "Bolivia"}},
            {"team": {"id": 33,  "name": "Paraguay"}},
            {"team": {"id": 21,  "name": "USA"}},
            {"team": {"id": 34,  "name": "Mexico"}},
            {"team": {"id": 35,  "name": "Panama"}},
            {"team": {"id": 36,  "name": "Jamaica"}},
            {"team": {"id": 37,  "name": "Costa Rica"}},
            {"team": {"id": 38,  "name": "Canada"}},
        ],
    }
    return equipos.get(league_id, [])

@app.route('/pronostico')
def pronostico():
    equipo1_id = request.args.get('equipo1_id')
    equipo2_id = request.args.get('equipo2_id')
    return jsonify({"message": "Funcionalidad de pronóstico personalizada no implementada aún."})

# =============================
# 🤖 Ruta del modelo de ML
# =============================
@app.route('/ml_pronostico', methods=['POST'])
def ml_pronostico():
    if modelo is None or scaler is None:
        return jsonify({"error": "Modelo no disponible"}), 503
    data = request.get_json()
    try:
        resultado = predecir_modelo(modelo, scaler, data)
        return jsonify({"resultado": resultado})
    except Exception as e:
        app.logger.error(str(e))
        return jsonify({"error": str(e)}), 500


# =============================
# 🚀 Iniciar servidor
# =============================
if __name__ == '__main__':
    app.run(debug=True)
