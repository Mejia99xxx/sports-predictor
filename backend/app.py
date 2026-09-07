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
    # Ligas now loaded client-side via /ligas endpoint to avoid slow server-side API calls
    return render_template("index.html")

@app.route("/partidos")
def mostrar_partidos():
    partidos = obtener_partidos_del_dia()
    return render_template("partidos.html", partidos=partidos)

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



@app.route("/predecir/equipos", methods=["POST"])
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
    return jsonify(obtener_equipos(league_id))

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
