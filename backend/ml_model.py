import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def cargar_datos():
    try:
        df = pd.read_csv("data/historial_partidos.csv")
        df['diferencia_goles'] = df['goles_local'] - df['goles_visitante']
        return df
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return pd.DataFrame()

def entrenar_modelo():
    df = cargar_datos()
    if df.empty:
        raise ValueError("⚠️ No se pudieron cargar datos para el entrenamiento.")
    
    X = df[['posesion_local', 'posesion_visitante', 'tiros_local', 'tiros_visitante', 
            'faltas_local', 'faltas_visitante', 'tarjetas_local', 'tarjetas_visitante', 
            'diferencia_goles']]
    y = df['resultado']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    return modelo, scaler

def guardar_modelo(modelo, scaler, ruta="modelo.pkl"):
    with open(ruta, "wb") as f:
        pickle.dump((modelo, scaler), f)

def cargar_modelo(ruta="modelo.pkl"):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"El archivo de modelo '{ruta}' no existe. "
            "Ejecute entrenar_modelo() primero para generar el archivo."
        )
    with open(ruta, "rb") as f:
        return pickle.load(f)

def predecir(modelo, scaler, entrada):
    if not isinstance(entrada, (list, tuple)) or len(entrada) != 9:
        tipo = type(entrada).__name__
        longitud = len(entrada) if hasattr(entrada, '__len__') else '?'
        raise ValueError(
            f"La entrada debe ser una lista o tupla de exactamente 9 valores numéricos. "
            f"Se recibió: {tipo} con {longitud} elementos."
        )
    entrada_scaled = scaler.transform([list(entrada)])
    prediccion = modelo.predict(entrada_scaled)[0]

    if prediccion == 1:
        return "Gana equipo local"
    elif prediccion == -1:
        return "Gana equipo visitante"
    return "Empate"

def preparar_datos_para_modelo(stats):
    datos = {}

    for equipo in stats:
        nombre_equipo = equipo['team']['name'].lower()
        for stat in equipo['statistics']:
            tipo = stat['type'].replace(" ", "_").lower()
            valor = stat['value'] if isinstance(stat['value'], (int, float)) else 0
            clave = f"{nombre_equipo}_{tipo}"
            datos[clave] = valor

    diferencia_goles = datos.get("local_goals", 0) - datos.get("visitante_goals", 0) if "local_goals" in datos else 0

    entrada = [
        datos.get("local_ball_possession", 0),
        datos.get("visitante_ball_possession", 0),
        datos.get("local_total_shots", 0),
        datos.get("visitante_total_shots", 0),
        datos.get("local_fouls", 0),
        datos.get("visitante_fouls", 0),
        datos.get("local_yellow_cards", 0),
        datos.get("visitante_yellow_cards", 0),
        diferencia_goles
    ]
    return entrada
