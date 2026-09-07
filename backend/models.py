from datetime import datetime
from database import db




class Equipo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    liga = db.Column(db.String(100), nullable=False)

    jugadores = db.relationship('Jugador', backref='equipo', lazy=True)
    estadisticas = db.relationship('EstadisticaPartido', backref='equipo', lazy=True)

class Jugador(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    equipo_id = db.Column(db.Integer, db.ForeignKey('equipo.id'), nullable=False)
    posicion = db.Column(db.String(50), nullable=False)
    goles = db.Column(db.Integer, default=0)
    asistencias = db.Column(db.Integer, default=0)

class Partido(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fecha = db.Column(db.Date, nullable=False)
    equipo_local_id = db.Column(db.Integer, db.ForeignKey('equipo.id'), nullable=False)
    equipo_visitante_id = db.Column(db.Integer, db.ForeignKey('equipo.id'), nullable=False)
    resultado = db.Column(db.String(10))  # Ej: "1-2", "Empate"

    estadisticas = db.relationship('EstadisticaPartido', backref='partido', lazy=True)

class EstadisticaPartido(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    partido_id = db.Column(db.Integer, db.ForeignKey('partido.id'), nullable=False)
    equipo_id = db.Column(db.Integer, db.ForeignKey('equipo.id'), nullable=False)
    goles = db.Column(db.Integer, default=0)
    posesion = db.Column(db.Float, default=0)
    tiros_al_arco = db.Column(db.Integer, default=0)
    tarjetas = db.Column(db.Integer, default=0)
    faltas = db.Column(db.Integer, default=0)

class Pronostico(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fecha = db.Column(db.DateTime, default=datetime.utcnow)
    metodo = db.Column(db.String(50))  # 'ML' o 'API'
    datos_entrada = db.Column(db.Text)
    resultado = db.Column(db.String(100))
