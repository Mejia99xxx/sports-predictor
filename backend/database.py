# database.py (Conexión a MySQL con SQLAlchemy)
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app):
    # NO vuelvas a llamar db.init_app(app) aquí
    with app.app_context():
        db.create_all()
        print("✅ Base de datos inicializada correctamente.")
