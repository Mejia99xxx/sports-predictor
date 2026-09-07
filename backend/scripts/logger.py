"""
Configuración centralizada de logging para el pipeline de ML.
"""
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger configurado con formato estándar.

    Args:
        name: Nombre del logger (normalmente __name__ del módulo).

    Returns:
        Logger configurado con handler de consola y formato estándar.
    """
    logger = logging.getLogger(name)

    # Evitar duplicar handlers si el logger ya fue configurado
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
