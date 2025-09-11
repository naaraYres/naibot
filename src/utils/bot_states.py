from enum import Enum

class BotState(Enum):
    """Estados del bot para mejor control de flujo."""
    BUSCANDO_SENAL = "buscando_senal"
    ESPERANDO_COMPRA = "esperando_respuesta_compra"
    OPERACION_ABIERTA = "operacion_abierta"
    DETENIDO = "detenido"
    ERROR = "error"