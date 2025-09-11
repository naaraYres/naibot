#apy keys y credenciales
import os
from typing import Dict

def get_api_credentials() -> Dict[str, str]:
    """Credenciales de la API de Deriv."""
    return {
        "app_id": os.getenv("DERIV_APP_ID", "1089"),
        "token": os.getenv("DERIV_TOKEN", "C0fGduROGsa9kQD"),
        "symbol": os.getenv("DERIV_SYMBOL", "frxEURUSD"),
    }
