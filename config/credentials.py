"""
Template de credenciales - RENOMBRAR A credentials.py y completar
"""
import os
def get_credentials():
    return {
        "app_id": os.getenv("DERIV_APP_ID", "TU_APP_ID_AQUI"),
        "token": os.getenv("DERIV_TOKEN", "TU_TOKEN_AQUI"),
        "symbol": os.getenv("DERIV_SYMBOL", "frxEURUSD"),
        "stake": os.getenv("DERIV_STAKE", "1.00"),
    }

# TELEGRAM (opcional)
TELEGRAM_CONFIG = {
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", "")
}

# EMAIL (opcional)  
EMAIL_CONFIG = {
    "smtp_server": os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("EMAIL_SMTP_PORT", "587")),
    "username": os.getenv("EMAIL_USERNAME", ""),
    "password": os.getenv("EMAIL_PASSWORD", ""),
    "to_email": os.getenv("EMAIL_TO", "")
}