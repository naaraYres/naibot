"""
Sistema de notificaciones para el trading bot.
Soporta Telegram, Discord, email y notificaciones del sistema.
"""

import os
import json
import smtplib
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import asyncio
try:
    import plyer  # Para notificaciones del sistema
except ImportError:
    plyer = None


class NotificationType(Enum):
    """Tipos de notificaciones."""
    INFO = "info"
    SUCCESS = "success" 
    WARNING = "warning"
    ERROR = "error"
    TRADE = "trade"
    SYSTEM = "system"


class NotificationChannel(Enum):
    """Canales de notificación disponibles."""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    SYSTEM = "system"
    CONSOLE = "console"


class NotificationManager:
    """Gestor principal de notificaciones."""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa el gestor de notificaciones."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Configuración de canales
        self.enabled_channels = config.get('enabled_channels', [NotificationChannel.CONSOLE.value])
        
        # Configuraciones específicas
        self.telegram_config = config.get('telegram', {})
        self.discord_config = config.get('discord', {})
        self.email_config = config.get('email', {})
        
        # Filtros por tipo de notificación
        self.notification_filters = config.get('filters', {
            NotificationType.ERROR.value: [NotificationChannel.TELEGRAM.value, NotificationChannel.EMAIL.value],
            NotificationType.TRADE.value: [NotificationChannel.TELEGRAM.value, NotificationChannel.DISCORD.value],
            NotificationType.WARNING.value: [NotificationChannel.TELEGRAM.value],
            NotificationType.SUCCESS.value: [NotificationChannel.TELEGRAM.value],
            NotificationType.INFO.value: [NotificationChannel.CONSOLE.value],
            NotificationType.SYSTEM.value: [NotificationChannel.EMAIL.value]
        })
        
        # Rate limiting
        self.last_notification_times = {}
        self.min_interval_seconds = config.get('min_interval_seconds', 60)
        
        # Inicializar canales
        self._initialize_channels()
        
        self.logger.info(f"NotificationManager inicializado - Canales activos: {self.enabled_channels}")

    def _initialize_channels(self) -> None:
        """Inicializa y valida los canales de notificación."""
        if NotificationChannel.TELEGRAM.value in self.enabled_channels:
            if not self._validate_telegram_config():
                self.logger.warning("Configuración de Telegram inválida")
                
        if NotificationChannel.DISCORD.value in self.enabled_channels:
            if not self._validate_discord_config():
                self.logger.warning("Configuración de Discord inválida")
                
        if NotificationChannel.EMAIL.value in self.enabled_channels:
            if not self._validate_email_config():
                self.logger.warning("Configuración de email inválida")

    def _validate_telegram_config(self) -> bool:
        """Valida la configuración de Telegram."""
        required_keys = ['bot_token', 'chat_id']
        return all(key in self.telegram_config for key in required_keys)

    def _validate_discord_config(self) -> bool:
        """Valida la configuración de Discord."""
        return 'webhook_url' in self.discord_config

    def _validate_email_config(self) -> bool:
        """Valida la configuración de email."""
        required_keys = ['smtp_server', 'smtp_port', 'username', 'password', 'to_email']
        return all(key in self.email_config for key in required_keys)

    def send_notification(self, message: str, 
                         notification_type: NotificationType = NotificationType.INFO,
                         data: Optional[Dict] = None,
                         force: bool = False) -> bool:
        """Envía una notificación a los canales apropiados."""
        try:
            # Rate limiting (excepto si es forzado)
            if not force and self._is_rate_limited(notification_type, message):
                return False
            
            # Determinar canales para este tipo de notificación
            target_channels = self._get_target_channels(notification_type)
            
            if not target_channels:
                return False
            
            success_count = 0
            
            # Enviar a cada canal
            for channel in target_channels:
                try:
                    if self._send_to_channel(channel, message, notification_type, data):
                        success_count += 1
                except Exception as e:
                    self.logger.error(f"Error enviando a {channel}: {e}")
            
            # Actualizar rate limiting
            self._update_rate_limit(notification_type, message)
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error en send_notification: {e}")
            return False

    def _is_rate_limited(self, notification_type: NotificationType, message: str) -> bool:
        """Verifica si la notificación está limitada por rate limiting."""
        key = f"{notification_type.value}:{hash(message)}"
        
        if key in self.last_notification_times:
            time_diff = datetime.now().timestamp() - self.last_notification_times[key]
            if time_diff < self.min_interval_seconds:
                self.logger.debug(f"Notificación rate limited: {key}")
                return True
        
        return False

    def _update_rate_limit(self, notification_type: NotificationType, message: str) -> None:
        """Actualiza el timestamp para rate limiting."""
        key = f"{notification_type.value}:{hash(message)}"
        self.last_notification_times[key] = datetime.now().timestamp()

    def _get_target_channels(self, notification_type: NotificationType) -> List[str]:
        """Obtiene los canales objetivo para un tipo de notificación."""
        filtered_channels = self.notification_filters.get(notification_type.value, [])
        return [ch for ch in filtered_channels if ch in self.enabled_channels]

    def _send_to_channel(self, channel: str, message: str, 
                        notification_type: NotificationType, data: Optional[Dict]) -> bool:
        """Envía notificación a un canal específico."""
        
        if channel == NotificationChannel.TELEGRAM.value:
            return self._send_telegram(message, notification_type, data)
        
        elif channel == NotificationChannel.DISCORD.value:
            return self._send_discord(message, notification_type, data)
        
        elif channel == NotificationChannel.EMAIL.value:
            return self._send_email(message, notification_type, data)
        
        elif channel == NotificationChannel.SYSTEM.value:
            return self._send_system_notification(message, notification_type)
        
        elif channel == NotificationChannel.CONSOLE.value:
            return self._send_console(message, notification_type)
        
        return False

    def _send_telegram(self, message: str, notification_type: NotificationType, 
                      data: Optional[Dict]) -> bool:
        """Envía notificación via Telegram."""
        try:
            if not self._validate_telegram_config():
                return False
            
            bot_token = self.telegram_config['bot_token']
            chat_id = self.telegram_config['chat_id']
            
            # Formatear mensaje
            formatted_message = self._format_telegram_message(message, notification_type, data)
            
            # API de Telegram
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            payload = {
                'chat_id': chat_id,
                'text': formatted_message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.debug("Notificación enviada via Telegram")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando Telegram: {e}")
            return False

    def _format_telegram_message(self, message: str, notification_type: NotificationType, 
                                data: Optional[Dict]) -> str:
        """Formatea mensaje para Telegram con HTML."""
        
        # Emojis por tipo
        emojis = {
            NotificationType.INFO: "ℹ️",
            NotificationType.SUCCESS: "✅", 
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
            NotificationType.TRADE: "📈",
            NotificationType.SYSTEM: "🤖"
        }
        
        emoji = emojis.get(notification_type, "📝")
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        formatted = f"{emoji} <b>{notification_type.value.upper()}</b> [{timestamp}]\n\n{message}"
        
        # Agregar datos adicionales si están disponibles
        if data:
            formatted += "\n\n<b>Detalles:</b>"
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    formatted += f"\n• {key}: <code>{value}</code>"
                else:
                    formatted += f"\n• {key}: {value}"
        
        return formatted

    def _send_discord(self, message: str, notification_type: NotificationType, 
                     data: Optional[Dict]) -> bool:
        """Envía notificación via Discord webhook."""
        try:
            if not self._validate_discord_config():
                return False
            
            webhook_url = self.discord_config['webhook_url']
            
            # Colores por tipo de notificación
            colors = {
                NotificationType.INFO: 0x3498db,      # Azul
                NotificationType.SUCCESS: 0x2ecc71,   # Verde
                NotificationType.WARNING: 0xf39c12,   # Naranja
                NotificationType.ERROR: 0xe74c3c,     # Rojo
                NotificationType.TRADE: 0x9b59b6,     # Púrpura
                NotificationType.SYSTEM: 0x95a5a6     # Gris
            }
            
            color = colors.get(notification_type, 0x95a5a6)
            
            # Crear embed
            embed = {
                "title": f"{notification_type.value.upper()}",
                "description": message,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "Trading Bot"
                }
            }
            
            # Agregar campos si hay datos adicionales
            if data:
                embed["fields"] = []
                for key, value in data.items():
                    embed["fields"].append({
                        "name": key,
                        "value": str(value),
                        "inline": True
                    })
            
            payload = {"embeds": [embed]}
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.debug("Notificación enviada via Discord")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando Discord: {e}")
            return False

    def _send_email(self, message: str, notification_type: NotificationType, 
                   data: Optional[Dict]) -> bool:
        """Envía notificación via email."""
        try:
            if not self._validate_email_config():
                return False
            
            smtp_server = self.email_config['smtp_server']
            smtp_port = int(self.email_config['smtp_port'])
            username = self.email_config['username']
            password = self.email_config['password']
            to_email = self.email_config['to_email']
            from_email = self.email_config.get('from_email', username)
            
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = f"Trading Bot - {notification_type.value.upper()}"
            
            # Formatear cuerpo del email
            body = self._format_email_body(message, notification_type, data)
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # Enviar email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            self.logger.debug("Notificación enviada via email")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando email: {e}")
            return False

    def _format_email_body(self, message: str, notification_type: NotificationType, 
                          data: Optional[Dict]) -> str:
        """Formatea el cuerpo del email."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        body = f"""
TRADING BOT NOTIFICATION
========================

Type: {notification_type.value.upper()}
Time: {timestamp}

Message:
{message}
"""
        
        if data:
            body += "\n\nAdditional Data:\n"
            for key, value in data.items():
                body += f"- {key}: {value}\n"
        
        body += "\n\n---\nThis is an automated message from your Trading Bot."
        
        return body

    def _send_system_notification(self, message: str, notification_type: NotificationType) -> bool:
        """Envía notificación del sistema (desktop)."""
        try:
            if not plyer:
                self.logger.debug("plyer no disponible para notificaciones del sistema")
                return False
            
            title = f"Trading Bot - {notification_type.value.upper()}"
            
            plyer.notification.notify(
                title=title,
                message=message[:200],  # Limitar longitud
                timeout=10
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando notificación del sistema: {e}")
            return False

    def _send_console(self, message: str, notification_type: NotificationType) -> bool:
        """Envía notificación a la consola."""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Colores ANSI para terminal
            colors = {
                NotificationType.INFO: '\033[94m',      # Azul
                NotificationType.SUCCESS: '\033[92m',   # Verde
                NotificationType.WARNING: '\033[93m',   # Amarillo
                NotificationType.ERROR: '\033[91m',     # Rojo
                NotificationType.TRADE: '\033[95m',     # Magenta
                NotificationType.SYSTEM: '\033[96m'     # Cian
            }
            
            reset_color = '\033[0m'
            color = colors.get(notification_type, '')
            
            formatted_message = f"{color}[{timestamp}] {notification_type.value.upper()}: {message}{reset_color}"
            print(formatted_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando a consola: {e}")
            return False

    # Métodos de conveniencia para diferentes tipos de notificaciones
    
    def notify_trade_opened(self, trade_info: Dict) -> bool:
        """Notifica que se abrió un trade."""
        message = (f"Trade abierto: {trade_info.get('contract_type', 'N/A')} "
                  f"en {trade_info.get('symbol', 'N/A')} por ${trade_info.get('stake', 0)}")
        
        return self.send_notification(message, NotificationType.TRADE, trade_info)

    def notify_trade_closed(self, trade_info: Dict) -> bool:
        """Notifica que se cerró un trade."""
        profit = trade_info.get('profit', 0)
        is_win = profit > 0
        
        status = "GANADO" if is_win else "PERDIDO"
        message = (f"Trade {status}: {trade_info.get('contract_type', 'N/A')} "
                  f"- Resultado: ${profit:.2f}")
        
        notification_type = NotificationType.SUCCESS if is_win else NotificationType.WARNING
        
        return self.send_notification(message, notification_type, trade_info)

    def notify_signal_detected(self, signal_info: Dict) -> bool:
        """Notifica que se detectó una señal."""
        message = (f"Señal detectada: {signal_info.get('signal_type', 'N/A')} "
                  f"en {signal_info.get('symbol', 'N/A')}")
        
        return self.send_notification(message, NotificationType.INFO, signal_info)

    def notify_error(self, error_message: str, error_data: Optional[Dict] = None) -> bool:
        """Notifica un error."""
        return self.send_notification(error_message, NotificationType.ERROR, error_data, force=True)

    def notify_system_event(self, event_message: str, event_data: Optional[Dict] = None) -> bool:
        """Notifica un evento del sistema."""
        return self.send_notification(event_message, NotificationType.SYSTEM, event_data)

    def notify_risk_warning(self, warning_message: str, risk_data: Optional[Dict] = None) -> bool:
        """Notifica una advertencia de riesgo."""
        return self.send_notification(warning_message, NotificationType.WARNING, risk_data, force=True)

    def send_daily_summary(self, summary_data: Dict) -> bool:
        """Envía un resumen diario."""
        try:
            total_trades = summary_data.get('total_trades', 0)
            winning_trades = summary_data.get('winning_trades', 0)
            total_profit = summary_data.get('total_profit', 0)
            win_rate = summary_data.get('win_rate', 0)
            
            message = f"""RESUMEN DIARIO
            
Trades totales: {total_trades}
Trades ganados: {winning_trades}
Tasa de éxito: {win_rate:.1f}%
P&L del día: ${total_profit:.2f}
            """
            
            # Determinar tipo de notificación basado en performance
            if total_profit > 0:
                notification_type = NotificationType.SUCCESS
            elif total_profit < -50:  # Pérdida significativa
                notification_type = NotificationType.WARNING
            else:
                notification_type = NotificationType.INFO
            
            return self.send_notification(message.strip(), notification_type, summary_data)
            
        except Exception as e:
            self.logger.error(f"Error enviando resumen diario: {e}")
            return False

    def test_notifications(self) -> Dict[str, bool]:
        """Prueba todos los canales de notificación configurados."""
        results = {}
        test_message = "Mensaje de prueba del Trading Bot"
        
        for channel in self.enabled_channels:
            try:
                result = self._send_to_channel(
                    channel, test_message, NotificationType.INFO, 
                    {"test": True, "timestamp": datetime.now().isoformat()}
                )
                results[channel] = result
                
            except Exception as e:
                self.logger.error(f"Error probando canal {channel}: {e}")
                results[channel] = False
        
        self.logger.info(f"Prueba de notificaciones completada: {results}")
        return results

    def get_notification_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de notificaciones."""
        return {
            'enabled_channels': self.enabled_channels,
            'rate_limit_entries': len(self.last_notification_times),
            'min_interval_seconds': self.min_interval_seconds,
            'telegram_configured': self._validate_telegram_config(),
            'discord_configured': self._validate_discord_config(),
            'email_configured': self._validate_email_config(),
        }

    def cleanup_rate_limits(self, hours_old: int = 24) -> int:
        """Limpia entradas antiguas del rate limiting."""
        cutoff_time = datetime.now().timestamp() - (hours_old * 3600)
        
        old_keys = [
            key for key, timestamp in self.last_notification_times.items()
            if timestamp < cutoff_time
        ]
        
        for key in old_keys:
            del self.last_notification_times[key]
        
        if old_keys:
            self.logger.debug(f"Limpiados {len(old_keys)} rate limits antiguos")
        
        return len(old_keys)


def create_notification_manager(config_path: Optional[str] = None) -> NotificationManager:
    """Factory function para crear un NotificationManager con configuración por defecto."""
    
    default_config = {
        'enabled_channels': [NotificationChannel.CONSOLE.value],
        'min_interval_seconds': 60,
        'filters': {
            NotificationType.ERROR.value: [NotificationChannel.CONSOLE.value],
            NotificationType.TRADE.value: [NotificationChannel.CONSOLE.value],
            NotificationType.WARNING.value: [NotificationChannel.CONSOLE.value],
            NotificationType.SUCCESS.value: [NotificationChannel.CONSOLE.value],
            NotificationType.INFO.value: [NotificationChannel.CONSOLE.value],
            NotificationType.SYSTEM.value: [NotificationChannel.CONSOLE.value]
        }
    }
    
    # Cargar configuración desde archivo si se proporciona
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Fusionar configuraciones
            default_config.update(file_config)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error cargando config de notificaciones: {e}")
    
    # Configurar desde variables de entorno
    if os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
        default_config['telegram'] = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        
        if NotificationChannel.TELEGRAM.value not in default_config['enabled_channels']:
            default_config['enabled_channels'].append(NotificationChannel.TELEGRAM.value)
    
    if os.getenv('DISCORD_WEBHOOK_URL'):
        default_config['discord'] = {
            'webhook_url': os.getenv('DISCORD_WEBHOOK_URL')
        }
        
        if NotificationChannel.DISCORD.value not in default_config['enabled_channels']:
            default_config['enabled_channels'].append(NotificationChannel.DISCORD.value)
    
    if os.getenv('EMAIL_SMTP_SERVER'):
        default_config['email'] = {
            'smtp_server': os.getenv('EMAIL_SMTP_SERVER'),
            'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'to_email': os.getenv('EMAIL_TO'),
            'from_email': os.getenv('EMAIL_FROM', os.getenv('EMAIL_USERNAME'))
        }
        
        if NotificationChannel.EMAIL.value not in default_config['enabled_channels']:
            default_config['enabled_channels'].append(NotificationChannel.EMAIL.value)
    
    return NotificationManager(default_config)


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración de ejemplo
    config = {
        'enabled_channels': ['console', 'telegram'],
        'telegram': {
            'bot_token': 'YOUR_BOT_TOKEN',
            'chat_id': 'YOUR_CHAT_ID'
        },
        'min_interval_seconds': 30
    }
    
    # Crear gestor
    notifier = NotificationManager(config)
    
    # Probar notificaciones
    notifier.test_notifications()
    
    # Ejemplos de uso
    notifier.notify_trade_opened({
        'symbol': 'frxEURUSD',
        'contract_type': 'CALL',
        'stake': 5.0,
        'entry_price': 1.1234
    })
    
    notifier.notify_error("Error de conexión con la API")
    
    notifier.send_daily_summary({
        'total_trades': 10,
        'winning_trades': 7,
        'total_profit': 25.50,
        'win_rate': 70.0
    })