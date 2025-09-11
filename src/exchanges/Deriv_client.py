import json
import time
import logging
import threading
from typing import Dict, Optional

try:
    import websocket
except ImportError:
    print("❌ Error: Falta dependencia websocket-client")
    print("📦 Instala con: pip install websocket-client")
    exit(1)

from .base_exchange import BaseExchange

class DerivWebSocketClient(BaseExchange):
    """Maneja la conexión WebSocket con la API de Deriv con mejor gestión de errores."""
    
    def __init__(self, app_id: str, token: str, symbol: str, granularity: int):
        """Inicializa el cliente WebSocket."""
        self.app_id = app_id
        self.token = token
        self.symbol = symbol
        self.granularity = granularity
        self.ws_url = f"wss://ws.deriv.com/websockets/v3?app_id={app_id}"
        self.ws = None
        self.logger = logging.getLogger("DerivWebSocketClient")
        self.is_connected = False
        self.reconnect_lock = threading.Lock()

    def connect(self) -> bool:
        """Establece la conexión WebSocket y autentica."""
        try:
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
            
            self.ws = websocket.create_connection(self.ws_url, timeout=10)
            self.ws.send(json.dumps({"authorize": self.token}))
            
            # Timeout para la respuesta de autenticación
            self.ws.settimeout(10)
            resp_str = self.ws.recv()
            resp = json.loads(resp_str)
            
            if resp.get('error'):
                self.logger.error(f"Error de autenticación: {resp['error']['message']}")
                return False
            
            if not resp.get('authorize'):
                self.logger.error("Respuesta de autorización inválida")
                return False
            
            # Restablecer timeout después de la autenticación
            self.ws.settimeout(None)
            self.is_connected = True
            self.logger.info("Conexión WebSocket establecida y autenticada correctamente.")
            return True
            
        except websocket.WebSocketTimeoutException:
            self.logger.error("Timeout al conectar o autenticar con Deriv")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"Error al decodificar respuesta de autenticación: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error al conectar con Deriv: {e}")
            return False

    def reconnect(self, max_attempts: int = 5, initial_delay: float = 2.0) -> bool:
        """Reintenta la conexión con backoff exponencial y thread safety."""
        with self.reconnect_lock:
            attempt = 1
            delay = initial_delay
            
            while attempt <= max_attempts:
                self.logger.info(f"Intento de reconexión {attempt}/{max_attempts}...")
                
                if self.connect():
                    if self.subscribe_candles():
                        return True
                
                self.logger.warning(f"Reconexión fallida. Esperando {delay:.1f} segundos...")
                time.sleep(delay)
                attempt += 1
                delay = min(delay * 1.5, 30.0)  # Max 30 segundos
            
            self.logger.error(f"No se pudo reconectar tras {max_attempts} intentos.")
            self.is_connected = False
            return False

    def subscribe_candles(self) -> bool:
        """Suscribe a datos de velas con validación."""
        try:
            ohlc_request = {
                "ticks_history": self.symbol,
                "adjust_start_time": 1,
                "count": 100,  # Aumentado para mejor análisis
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": self.granularity,
                "subscribe": 1
            }
            self.ws.send(json.dumps(ohlc_request))
            self.logger.info(f"Suscrito a velas de {self.symbol} con granularidad {self.granularity}s.")
            return True
        except Exception as e:
            self.logger.error(f"Error al suscribirse a velas: {e}")
            return False

    def send(self, data: Dict) -> bool:
        """Envía un mensaje al WebSocket con validación."""
        try:
            if not self.is_connected or not self.ws:
                self.logger.error("No hay conexión activa para enviar mensaje")
                return False
            
            self.ws.send(json.dumps(data))
            return True
        except Exception as e:
            self.logger.error(f"Error al enviar mensaje: {e}")
            self.is_connected = False
            return False

    def receive(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Recibe un mensaje del WebSocket con timeout opcional."""
        try:
            if not self.is_connected or not self.ws:
                return None
            
            if timeout:
                self.ws.settimeout(timeout)
            
            msg = self.ws.recv()
            
            if timeout:
                self.ws.settimeout(None)
            
            return json.loads(msg) if msg else None
            
        except websocket.WebSocketTimeoutException:
            if timeout:
                self.ws.settimeout(None)
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error al decodificar mensaje: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error al recibir mensaje: {e}")
            self.is_connected = False
            return None

    def close(self) -> None:
        """Cierra la conexión WebSocket."""
        try:
            if self.ws:
                self.ws.close()
                self.is_connected = False
                self.logger.info("Conexión WebSocket cerrada.")
        except Exception as e:
            self.logger.error(f"Error al cerrar conexión: {e}")