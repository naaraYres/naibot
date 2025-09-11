from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseExchange(ABC):
    """Clase base para conexiones de exchange."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establece la conexión con el exchange."""
        pass
    
    @abstractmethod
    def reconnect(self, max_attempts: int = 5, initial_delay: float = 2.0) -> bool:
        """Reintenta la conexión con backoff exponencial."""
        pass
    
    @abstractmethod
    def subscribe_candles(self) -> bool:
        """Suscribe a datos de velas."""
        pass
    
    @abstractmethod
    def send(self, data: Dict) -> bool:
        """Envía un mensaje al exchange."""
        pass
    
    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Recibe un mensaje del exchange."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Cierra la conexión."""
        pass