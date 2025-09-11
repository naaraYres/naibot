from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

class BaseStrategy(ABC):
    """Clase base para estrategias de trading."""
    
    @abstractmethod
    def analyze(self, new_candle: Dict) -> Optional[Tuple[str, str, Dict]]:
        """Analiza el mercado y genera señal con información detallada."""
        pass
    
    @abstractmethod
    def should_trade(self, timestamp: int) -> bool:
        """Determina si se debe operar basado en timestamp y cooldowns."""
        pass
    
    @abstractmethod
    def update_active_zones(self, close_price) -> None:
        """Actualiza soportes y resistencias tras rupturas significativas."""
        pass