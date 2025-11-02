import yaml
from pathlib import Path
from typing import List, Dict
from utils.logger import get_logger

logger = get_logger("symbol_config")

class SymbolConfig:
    """Manage trading symbol configuration."""

    def __init__(self):
        self.config_path = Path(__file__).parent / "symbols.yaml"
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        """Load symbols from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("symbol_config_loaded", path=str(self.config_path))
        return config

    def get_enabled_symbols(self, asset_type: str = "stocks") -> List[str]:
        """Get list of enabled symbols."""
        assets = self._config.get("assets", {}).get(asset_type, [])
        enabled = [
            asset["symbol"]
            for asset in assets
            if asset.get("enabled", False)
        ]
        return enabled

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get configuration for a specific symbol."""
        for asset_type in ["stocks", "crypto"]:
            assets = self._config.get("assets", {}).get(asset_type, [])
            for asset in assets:
                if asset["symbol"] == symbol:
                    return asset
        return {}

# Global instance
symbol_config = SymbolConfig()
