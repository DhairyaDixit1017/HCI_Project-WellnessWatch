from dataclasses import asdict
from typing import List

from .config import AppConfig, load_config


class ModeManager:
    def __init__(self, config_path: str = "data/configs/default_config.yaml"):
        self.config_path = config_path

    def cycle_mode(self, current_mode: str, available: List[str]) -> AppConfig:
        if not available:
            return load_config(self.config_path)
        if current_mode not in available:
            next_mode = available[0]
        else:
            i = available.index(current_mode)
            next_mode = available[(i + 1) % len(available)]

        import yaml
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw["active_mode"] = next_mode
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False)

        return load_config(self.config_path)