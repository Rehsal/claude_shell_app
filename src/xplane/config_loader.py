"""
Configuration loader for X-Plane Copilot.

Manages paths and settings from config/xplane_config.json.
"""

import json
import os
from pathlib import Path
from typing import Optional


class XPlaneConfig:
    """
    Loads and provides access to X-Plane configuration settings.

    The config file is expected at: config/xplane_config.json
    relative to the project root.
    """

    _instance: Optional['XPlaneConfig'] = None
    _config: dict = {}
    _config_path: Optional[Path] = None
    _last_modified: float = 0

    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern - only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize or reload configuration.

        Args:
            config_path: Optional explicit path to config file.
                         If not provided, searches in standard locations.
        """
        if config_path:
            self._config_path = Path(config_path)
        elif self._config_path is None:
            self._config_path = self._find_config_file()

        self._load_if_changed()

    def _find_config_file(self) -> Path:
        """Find the config file in standard locations."""
        # Try relative to this file first
        module_dir = Path(__file__).parent.parent.parent
        candidates = [
            module_dir / "config" / "xplane_config.json",
            Path("config/xplane_config.json"),
            Path("xplane_config.json"),
        ]

        for path in candidates:
            if path.exists():
                return path.resolve()

        # Default to first candidate (will create if needed)
        return candidates[0].resolve()

    def _load_if_changed(self) -> bool:
        """Reload config if file has changed. Returns True if reloaded."""
        if not self._config_path or not self._config_path.exists():
            return False

        mtime = self._config_path.stat().st_mtime
        if mtime > self._last_modified:
            self._load_config()
            self._last_modified = mtime
            return True
        return False

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if self._config_path and self._config_path.exists():
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = {}

    def reload(self) -> bool:
        """Force reload configuration. Returns True if successful."""
        self._last_modified = 0
        return self._load_if_changed()

    def save(self) -> None:
        """Save current configuration to file."""
        if self._config_path:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
            self._last_modified = self._config_path.stat().st_mtime

    @property
    def xplane_install_path(self) -> str:
        """Get X-Plane installation path."""
        return self._config.get("xplane", {}).get("install_path", "")

    @property
    def commands_xml_path(self) -> str:
        """Get path to XPRemote commands.xml."""
        return self._config.get("xplane", {}).get("commands_xml_path", "")

    @property
    def zibo_737_path(self) -> str:
        """Get path to Zibo 737 aircraft folder."""
        return self._config.get("xplane", {}).get("aircraft", {}).get("zibo_737_path", "")

    @property
    def extplane_host(self) -> str:
        """Get ExtPlane plugin host."""
        return self._config.get("xplane", {}).get("extplane", {}).get("host", "127.0.0.1")

    @property
    def extplane_port(self) -> int:
        """Get ExtPlane plugin port."""
        return self._config.get("xplane", {}).get("extplane", {}).get("port", 51000)

    @property
    def simbrief_pilot_id(self) -> str:
        """Get SimBrief pilot ID."""
        return self._config.get("xplane", {}).get("fms", {}).get("simbrief_pilot_id", "")

    @simbrief_pilot_id.setter
    def simbrief_pilot_id(self, value: str) -> None:
        self._config.setdefault("xplane", {}).setdefault("fms", {})["simbrief_pilot_id"] = value

    @property
    def fms_keypress_delay(self) -> float:
        """Get FMS keypress delay in seconds."""
        ms = self._config.get("xplane", {}).get("fms", {}).get("keypress_delay_ms", 80)
        return ms / 1000.0

    @property
    def fms_page_delay(self) -> float:
        """Get FMS page settle delay in seconds."""
        ms = self._config.get("xplane", {}).get("fms", {}).get("page_settle_delay_ms", 500)
        return ms / 1000.0

    @property
    def fms_verify_retries(self) -> int:
        """Get FMS verify retry count."""
        return self._config.get("xplane", {}).get("fms", {}).get("verify_retries", 2)

    @property
    def active_profile(self) -> str:
        """Get currently active aircraft profile name."""
        return self._config.get("active_profile", "X-Plane")

    @active_profile.setter
    def active_profile(self, value: str) -> None:
        """Set active aircraft profile."""
        self._config["active_profile"] = value

    @property
    def fallback_profile(self) -> str:
        """Get fallback profile name (usually 'X-Plane')."""
        return self._config.get("fallback_profile", "X-Plane")

    def get(self, key: str, default=None):
        """Get arbitrary config value by dot-notation key (e.g., 'xplane.install_path')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value) -> None:
        """Set arbitrary config value by dot-notation key."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def validate(self) -> dict:
        """
        Validate configuration and check that paths exist.

        Returns:
            dict with 'valid' bool and 'errors' list of strings
        """
        errors = []

        if not self.commands_xml_path:
            errors.append("commands_xml_path is not configured")
        elif not Path(self.commands_xml_path).exists():
            errors.append(f"commands.xml not found at: {self.commands_xml_path}")

        if not self.xplane_install_path:
            errors.append("xplane.install_path is not configured")
        elif not Path(self.xplane_install_path).exists():
            errors.append(f"X-Plane install path not found: {self.xplane_install_path}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def to_dict(self) -> dict:
        """Return full configuration as dictionary."""
        return self._config.copy()
