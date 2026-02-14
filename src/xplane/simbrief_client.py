"""
SimBrief OFP client for CopilotAI.

Fetches and parses a SimBrief Operational Flight Plan (OFP) into a
structured dataclass for use by the FMS programmer.
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SIMBRIEF_API_URL = "https://www.simbrief.com/api/xml.fetcher.php"


@dataclass
class NavlogWaypoint:
    ident: str
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    altitude: int = 0
    wind_dir: int = 0
    wind_spd: int = 0
    airway: str = ""
    type: str = ""  # e.g. "wpt", "apt", "vor", "ndb"
    is_sid_star: bool = False


@dataclass
class SimBriefData:
    # Route
    origin: str = ""
    destination: str = ""
    alternate: str = ""
    route: str = ""
    flight_number: str = ""
    airline: str = ""
    sid: str = ""
    sid_trans: str = ""
    star: str = ""
    star_trans: str = ""
    origin_runway: str = ""
    dest_runway: str = ""

    # Performance
    cruise_altitude: int = 0
    cost_index: int = 0
    initial_altitude: int = 0

    # Fuel (lbs)
    fuel_block: int = 0
    fuel_taxi: int = 0
    fuel_trip: int = 0
    fuel_reserve: int = 0
    fuel_alternate: int = 0
    fuel_min_takeoff: int = 0

    # Weights (lbs)
    zfw: int = 0
    payload: int = 0
    pax_count: int = 0
    cargo: int = 0
    estimated_tow: int = 0
    estimated_ldw: int = 0
    max_tow: int = 0
    max_ldw: int = 0
    max_zfw: int = 0

    # Takeoff performance
    v1: int = 0
    vr: int = 0
    v2: int = 0
    flap_setting: str = ""
    trim: str = ""
    assumed_temp: int = 0

    # Transition altitudes
    trans_alt: int = 0
    trans_level: int = 0

    # Wind
    avg_wind_dir: int = 0
    avg_wind_spd: int = 0

    # Navlog
    navlog: List[NavlogWaypoint] = field(default_factory=list)

    # Units used in the OFP
    units: str = "lbs"

    # Raw JSON for debugging
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


# Cache directory relative to project root
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "simbrief_cache"


class SimBriefClient:
    """Fetches and parses a SimBrief OFP."""

    def __init__(self):
        self._cached_data: Optional[SimBriefData] = None
        self._cached_pilot_id: str = ""

    @property
    def cached_data(self) -> Optional[SimBriefData]:
        return self._cached_data

    def fetch_xml(self, pilot_id: str) -> str:
        """Fetch the raw SimBrief OFP XML for Zibo UPLINK."""
        url = f"{SIMBRIEF_API_URL}?userid={pilot_id}"
        logger.info(f"Fetching SimBrief XML: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CopilotAI/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"SimBrief XML fetch: HTTP {e.code}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"SimBrief connection error: {e.reason}") from e

    def fetch(self, pilot_id: str) -> SimBriefData:
        """Fetch the latest OFP for the given SimBrief pilot ID."""
        url = f"{SIMBRIEF_API_URL}?userid={pilot_id}&json=1"
        logger.info(f"Fetching SimBrief OFP: {url}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CopilotAI/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Try to read the error body for a better message
            try:
                body = json.loads(e.read().decode("utf-8"))
                msg = body.get("fetch", {}).get("status", f"HTTP {e.code}")
            except Exception:
                msg = f"HTTP {e.code}"
            raise RuntimeError(f"SimBrief: {msg}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"SimBrief connection error: {e.reason}") from e

        if "fetch" in raw and "status" in raw["fetch"]:
            status = raw["fetch"]["status"]
            if status != "Success":
                raise RuntimeError(f"SimBrief fetch failed: {status}")

        data = self._parse(raw)
        self._cached_data = data
        self._cached_pilot_id = pilot_id
        logger.info(f"SimBrief OFP parsed: {data.origin}->{data.destination} FL{data.cruise_altitude // 100}")

        # Save to disk cache
        try:
            self._save_to_cache(data, raw)
        except Exception as e:
            logger.warning(f"Failed to save SimBrief cache: {e}")

        return data

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _save_to_cache(self, data: SimBriefData, raw: Dict) -> None:
        """Save parsed plan + raw JSON to disk cache.

        Uses origin+destination+flight_number as the key so re-fetching
        the same plan overwrites rather than creating duplicates.
        Different routes or flight numbers get separate entries.
        """
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        flt = data.flight_number or "NoFlt"
        key = f"{data.origin}{data.destination}_{flt}"
        filename = f"{key}.json"
        cache_entry = {
            "origin": data.origin,
            "destination": data.destination,
            "flight_number": data.flight_number,
            "timestamp": ts,
            "raw": raw,
        }
        path = _CACHE_DIR / filename
        path.write_text(json.dumps(cache_entry), encoding="utf-8")
        logger.info(f"Cached SimBrief plan: {path.name}")
        # Clean up old timestamp-based duplicates for this route
        for old in _CACHE_DIR.glob(f"{data.origin}{data.destination}_*.json"):
            if old != path and old.name != filename:
                try:
                    old.unlink()
                except Exception:
                    pass

    def list_cached(self) -> List[Dict]:
        """List cached plans, newest first."""
        if not _CACHE_DIR.exists():
            return []
        results = []
        for f in sorted(_CACHE_DIR.glob("*.json"), reverse=True):
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                results.append({
                    "filename": f.name,
                    "origin": meta.get("origin", ""),
                    "destination": meta.get("destination", ""),
                    "flight_number": meta.get("flight_number", ""),
                    "timestamp": meta.get("timestamp", 0),
                })
            except Exception:
                continue
        return results

    def delete_cached(self, filename: str) -> bool:
        """Delete a cached plan by filename. Returns True if deleted."""
        path = _CACHE_DIR / filename
        if path.exists():
            path.unlink()
            return True
        return False

    def load_cached(self, filename: str) -> SimBriefData:
        """Load a cached plan by filename and return parsed SimBriefData."""
        path = _CACHE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Cached plan not found: {filename}")
        cache_entry = json.loads(path.read_text(encoding="utf-8"))
        raw = cache_entry.get("raw", {})
        data = self._parse(raw)
        self._cached_data = data
        logger.info(f"Loaded cached plan: {data.origin}->{data.destination}")
        return data

    def _parse(self, raw: Dict) -> SimBriefData:
        d = SimBriefData(raw=raw)

        general = raw.get("general", {})
        origin_info = raw.get("origin", {})
        dest_info = raw.get("destination", {})
        alt_info = raw.get("alternate", {})
        fuel = raw.get("fuel", {})
        weights = raw.get("weights", {})
        params = raw.get("params", {})
        atc = raw.get("atc", {})

        # Route info
        d.origin = origin_info.get("icao_code", "")
        d.destination = dest_info.get("icao_code", "")
        d.alternate = alt_info.get("icao_code", "") if isinstance(alt_info, dict) else ""
        d.route = general.get("route", "")
        d.flight_number = atc.get("flightplan_id", general.get("flight_number", ""))
        d.airline = general.get("icao_airline", "")
        d.cost_index = _int(general.get("costindex", 0))
        d.initial_altitude = _int(general.get("initial_altitude", 0))
        # stepclimb_string format varies: "KRSW/0130" or "FL350/..." — use initial_altitude as primary
        d.cruise_altitude = _int(general.get("initial_altitude", 0))

        # Origin/dest details
        d.origin_runway = origin_info.get("plan_rwy", "")
        d.dest_runway = dest_info.get("plan_rwy", "")

        # SID/STAR from navlog
        navlog_raw = raw.get("navlog", {}).get("fix", [])
        if isinstance(navlog_raw, list) and navlog_raw:
            # First fix often has SID info, last fixes have STAR info
            first = navlog_raw[0] if navlog_raw else {}
            d.sid = first.get("via_airway", "") if first.get("is_sid_star", "0") == "1" else ""

        # Parse SID/STAR from route string as backup
        route_parts = d.route.split()
        if route_parts:
            # SID is typically first element, STAR is typically last before dest
            # But route format varies; we'll get better data from navlog
            pass

        # Fuel
        d.fuel_block = _int(fuel.get("plan_ramp", 0))
        d.fuel_taxi = _int(fuel.get("taxi", 0))
        d.fuel_trip = _int(fuel.get("enroute_burn", 0))
        d.fuel_reserve = _int(fuel.get("reserve", 0))
        d.fuel_alternate = _int(fuel.get("alternate_burn", 0))
        d.fuel_min_takeoff = _int(fuel.get("min_takeoff", 0))
        d.units = fuel.get("plan_ramp_unit", "lbs")

        # Weights
        d.zfw = _int(weights.get("est_zfw", 0))
        d.payload = _int(weights.get("payload", 0))
        d.pax_count = _int(weights.get("pax_count", 0))
        d.cargo = _int(weights.get("cargo", 0))
        d.estimated_tow = _int(weights.get("est_tow", 0))
        d.estimated_ldw = _int(weights.get("est_ldw", 0))
        d.max_tow = _int(weights.get("max_tow", 0))
        d.max_ldw = _int(weights.get("max_ldw", 0))
        d.max_zfw = _int(weights.get("max_zfw", 0))

        # Transition altitudes
        d.trans_alt = _int(origin_info.get("trans_alt", 0))
        d.trans_level = _int(dest_info.get("trans_level", 0))

        # Wind
        d.avg_wind_dir = _int(general.get("avg_wind_dir", 0))
        d.avg_wind_spd = _int(general.get("avg_wind_comp", 0))

        # Takeoff performance (from SimBrief's computed values)
        # These may not always be present
        takeoff = raw.get("takeoff", {})
        if not takeoff:
            # Some OFP formats put this under general or params
            takeoff = {}
        d.flap_setting = str(params.get("flap_setting", general.get("flap_setting", "")))

        # V-speeds from FMS or SimBrief
        d.v1 = _int(takeoff.get("v1", 0))
        d.vr = _int(takeoff.get("vr", 0))
        d.v2 = _int(takeoff.get("v2", 0))
        d.trim = str(general.get("stab_trim", ""))
        d.assumed_temp = _int(general.get("assumed_temp", params.get("assumed_temp", 0)))

        # Navlog waypoints
        d.navlog = []
        d.sid = ""
        d.star = ""
        d.sid_trans = ""
        d.star_trans = ""

        if isinstance(navlog_raw, list):
            for fix in navlog_raw:
                wpt = NavlogWaypoint(
                    ident=fix.get("ident", ""),
                    name=fix.get("name", ""),
                    lat=_float(fix.get("pos_lat", 0)),
                    lon=_float(fix.get("pos_long", 0)),
                    altitude=_int(fix.get("altitude_feet", 0)),
                    wind_dir=_int(fix.get("wind_dir", 0)),
                    wind_spd=_int(fix.get("wind_spd", 0)),
                    airway=fix.get("via_airway", ""),
                    type=fix.get("type", ""),
                    is_sid_star=fix.get("is_sid_star", "0") == "1",
                )
                d.navlog.append(wpt)

                # Extract SID/STAR names from navlog
                if wpt.is_sid_star and wpt.airway:
                    airway_upper = wpt.airway.upper()
                    if "SID" in airway_upper or (not d.sid and wpt == d.navlog[0]):
                        if not d.sid:
                            d.sid = wpt.airway
                    elif "STAR" in airway_upper:
                        d.star = wpt.airway

        return d


def _int(val) -> int:
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return 0


def _float(val) -> float:
    try:
        return float(str(val))
    except (ValueError, TypeError):
        return 0.0
