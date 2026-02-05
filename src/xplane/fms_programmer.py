"""
FMS Programmer for the Zibo 737 CDU.

Autonomously programs the FMC via ExtPlane keypress commands using
data from a SimBrief OFP. Reads the CDU screen datarefs to verify
each entry.
"""

import base64
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .extplane_client import ExtPlaneClient
from .simbrief_client import SimBriefClient, SimBriefData

logger = logging.getLogger(__name__)


class FMSState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


# CDU key name mappings for the Zibo 737 FMC1
_CDU_KEY_MAP = {
    # Letters
    **{ch: f"fmc1_{ch}" for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    # Digits
    **{ch: f"fmc1_{ch}" for ch in "0123456789"},
    # Special
    "/": "fmc1_slash",
    ".": "fmc1_period",
    "-": "fmc1_minus",
    " ": "fmc1_SP",
    "+": "fmc1_plus",
    # Function keys
    "CLR": "fmc1_clr",
    "DEL": "fmc1_del",
    "EXEC": "fmc1_exec",
    "INIT_REF": "fmc1_init_ref",
    "RTE": "fmc1_rte",
    "DEP_ARR": "fmc1_dep_app",
    "LEGS": "fmc1_legs",
    "HOLD": "fmc1_hold",
    "PROG": "fmc1_prog",
    "N1_LIMIT": "fmc1_n1_lim",
    "FIX": "fmc1_fix",
    "PREV_PAGE": "fmc1_prev_page",
    "NEXT_PAGE": "fmc1_next_page",
    "CLB": "fmc1_clb",
    "CRZ": "fmc1_crz",
    "DES": "fmc1_des",
    # Line select keys
    "LSK_1L": "fmc1_1L",
    "LSK_2L": "fmc1_2L",
    "LSK_3L": "fmc1_3L",
    "LSK_4L": "fmc1_4L",
    "LSK_5L": "fmc1_5L",
    "LSK_6L": "fmc1_6L",
    "LSK_1R": "fmc1_1R",
    "LSK_2R": "fmc1_2R",
    "LSK_3R": "fmc1_3R",
    "LSK_4R": "fmc1_4R",
    "LSK_5R": "fmc1_5R",
    "LSK_6R": "fmc1_6R",
}

# Zibo 737 FMC button command prefix
_CDU_CMD_PREFIX = "laminar/B738/button/"

# FMC screen datarefs — Zibo uses Line{nn}_L for the main text content
_FMC_LARGE_LINES = [f"laminar/B738/fmc1/Line{i:02d}_L" for i in range(14)]
_FMC_SCRATCHPAD = "laminar/B738/fmc1/Line_entry"
_FMC_EXEC_LIGHT = "laminar/B738/indicators/fmc_exec_lights"


def _decode_fmc_line(val) -> str:
    """Decode an FMC line dataref value. May be base64-encoded bytes."""
    if val is None:
        return ""
    s = str(val)
    # ExtPlane returns byte datarefs as base64
    try:
        decoded = base64.b64decode(s).decode("utf-8", errors="replace").rstrip("\x00")
        return decoded
    except Exception:
        return s


class CDUInterface:
    """Low-level CDU abstraction for the Zibo 737 FMC."""

    def __init__(self, client: ExtPlaneClient, keypress_delay: float = 0.08,
                 page_delay: float = 0.5):
        self.client = client
        self.keypress_delay = keypress_delay
        self.page_delay = page_delay

    def press_key(self, key: str):
        """Press a CDU key by name (e.g. 'A', 'EXEC', 'LSK_1L')."""
        mapped = _CDU_KEY_MAP.get(key.upper())
        if not mapped:
            logger.warning(f"Unknown CDU key: {key}")
            return
        cmd = f"{_CDU_CMD_PREFIX}{mapped}"
        logger.debug(f"CDU press: {key} -> cmd once {cmd}")
        self.client.send_command(cmd)
        time.sleep(self.keypress_delay)

    def type_string(self, text: str):
        """Type a string on the CDU scratchpad."""
        for ch in text.upper():
            if ch in _CDU_KEY_MAP:
                self.press_key(ch)
            else:
                logger.warning(f"Unmapped character: {ch!r}")

    def press_lsk(self, side: str, row: int):
        """Press a line select key. side='L' or 'R', row=1-6."""
        key = f"LSK_{row}{side.upper()}"
        self.press_key(key)

    def press_exec(self):
        """Press EXEC, checking the EXEC light first."""
        exec_light = self.client.get_dataref(_FMC_EXEC_LIGHT, timeout=1.0)
        if exec_light is not None:
            try:
                if float(exec_light) < 0.5:
                    logger.info("EXEC light not illuminated, skipping EXEC press")
                    return
            except (TypeError, ValueError):
                pass
        self.press_key("EXEC")
        time.sleep(self.page_delay)

    def clear_scratchpad(self):
        """Clear the scratchpad fully.

        On the Zibo 737, CLR deletes one character at a time (backspace),
        so long scratchpad content needs many presses.
        """
        for _ in range(30):
            sp = self.read_scratchpad()
            if not sp or not sp.strip():
                return
            self.press_key("CLR")
            time.sleep(0.1)

    def read_screen(self) -> List[str]:
        """Read all 14 FMC screen lines."""
        lines = []
        for dr in _FMC_LARGE_LINES:
            val = self.client.get_dataref(dr, timeout=0.5)
            lines.append(_decode_fmc_line(val))
        return lines

    def read_scratchpad(self) -> str:
        """Read the scratchpad/entry line."""
        val = self.client.get_dataref(_FMC_SCRATCHPAD, timeout=0.5)
        return _decode_fmc_line(val)

    def read_line(self, line_num: int) -> str:
        """Read a specific screen line (0-13)."""
        if 0 <= line_num < len(_FMC_LARGE_LINES):
            val = self.client.get_dataref(_FMC_LARGE_LINES[line_num], timeout=0.5)
            return _decode_fmc_line(val)
        return ""

    def verify_text(self, expected: str, line: int, retries: int = 2,
                    case_sensitive: bool = False) -> bool:
        """Check if a line contains expected text. Retries on failure."""
        for attempt in range(retries + 1):
            time.sleep(0.3)
            actual = self.read_line(line)
            a = actual if case_sensitive else actual.upper()
            e = expected if case_sensitive else expected.upper()
            if e in a:
                return True
            if attempt < retries:
                logger.debug(f"Verify retry {attempt + 1}: expected '{expected}' in line {line}, got '{actual}'")
                time.sleep(0.5)
        return False

    def check_scratchpad_error(self) -> Optional[str]:
        """Read scratchpad and return error message if any, else None."""
        sp = self.read_scratchpad()
        error_keywords = ["INVALID ENTRY", "NOT IN DATA BASE", "UNABLE",
                          "VERIFY POSITION", "KEY NOT ACTIVE"]
        for kw in error_keywords:
            if kw in sp.upper():
                return sp
        return None

    def enter_value_at_lsk(self, value: str, side: str, row: int,
                           clear_first: bool = True) -> bool:
        """Type a value and press an LSK. Returns False if scratchpad error."""
        if clear_first:
            self.clear_scratchpad()
        self.type_string(value)
        time.sleep(0.1)
        self.press_lsk(side, row)
        time.sleep(0.3)
        err = self.check_scratchpad_error()
        if err:
            logger.warning(f"CDU error after LSK {row}{side} with '{value}': {err}")
            self.clear_scratchpad()
            return False
        return True

    def wait_for_page(self, delay: Optional[float] = None):
        """Wait for a page to settle after navigation."""
        time.sleep(delay or self.page_delay)


class FMSProgrammer:
    """Page-by-page FMS programmer using SimBrief data."""

    def __init__(self, client: ExtPlaneClient, keypress_delay: float = 0.08,
                 page_delay: float = 0.5, verify_retries: int = 2,
                 xplane_path: str = ""):
        self.client = client
        self.cdu = CDUInterface(client, keypress_delay, page_delay)
        self.verify_retries = verify_retries
        self.xplane_path = xplane_path

        self._simbrief_client = SimBriefClient()
        self._data: Optional[SimBriefData] = None

        self._state = FMSState.IDLE
        self._current_step = ""
        self._log: List[str] = []
        self._progress = 0.0
        self._total_steps = 8  # Number of programming pages
        self._page_results: Dict[str, str] = {}  # page name -> "running"/"completed"/"error"

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def simbrief_data(self) -> Optional[SimBriefData]:
        return self._data

    def fetch_simbrief(self, pilot_id: str) -> SimBriefData:
        """Fetch SimBrief OFP and cache it. Also writes b738x.xml for Zibo UPLINK."""
        self._log_msg(f"Fetching SimBrief OFP for pilot {pilot_id}...")
        self._data = self._simbrief_client.fetch(pilot_id)
        self._log_msg(f"OFP loaded: {self._data.origin} -> {self._data.destination}")

        # Write b738x.xml for Zibo UPLINK
        try:
            self._write_uplink_file(pilot_id)
        except Exception as e:
            self._log_msg(f"Warning: Could not write b738x.xml: {e}")

        return self._data

    def _write_uplink_file(self, pilot_id: str):
        """Fetch SimBrief XML and save as b738x.xml for Zibo UPLINK."""
        if not self.xplane_path:
            self._log_msg("No X-Plane path configured, skipping b738x.xml")
            return

        fms_plans = Path(self.xplane_path) / "Output" / "FMS plans"
        fms_plans.mkdir(parents=True, exist_ok=True)

        xml_data = self._simbrief_client.fetch_xml(pilot_id)
        xml_path = fms_plans / "b738x.xml"
        xml_path.write_text(xml_data, encoding="utf-8")
        self._log_msg(f"Wrote {xml_path} ({len(xml_data)} bytes)")

    def trigger_uplink(self):
        """Load weights via file-based trigger for xlua helper script.

        Writes pax/cargo/fuel to a file that the xlua B738.copilotai script
        polls every 2 seconds. The xlua script can write to Zibo's internal
        datarefs since it runs in the same xlua context.
        """
        self._ensure_connected()
        d = self._data
        if not d:
            self._log_msg("No SimBrief data for weight loading")
            return

        KGS_LBS = 2.20462

        # SimBrief provides values in lbs; convert cargo/fuel to kg for Zibo
        pax_count = d.pax_count
        cargo_kg = d.cargo / KGS_LBS if d.cargo else 0
        fuel_kg = d.fuel_block / KGS_LBS if d.fuel_block else 0

        self._log_msg(f"Loading weights: {pax_count} pax, cargo={cargo_kg:.0f}kg, fuel={fuel_kg:.0f}kg")

        # Write target values to file for xlua script to read
        weights_path = Path(self.xplane_path) / "Output" / "FMS plans" / "copilotai_weights.txt"
        weights_path.write_text(
            f"pax={pax_count}\ncargo_kg={cargo_kg:.0f}\nfuel_kg={fuel_kg:.0f}\n",
            encoding="utf-8",
        )
        self._log_msg(f"Wrote weights file: {weights_path}")

        # Wait for xlua after_physics() to pick up the file (polls every 2s)
        self._log_msg("Waiting for xlua script to process file...")
        for i in range(15):
            time.sleep(2.0)
            if self._stop_event and self._stop_event.is_set():
                return
            try:
                content = weights_path.read_text(encoding="utf-8").strip()
                if content == "DONE":
                    self._log_msg("Weight loading complete (file marked DONE)")
                    break
            except OSError:
                pass
        else:
            self._log_msg("Warning: xlua script did not process file within timeout")

        # Wait for Zibo to process the weight change
        time.sleep(3.0)

        # Verify weights
        zfw = self.client.get_dataref("laminar/B738/tab/zfw_weight", timeout=1.0)
        tow = self.client.get_dataref("laminar/B738/tab/tow_weight", timeout=1.0)
        fuel = self.client.get_dataref("sim/flightmodel/weight/m_fuel_total", timeout=1.0)
        self._log_msg(f"After load: ZFW={zfw}, TOW={tow}, fuel={fuel}")

    def program_all(self):
        """Run the full programming sequence in a background thread."""
        if self._state == FMSState.RUNNING:
            raise RuntimeError("Programming already in progress")
        if not self._data:
            raise RuntimeError("No SimBrief data loaded. Fetch OFP first.")

        self._stop_event.clear()
        self._state = FMSState.RUNNING
        self._progress = 0.0
        self._current_step = "Starting"
        self._page_results.clear()

        self._thread = threading.Thread(target=self._run_all, daemon=True,
                                        name="fms-programmer")
        self._thread.start()

    def program_page(self, page: str):
        """Run a single page method in a background thread."""
        if self._state == FMSState.RUNNING:
            raise RuntimeError("Programming already in progress")
        if not self._data:
            raise RuntimeError("No SimBrief data loaded. Fetch OFP first.")

        method_map = {
            "init_ref": self.program_init_ref,
            "route": self.program_route,
            "dep_arr": self.program_dep_arr,
            "perf_init": self.program_perf_init,
            "n1_limit": self.program_n1_limit,
            "takeoff_ref": self.program_takeoff_ref,
            "vnav": self.program_vnav,
            "uplink": self.trigger_uplink,
        }
        method = method_map.get(page)
        if not method:
            raise ValueError(f"Unknown page: {page}. Valid: {list(method_map.keys())}")

        self._stop_event.clear()
        self._state = FMSState.RUNNING
        self._current_step = page
        self._page_results[page] = "running"

        def run():
            try:
                self._ensure_connected()
                method()
                self._state = FMSState.COMPLETED
                self._page_results[page] = "completed"
            except Exception as e:
                self._log_msg(f"Error on {page}: {e}")
                logger.exception(f"FMS page error: {page}")
                self._state = FMSState.ERROR
                self._page_results[page] = "error"

        self._thread = threading.Thread(target=run, daemon=True, name="fms-programmer")
        self._thread.start()

    def stop(self):
        """Abort the current programming sequence."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        if self._state == FMSState.RUNNING:
            self._state = FMSState.STOPPED
        self._log_msg("Programming stopped")

    def reset(self):
        """Reset programmer state so the user can start fresh."""
        if self._state == FMSState.RUNNING:
            self.stop()
        self._state = FMSState.IDLE
        self._current_step = ""
        self._log.clear()
        self._progress = 0.0
        self._page_results.clear()
        self._log_msg("Programmer reset")

    def get_status(self) -> Dict:
        return {
            "state": self._state.value,
            "current_step": self._current_step,
            "progress": round(self._progress, 1),
            "log": list(self._log[-200:]),
            "has_data": self._data is not None,
            "route": f"{self._data.origin}->{self._data.destination}" if self._data else None,
            "page_results": dict(self._page_results),
        }

    def read_cdu_screen(self) -> Dict:
        """Read current CDU screen for display."""
        if not self.client.is_connected:
            return {"lines": [], "scratchpad": "", "connected": False}
        lines = self.cdu.read_screen()
        sp = self.cdu.read_scratchpad()
        return {"lines": lines, "scratchpad": sp, "connected": True}

    # ------------------------------------------------------------------
    # Internal: full sequence
    # ------------------------------------------------------------------

    def _run_all(self):
        """Execute the full programming sequence."""
        try:
            self._ensure_connected()

            steps = [
                ("efb_load", self.trigger_uplink),
                ("init_ref", self.program_init_ref),
                ("route", self.program_route),
                ("dep_arr", self.program_dep_arr),
                ("perf_init", self.program_perf_init),
                ("n1_limit", self.program_n1_limit),
                ("takeoff_ref", self.program_takeoff_ref),
                ("vnav", self.program_vnav),
            ]

            for i, (name, method) in enumerate(steps):
                if self._stop_event.is_set():
                    self._state = FMSState.STOPPED
                    return
                self._current_step = name
                self._page_results[name] = "running"
                self._progress = (i / len(steps)) * 100
                self._log_msg(f"--- {name.upper()} ---")
                try:
                    method()
                    self._page_results[name] = "completed"
                except Exception as e:
                    self._log_msg(f"Error on {name}: {e} (continuing)")
                    logger.exception(f"FMS step error: {name}")
                    self._page_results[name] = "error"

            self._progress = 100.0
            self._current_step = "Complete"
            self._state = FMSState.COMPLETED
            self._log_msg("FMS programming complete")

        except Exception as e:
            self._log_msg(f"Fatal error: {e}")
            logger.exception("FMS fatal error")
            self._state = FMSState.ERROR

    def _ensure_connected(self):
        if not self.client.is_connected:
            if not self.client.connect():
                raise RuntimeError("Cannot connect to ExtPlane")

    def _check_stop(self):
        if self._stop_event.is_set():
            raise StopIteration("Programming stopped by user")

    def _log_msg(self, msg: str):
        with self._lock:
            self._log.append(msg)
        logger.info(f"[FMS] {msg}")

    # ------------------------------------------------------------------
    # Page: INIT REF
    # ------------------------------------------------------------------

    def program_init_ref(self):
        """Program PERF INIT page via N1 LIMIT -> LSK 6L.

        Zibo PERF INIT layout (confirmed by testing):
          LSK 1L: GW/CG%       LSK 1R: CRZ ALT
          LSK 2L: fuel density  LSK 2R: fuel reserves
          LSK 3L: ZFW           LSK 3R: ISA DEV / OAT
          LSK 4L: reserves      LSK 4R: TRANS ALT
          LSK 5L: data          LSK 5R: REQUEST
          LSK 6L: INDEX         LSK 6R: N1 LIMIT

        Note: ZFW/GW may require EFB loading on the Zibo. Weight entries
        via CDU may give INVALID ENTRY if weights aren't loaded via EFB first.
        """
        d = self._data
        self._check_stop()

        # Navigate to PERF INIT: N1 LIMIT -> LSK 6L
        self.cdu.clear_scratchpad()
        self.cdu.press_key("N1_LIMIT")
        self.cdu.wait_for_page()
        self.cdu.press_lsk("L", 6)
        self.cdu.wait_for_page()

        title = self.cdu.read_line(0)
        self._log_msg(f"PERF INIT title: {title}")

        self._check_stop()

        # CRZ ALT -> LSK 1R
        if d.cruise_altitude:
            alt_str = f"FL{d.cruise_altitude // 100}" if d.cruise_altitude >= 18000 else str(d.cruise_altitude)
            self._log_msg(f"Entering cruise alt: {alt_str}")
            self.cdu.enter_value_at_lsk(alt_str, "R", 1)
            self._check_stop()

        # TRANS ALT -> LSK 4R
        if d.trans_alt:
            self._log_msg(f"Entering transition alt: {d.trans_alt}")
            self.cdu.enter_value_at_lsk(str(d.trans_alt), "R", 4)
            self._check_stop()

        # EXEC if light is on
        self.cdu.press_exec()
        self._log_msg("INIT REF complete")

    # ------------------------------------------------------------------
    # Clear FMC route
    # ------------------------------------------------------------------

    def clear_route(self):
        """Clear the current FMC route by deleting origin/dest on RTE page."""
        self._ensure_connected()
        self._log_msg("Clearing FMC route...")

        self.cdu.clear_scratchpad()
        self.cdu.press_key("RTE")
        self.cdu.wait_for_page()

        # Press DEL key then LSK 1L (origin) to delete it
        self.cdu.press_key("DEL")
        self.cdu.press_lsk("L", 1)
        self.cdu.wait_for_page()

        # Press DEL key then LSK 1R (destination) to delete it
        self.cdu.press_key("DEL")
        self.cdu.press_lsk("R", 1)
        self.cdu.wait_for_page()

        # EXEC the deletion
        self.cdu.press_exec()
        self.cdu.wait_for_page()

        # Clear DEP/ARR by pressing DEP_ARR and checking
        self.cdu.clear_scratchpad()
        self._log_msg("FMC route cleared")

    # ------------------------------------------------------------------
    # Page: ROUTE
    # ------------------------------------------------------------------

    def program_route(self):
        """Program RTE page: origin, dest, flight number, route."""
        d = self._data
        self._check_stop()

        self.cdu.clear_scratchpad()
        self.cdu.press_key("RTE")
        self.cdu.wait_for_page()

        title = self.cdu.read_line(0)
        self._log_msg(f"RTE page title: {title}")

        # Origin -> LSK 1L
        if d.origin:
            self._log_msg(f"Entering origin: {d.origin}")
            self.cdu.enter_value_at_lsk(d.origin, "L", 1)
            self._check_stop()

        # Destination -> LSK 1R
        if d.destination:
            self._log_msg(f"Entering destination: {d.destination}")
            self.cdu.enter_value_at_lsk(d.destination, "R", 1)
            self._check_stop()

        # Flight number -> LSK 2L
        if d.flight_number:
            self._log_msg(f"Entering flight number: {d.flight_number}")
            self.cdu.enter_value_at_lsk(d.flight_number, "L", 2)
            self._check_stop()

        # EXEC to activate route
        self.cdu.press_exec()
        self._check_stop()

        # Go to RTE page 2 for airways/waypoints
        self.cdu.press_key("NEXT_PAGE")
        self.cdu.wait_for_page()

        # Enter route waypoints as airway/waypoint pairs
        self._enter_route_waypoints()

        # Final EXEC
        self.cdu.press_exec()
        self._log_msg("RTE complete")

    def _enter_route_waypoints(self):
        """Enter airway/waypoint pairs on RTE page 2+."""
        d = self._data
        if not d.navlog:
            self._log_msg("No navlog waypoints to enter")
            return

        # Build airway/waypoint pairs from navlog
        pairs = []
        for wpt in d.navlog:
            if wpt.is_sid_star:
                continue  # SID/STAR handled separately
            if wpt.type in ("apt",):
                continue  # Skip airports
            if wpt.airway and wpt.airway != "DCT":
                pairs.append((wpt.airway, wpt.ident))
            elif wpt.ident:
                pairs.append(("DIRECT", wpt.ident))

        if not pairs:
            self._log_msg("No route pairs to enter")
            return

        self._log_msg(f"Entering {len(pairs)} route segments")
        lsk_row = 1  # Start at row 1 on RTE page 2

        for airway, waypoint in pairs:
            self._check_stop()

            if lsk_row > 6:
                # Page is full, go to next page
                self.cdu.press_key("NEXT_PAGE")
                self.cdu.wait_for_page()
                lsk_row = 1

            # Airway -> LSK left side
            if airway and airway != "DIRECT":
                self.cdu.enter_value_at_lsk(airway, "L", lsk_row)
            else:
                # Direct: just enter the waypoint on the right
                pass

            # Waypoint -> LSK right side
            self.cdu.enter_value_at_lsk(waypoint, "R", lsk_row)
            self._log_msg(f"  {airway} -> {waypoint}")

            lsk_row += 1

    # ------------------------------------------------------------------
    # Page: DEP/ARR
    # ------------------------------------------------------------------

    def program_dep_arr(self):
        """Program DEP/ARR page: SID, STAR, runways."""
        d = self._data
        self._check_stop()

        self.cdu.clear_scratchpad()
        self.cdu.press_key("DEP_ARR")
        self.cdu.wait_for_page()

        title = self.cdu.read_line(0)
        self._log_msg(f"DEP/ARR title: {title}")

        # Read DEP/ARR INDEX and find origin/destination rows
        # FMC line layout: line 0 = title, lines 1+2 = LSK 1, lines 3+4 = LSK 2, etc.
        # Use first match only (lines repeat due to small/large font overlay)
        index_lines = self.cdu.read_screen()
        origin_lsk = 1  # default
        dest_lsk = 2    # default
        origin_found = False
        dest_found = False
        for i, line in enumerate(index_lines):
            if i < 1:
                continue
            line_upper = line.upper().strip() if line else ""
            lsk = (i + 1) // 2
            if lsk < 1 or lsk > 6:
                continue
            if not origin_found and d.origin and d.origin in line_upper:
                origin_lsk = lsk
                origin_found = True
            if not dest_found and d.destination and d.destination in line_upper:
                dest_lsk = lsk
                dest_found = True
        # If both map to same LSK, destination must be the next row
        if origin_lsk == dest_lsk:
            dest_lsk = origin_lsk + 1
        self._log_msg(f"DEP/ARR INDEX: origin {d.origin} -> LSK {origin_lsk}, dest {d.destination} -> LSK {dest_lsk}")

        # Select DEP for origin
        self.cdu.clear_scratchpad()
        self.cdu.press_lsk("L", origin_lsk)
        self.cdu.wait_for_page(1.0)
        self._check_stop()

        dep_title = self.cdu.read_line(0)
        self._log_msg(f"DEP page title: {dep_title}")

        # The DEPARTURES page shows SIDs on the left and runways on the right.
        # Select runway first (right side), then SID if available.
        if d.origin_runway:
            self._log_msg(f"Looking for origin runway: {d.origin_runway}")
            self._select_from_cdu_list(d.origin_runway, max_pages=5)
            self.cdu.wait_for_page(1.0)
            self._check_stop()

        if d.sid:
            self._log_msg(f"Looking for SID: {d.sid}")
            self._select_from_cdu_list(d.sid, max_pages=5)
            self.cdu.wait_for_page(1.0)
            self._check_stop()

        # EXEC for departure
        self.cdu.press_exec()
        self._check_stop()

        # Now go back to DEP/ARR INDEX and select ARR
        self.cdu.clear_scratchpad()
        self.cdu.press_key("DEP_ARR")
        self.cdu.wait_for_page()

        # Re-read DEP/ARR INDEX to find destination row
        index_lines2 = self.cdu.read_screen()
        dest_lsk2 = dest_lsk  # use previously determined row
        origin_lsk2 = origin_lsk
        o_found = False
        d_found = False
        for i, line in enumerate(index_lines2):
            if i < 1:
                continue
            line_upper = line.upper().strip() if line else ""
            lsk = (i + 1) // 2
            if lsk < 1 or lsk > 6:
                continue
            if not o_found and d.origin and d.origin in line_upper:
                origin_lsk2 = lsk
                o_found = True
            if not d_found and d.destination and d.destination in line_upper:
                dest_lsk2 = lsk
                d_found = True
        if origin_lsk2 == dest_lsk2:
            dest_lsk2 = origin_lsk2 + 1
        self._log_msg(f"Selecting ARR for {d.destination} at LSK {dest_lsk2}R")
        self.cdu.clear_scratchpad()
        self.cdu.press_lsk("R", dest_lsk2)
        self.cdu.wait_for_page(1.0)
        self._check_stop()

        arr_title = self.cdu.read_line(0)
        self._log_msg(f"ARR page title: {arr_title}")

        # Program STAR
        if d.star:
            self._log_msg(f"Looking for STAR: {d.star}")
            self._select_from_cdu_list(d.star, max_pages=5)
            self.cdu.wait_for_page(1.0)
            self._check_stop()

        # Select approach runway
        if d.dest_runway:
            self._log_msg(f"Looking for dest runway: {d.dest_runway}")
            self._select_from_cdu_list(d.dest_runway, max_pages=5)
            self.cdu.wait_for_page(1.0)
            self._check_stop()

        # EXEC for arrival
        self.cdu.press_exec()
        self._log_msg("DEP/ARR complete")

    def _select_from_cdu_list(self, target: str, max_pages: int = 5) -> bool:
        """Scan CDU lines for a matching entry and press its LSK.

        For runway numbers, also tries common CDU prefixes like RW, RWY.
        """
        target_upper = target.upper().strip()

        # Build search variants — especially for runway numbers
        variants = [target_upper]
        # If target looks like a runway number (digits, optional L/C/R suffix)
        if re.match(r'^\d{1,2}[LCR]?$', target_upper):
            # CDU may show "RW24", "RWY24", "RUNWAY 24", or just "24"
            variants.extend([
                f"RW{target_upper}",
                f"RWY{target_upper}",
            ])
            # If no L/C/R suffix, also try with those suffixes
            if not target_upper[-1] in "LCR":
                for suffix in ("L", "C", "R"):
                    variants.append(f"RW{target_upper}{suffix}")
                    variants.append(target_upper + suffix)

        prev_content = None
        for page in range(max_pages):
            self._check_stop()
            lines = self.cdu.read_screen()

            # Detect page cycling (same content as previous page)
            content_key = tuple(l.strip() for l in lines)
            if prev_content is not None and content_key == prev_content:
                self._log_msg(f"  Page repeated, stopping search")
                break
            prev_content = content_key

            # Log what we see for debugging
            non_empty = [(i, l) for i, l in enumerate(lines) if l.strip()]
            if non_empty:
                self._log_msg(f"  CDU page {page + 1}: {len(non_empty)} lines with content")
                for idx, txt in non_empty[:6]:
                    self._log_msg(f"    [{idx}] {txt.strip()}")

            # Check each content line for any variant match
            for line_idx, line_text in enumerate(lines):
                if not line_text or not line_text.strip():
                    continue
                line_upper = line_text.upper()
                for variant in variants:
                    if variant in line_upper:
                        # Determine which LSK corresponds to this line
                        # Line 0 = title, lines 1+2 -> LSK 1, 3+4 -> LSK 2, etc.
                        lsk_row = (line_idx + 1) // 2
                        if lsk_row > 6:
                            continue
                        # Determine side based on position in line
                        mid = len(line_text) // 2
                        pos = line_upper.find(variant)
                        side = "L" if pos < mid else "R"
                        self._log_msg(f"Found '{variant}' on line {line_idx}, pressing LSK {lsk_row}{side}")
                        self.cdu.clear_scratchpad()
                        self.cdu.press_lsk(side, lsk_row)
                        self.cdu.wait_for_page()
                        return True

            # Not found on this page, try next
            self.cdu.press_key("NEXT_PAGE")
            self.cdu.wait_for_page()

        self._log_msg(f"Could not find '{target}' (tried {variants}) after {max_pages} pages")
        return False

    # ------------------------------------------------------------------
    # Page: PERF INIT (via N1 LIMIT page access)
    # ------------------------------------------------------------------

    def program_perf_init(self):
        """PERF INIT already handled in program_init_ref. Verify page state."""
        self._check_stop()
        self._log_msg("PERF INIT already entered in INIT REF step, skipping")

    # ------------------------------------------------------------------
    # Page: N1 LIMIT
    # ------------------------------------------------------------------

    def program_n1_limit(self):
        """Program N1 LIMIT page: assumed temp, derate."""
        d = self._data
        self._check_stop()

        self.cdu.clear_scratchpad()
        self.cdu.press_key("N1_LIMIT")
        self.cdu.wait_for_page()

        title = self.cdu.read_line(0)
        self._log_msg(f"N1 LIMIT title: {title}")

        # Assumed temp -> LSK 1L
        if d.assumed_temp:
            self._log_msg(f"Entering assumed temp: {d.assumed_temp}")
            self.cdu.enter_value_at_lsk(str(d.assumed_temp), "L", 1)
            self._check_stop()

        self._log_msg("N1 LIMIT complete")

    # ------------------------------------------------------------------
    # Page: TAKEOFF REF
    # ------------------------------------------------------------------

    def program_takeoff_ref(self):
        """Program TAKEOFF REF: flaps, CG, verify V-speeds."""
        d = self._data
        self._check_stop()

        # Navigate to TAKEOFF REF: N1 LIMIT -> LSK 6R (TAKEOFF)
        self.cdu.clear_scratchpad()
        self.cdu.press_key("N1_LIMIT")
        self.cdu.wait_for_page()
        self.cdu.press_lsk("R", 6)
        self.cdu.wait_for_page()

        found = False
        title = self.cdu.read_line(0)
        if title and "TAKEOFF" in title.upper():
            found = True

        title = self.cdu.read_line(0)
        self._log_msg(f"TAKEOFF REF title: {title}")

        if not found:
            self._log_msg("Could not find TAKEOFF REF page, skipping")
            return

        # Flap setting -> LSK 1L (default to 5 for 737 if not provided)
        flap = d.flap_setting if d.flap_setting else "5"
        self._log_msg(f"Entering flap setting: {flap}")
        self.cdu.enter_value_at_lsk(flap, "L", 1)
        self.cdu.wait_for_page()
        self._check_stop()

        # CG/Trim -> LSK 2L
        if d.trim:
            self._log_msg(f"Entering trim: {d.trim}")
            self.cdu.enter_value_at_lsk(str(d.trim), "L", 2)
            self.cdu.wait_for_page()
            self._check_stop()

        # Wait for V-speeds to calculate then read them
        # Layout: line 1 right=V1, line 2 right=VR, line 3 right=V2
        time.sleep(1.5)
        screen = self.cdu.read_screen()

        # Log all content lines for debugging
        non_empty = [(i, l.strip()) for i, l in enumerate(screen) if l and l.strip()]
        for idx, txt in non_empty:
            self._log_msg(f"  TKREF [{idx}] {txt}")

        # Extract V-speeds from the right side of lines 1, 2, 3
        def extract_right_number(line):
            if not line:
                return ""
            # Right side of CDU line — look for numbers in the right half
            right_half = line[len(line)//2:]
            nums = re.findall(r'\d{2,3}', right_half)
            return nums[-1] if nums else ""

        v1 = extract_right_number(screen[1] if len(screen) > 1 else "")
        v_r = extract_right_number(screen[2] if len(screen) > 2 else "")
        v2 = extract_right_number(screen[3] if len(screen) > 3 else "")

        if v1 or v_r or v2:
            self._log_msg(f"V-speeds: V1={v1} VR={v_r} V2={v2}")
        else:
            self._log_msg("V-speeds not calculated (check CG/weight entry)")

        # EXEC
        self.cdu.press_exec()
        self._log_msg("TAKEOFF REF complete")

    # ------------------------------------------------------------------
    # Page: VNAV (CLB / CRZ / DES) - verification only
    # ------------------------------------------------------------------

    def program_vnav(self):
        """Verify VNAV pages: CLB, CRZ, DES."""
        self._check_stop()

        self.cdu.clear_scratchpad()
        self.cdu.press_key("CLB")
        self.cdu.wait_for_page()
        title = self.cdu.read_line(0)
        self._log_msg(f"CLB page: {title}")
        self._check_stop()

        # CRZ page
        self.cdu.press_key("CRZ")
        self.cdu.wait_for_page()
        title = self.cdu.read_line(0)
        self._log_msg(f"CRZ page: {title}")
        self._check_stop()

        # DES page
        self.cdu.press_key("DES")
        self.cdu.wait_for_page()
        title = self.cdu.read_line(0)
        self._log_msg(f"DES page: {title}")

        self._log_msg("VNAV verification complete")

    # ------------------------------------------------------------------
    # Synchronous execution (for checklist integration)
    # ------------------------------------------------------------------

    def run_sync(self, page: str) -> bool:
        """Run a page synchronously (blocking). For checklist integration."""
        if not self._data:
            self._log_msg("No SimBrief data - cannot program")
            return False

        method_map = {
            "program_all": self._run_all_sync,
            "init_ref": self.program_init_ref,
            "route": self.program_route,
            "dep_arr": self.program_dep_arr,
            "perf_init": self.program_perf_init,
            "n1_limit": self.program_n1_limit,
            "takeoff_ref": self.program_takeoff_ref,
            "vnav": self.program_vnav,
        }
        method = method_map.get(page)
        if not method:
            self._log_msg(f"Unknown FMS page: {page}")
            return False

        try:
            self._ensure_connected()
            method()
            return True
        except Exception as e:
            self._log_msg(f"FMS sync error on {page}: {e}")
            return False

    def _run_all_sync(self):
        """Synchronous version of _run_all for checklist integration."""
        steps = [
            self.program_init_ref,
            self.program_route,
            self.program_dep_arr,
            self.program_perf_init,
            self.program_n1_limit,
            self.program_takeoff_ref,
            self.program_vnav,
        ]
        for step in steps:
            step()
