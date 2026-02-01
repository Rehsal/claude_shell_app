"""
FMS Programmer for the Zibo 737 CDU.

Autonomously programs the FMC via ExtPlane keypress commands using
data from a SimBrief OFP. Reads the CDU screen datarefs to verify
each entry.
"""

import base64
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
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
    " ": "fmc1_space",
    "+": "fmc1_plus",
    # Function keys
    "CLR": "fmc1_clr",
    "DEL": "fmc1_del",
    "EXEC": "fmc1_exec",
    "INIT_REF": "fmc1_init_ref",
    "RTE": "fmc1_rte",
    "DEP_ARR": "fmc1_dep_arr",
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

# FMC screen datarefs (14 lines: 7 large + 7 small interleaved)
_FMC_LINE_DATAREFS = {
    "title": "laminar/B738/fmc1/Line00_X",
    **{f"line_{i}_label": f"laminar/B738/fmc1/Line{i:02d}_X" for i in range(14)},
}

# Large lines (the main content lines)
_FMC_LARGE_LINES = [f"laminar/B738/fmc1/Line{i:02d}_X" for i in range(14)]
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
        """Clear the scratchpad by pressing CLR."""
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
                 page_delay: float = 0.5, verify_retries: int = 2):
        self.client = client
        self.cdu = CDUInterface(client, keypress_delay, page_delay)
        self.verify_retries = verify_retries

        self._simbrief_client = SimBriefClient()
        self._data: Optional[SimBriefData] = None

        self._state = FMSState.IDLE
        self._current_step = ""
        self._log: List[str] = []
        self._progress = 0.0
        self._total_steps = 8  # Number of programming pages

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
        """Fetch SimBrief OFP and cache it."""
        self._log_msg(f"Fetching SimBrief OFP for pilot {pilot_id}...")
        self._data = self._simbrief_client.fetch(pilot_id)
        self._log_msg(f"OFP loaded: {self._data.origin} -> {self._data.destination}")
        return self._data

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
        }
        method = method_map.get(page)
        if not method:
            raise ValueError(f"Unknown page: {page}. Valid: {list(method_map.keys())}")

        self._stop_event.clear()
        self._state = FMSState.RUNNING
        self._current_step = page

        def run():
            try:
                self._ensure_connected()
                method()
                self._state = FMSState.COMPLETED
            except Exception as e:
                self._log_msg(f"Error on {page}: {e}")
                logger.exception(f"FMS page error: {page}")
                self._state = FMSState.ERROR

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

    def get_status(self) -> Dict:
        return {
            "state": self._state.value,
            "current_step": self._current_step,
            "progress": round(self._progress, 1),
            "log": list(self._log[-200:]),
            "has_data": self._data is not None,
            "route": f"{self._data.origin}->{self._data.destination}" if self._data else None,
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
                self._progress = (i / len(steps)) * 100
                self._log_msg(f"--- {name.upper()} ---")
                try:
                    method()
                except Exception as e:
                    self._log_msg(f"Error on {name}: {e} (continuing)")
                    logger.exception(f"FMS step error: {name}")

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
        """Program INIT REF page: cost index, cruise alt, ZFW, reserves."""
        d = self._data
        self._check_stop()

        # Navigate to INIT REF
        self.cdu.press_key("INIT_REF")
        self.cdu.wait_for_page()

        # Verify we're on IDENT page (first page of INIT REF)
        title_line = self.cdu.read_line(0)
        self._log_msg(f"INIT REF title: {title_line}")

        # Go to PERF INIT (page 2 from INIT REF -> POS INIT -> PERF INIT)
        # Actually for Zibo: INIT REF -> IDENT page. Next page -> POS INIT. Next -> PERF INIT.
        self.cdu.press_key("NEXT_PAGE")
        self.cdu.wait_for_page()
        self._check_stop()

        # Check if we're on POS INIT
        self.cdu.press_key("NEXT_PAGE")
        self.cdu.wait_for_page()
        self._check_stop()

        # Now on PERF INIT
        title = self.cdu.read_line(0)
        self._log_msg(f"PERF INIT title: {title}")

        # Cost Index -> LSK 1R
        if d.cost_index:
            self._log_msg(f"Entering cost index: {d.cost_index}")
            self.cdu.enter_value_at_lsk(str(d.cost_index), "R", 1)
            self._check_stop()

        # Cruise Altitude -> LSK 2R
        if d.cruise_altitude:
            alt_str = f"FL{d.cruise_altitude // 100}" if d.cruise_altitude >= 18000 else str(d.cruise_altitude)
            self._log_msg(f"Entering cruise alt: {alt_str}")
            self.cdu.enter_value_at_lsk(alt_str, "R", 2)
            self._check_stop()

        # ZFW -> LSK 3R (in thousands: e.g. 128.5 for 128500 lbs)
        if d.zfw:
            zfw_str = f"{d.zfw / 1000:.1f}"
            self._log_msg(f"Entering ZFW: {zfw_str}")
            self.cdu.enter_value_at_lsk(zfw_str, "L", 3)
            self._check_stop()

        # Reserves -> LSK 4R
        if d.fuel_reserve:
            res_str = f"{d.fuel_reserve / 1000:.1f}"
            self._log_msg(f"Entering reserves: {res_str}")
            self.cdu.enter_value_at_lsk(res_str, "L", 4)
            self._check_stop()

        # EXEC if light is on
        self.cdu.press_exec()
        self._log_msg("INIT REF complete")

    # ------------------------------------------------------------------
    # Page: ROUTE
    # ------------------------------------------------------------------

    def program_route(self):
        """Program RTE page: origin, dest, flight number, route."""
        d = self._data
        self._check_stop()

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

        self.cdu.press_key("DEP_ARR")
        self.cdu.wait_for_page()

        title = self.cdu.read_line(0)
        self._log_msg(f"DEP/ARR title: {title}")

        # Select DEP (LSK 1L typically)
        self.cdu.press_lsk("L", 1)
        self.cdu.wait_for_page()
        self._check_stop()

        # Program SID
        if d.sid:
            self._log_msg(f"Looking for SID: {d.sid}")
            self._select_from_cdu_list(d.sid, max_pages=5)
            self._check_stop()

        # Select runway
        if d.origin_runway:
            self._log_msg(f"Looking for runway: {d.origin_runway}")
            self._select_from_cdu_list(d.origin_runway, max_pages=3)
            self._check_stop()

        # EXEC for departure
        self.cdu.press_exec()
        self._check_stop()

        # Now go back and select ARR
        self.cdu.press_key("DEP_ARR")
        self.cdu.wait_for_page()

        # Select ARR (LSK 1R typically)
        self.cdu.press_lsk("R", 1)
        self.cdu.wait_for_page()
        self._check_stop()

        # Program STAR
        if d.star:
            self._log_msg(f"Looking for STAR: {d.star}")
            self._select_from_cdu_list(d.star, max_pages=5)
            self._check_stop()

        # Select approach runway
        if d.dest_runway:
            self._log_msg(f"Looking for dest runway: {d.dest_runway}")
            self._select_from_cdu_list(d.dest_runway, max_pages=3)
            self._check_stop()

        # EXEC for arrival
        self.cdu.press_exec()
        self._log_msg("DEP/ARR complete")

    def _select_from_cdu_list(self, target: str, max_pages: int = 5) -> bool:
        """Scan CDU lines for a matching entry and press its LSK."""
        target_upper = target.upper().strip()

        for page in range(max_pages):
            self._check_stop()
            lines = self.cdu.read_screen()

            # Check each content line (skip title/labels, check lines with content)
            for line_idx, line_text in enumerate(lines):
                if not line_text:
                    continue
                if target_upper in line_text.upper():
                    # Determine which LSK corresponds to this line
                    # Lines 1-2 -> LSK 1, 3-4 -> LSK 2, etc. (label + data pairs)
                    lsk_row = (line_idx // 2) + 1
                    if lsk_row > 6:
                        continue
                    # Determine side: if target is in left half, use L; right half, use R
                    # Simple heuristic: check if it appears in first or second half of line
                    mid = len(line_text) // 2
                    pos = line_text.upper().find(target_upper)
                    side = "L" if pos < mid else "R"
                    self._log_msg(f"Found '{target}' on line {line_idx}, pressing LSK {lsk_row}{side}")
                    self.cdu.press_lsk(side, lsk_row)
                    self.cdu.wait_for_page()
                    return True

            # Not found on this page, try next
            self.cdu.press_key("NEXT_PAGE")
            self.cdu.wait_for_page()

        self._log_msg(f"Could not find '{target}' in CDU list after {max_pages} pages")
        return False

    # ------------------------------------------------------------------
    # Page: PERF INIT (via N1 LIMIT page access)
    # ------------------------------------------------------------------

    def program_perf_init(self):
        """Program PERF INIT: cost index, cruise alt, fuel, trans alt."""
        d = self._data
        self._check_stop()

        # Navigate to PERF INIT via INIT REF
        self.cdu.press_key("INIT_REF")
        self.cdu.wait_for_page()
        self.cdu.press_key("NEXT_PAGE")
        self.cdu.wait_for_page()
        self.cdu.press_key("NEXT_PAGE")
        self.cdu.wait_for_page()

        title = self.cdu.read_line(0)
        self._log_msg(f"PERF INIT title: {title}")

        # Cost Index -> LSK 1R (if not already set)
        if d.cost_index:
            self.cdu.enter_value_at_lsk(str(d.cost_index), "R", 1)
            self._check_stop()

        # Cruise Alt -> LSK 2R
        if d.cruise_altitude:
            alt_str = f"FL{d.cruise_altitude // 100}" if d.cruise_altitude >= 18000 else str(d.cruise_altitude)
            self.cdu.enter_value_at_lsk(alt_str, "R", 2)
            self._check_stop()

        # Fuel (block fuel) -> LSK 3R (in thousands)
        if d.fuel_block:
            fuel_str = f"{d.fuel_block / 1000:.1f}"
            self._log_msg(f"Entering block fuel: {fuel_str}")
            self.cdu.enter_value_at_lsk(fuel_str, "R", 3)
            self._check_stop()

        # Transition altitude -> LSK 5L
        if d.trans_alt:
            self._log_msg(f"Entering transition alt: {d.trans_alt}")
            self.cdu.enter_value_at_lsk(str(d.trans_alt), "L", 5)
            self._check_stop()

        self.cdu.press_exec()
        self._log_msg("PERF INIT complete")

    # ------------------------------------------------------------------
    # Page: N1 LIMIT
    # ------------------------------------------------------------------

    def program_n1_limit(self):
        """Program N1 LIMIT page: assumed temp, derate."""
        d = self._data
        self._check_stop()

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

        # Navigate to TAKEOFF REF - it's accessible from N1 LIMIT via NEXT PAGE
        # or via INIT REF pages. Let's use INIT REF and page through.
        self.cdu.press_key("INIT_REF")
        self.cdu.wait_for_page()
        # Page through to TAKEOFF REF (usually 4th or 5th page in INIT REF)
        for _ in range(4):
            self.cdu.press_key("NEXT_PAGE")
            self.cdu.wait_for_page()
            self._check_stop()

        title = self.cdu.read_line(0)
        self._log_msg(f"TAKEOFF REF title: {title}")

        # Flap setting -> LSK 1L
        if d.flap_setting:
            self._log_msg(f"Entering flap setting: {d.flap_setting}")
            self.cdu.enter_value_at_lsk(str(d.flap_setting), "L", 1)
            self._check_stop()

        # CG/Trim -> LSK 2L
        if d.trim:
            self._log_msg(f"Entering trim: {d.trim}")
            self.cdu.enter_value_at_lsk(str(d.trim), "L", 2)
            self._check_stop()

        # Read V-speeds to verify they auto-populated
        time.sleep(1.0)
        screen = self.cdu.read_screen()
        self._log_msg(f"V-speed lines: {screen[6:10]}")

        # EXEC
        self.cdu.press_exec()
        self._log_msg("TAKEOFF REF complete")

    # ------------------------------------------------------------------
    # Page: VNAV (CLB / CRZ / DES) - verification only
    # ------------------------------------------------------------------

    def program_vnav(self):
        """Verify VNAV pages: CLB, CRZ, DES."""
        self._check_stop()

        # CLB page
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
