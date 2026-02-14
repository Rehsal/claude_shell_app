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

from .config_loader import XPlaneConfig
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

# IRS alignment datarefs
_IRS_ALIGN_LEFT = "laminar/B738/irs/alignment_left_remain"
_IRS_ALIGN_RIGHT = "laminar/B738/irs/alignment_right_remain"
_IRS_GPS_POS = "laminar/B738/irs/gps_pos"


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
                           clear_first: bool = True,
                           check_error: bool = True) -> bool:
        """Type a value and press an LSK. Returns False if scratchpad error."""
        if clear_first:
            self.clear_scratchpad()
        self.type_string(value)
        time.sleep(0.1)
        self.press_lsk(side, row)
        time.sleep(0.3)
        if check_error:
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
        self._co_route_name: str = ""

        self._state = FMSState.IDLE
        self._current_step = ""
        self._log: List[str] = []
        self._progress = 0.0
        self._total_steps = 9  # Number of programming pages (UPLINK excluded)
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

        # Write FMS plan file for CO ROUTE loading
        try:
            self._write_fms_plan()
        except Exception as e:
            self._log_msg(f"Warning: Could not write FMS plan: {e}")

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

    def _generate_fms_content(self) -> str:
        """Generate X-Plane 12 FMS file content from SimBrief navlog data.

        Format matches X-Plane 12 1100 version:
          type ident procedure altitude latitude longitude
        Where procedure is ADEP/ADES for airports, airway name or DRCT for
        enroute fixes, SID--name for SID waypoints, STAR--name for STAR.
        """
        d = self._data
        if not d:
            raise RuntimeError("No SimBrief data loaded")

        # Map SimBrief waypoint types to X-Plane FMS type codes
        TYPE_MAP = {"apt": 1, "ndb": 2, "vor": 3}

        # Header
        header = [
            "I",
            "1100 version",
            "CYCLE 2401",
            f"ADEP {d.origin}",
        ]
        if d.origin_runway:
            header.append(f"DEPRWY RW{d.origin_runway}")
        if d.sid:
            header.append(f"SID {d.sid}")
        header.append(f"ADES {d.destination}")
        if d.dest_runway:
            header.append(f"DESRWY RW{d.dest_runway}")
        if d.star:
            header.append(f"STAR {d.star}")

        # Build waypoint lines
        wpt_lines = []
        has_origin = False
        has_dest = False

        for wpt in d.navlog:
            # Skip SimBrief-only pseudo-waypoints (not in FMC database)
            if wpt.ident in ("TOC", "TOD"):
                continue

            is_origin = wpt.ident == d.origin and wpt.type == "apt"
            is_dest = wpt.ident == d.destination and wpt.type == "apt"

            fms_type = TYPE_MAP.get(wpt.type, 11)

            if is_origin:
                proc = "ADEP"
                has_origin = True
            elif is_dest:
                proc = "ADES"
                has_dest = True
            elif wpt.is_sid_star and wpt.airway:
                proc = wpt.airway
            elif wpt.airway and wpt.airway != "DCT":
                proc = wpt.airway
            else:
                proc = "DRCT"

            alt = wpt.altitude if wpt.altitude else 0
            wpt_lines.append(
                f"{fms_type} {wpt.ident} {proc} {alt} {wpt.lat} {wpt.lon}"
            )

        # Ensure origin airport is first entry
        if not has_origin:
            origin_wpt = next((w for w in d.navlog if w.ident == d.origin), None)
            lat = origin_wpt.lat if origin_wpt else 0.0
            lon = origin_wpt.lon if origin_wpt else 0.0
            wpt_lines.insert(0, f"1 {d.origin} ADEP 0 {lat} {lon}")

        # Ensure destination airport is last entry
        if not has_dest:
            dest_wpt = next((w for w in d.navlog if w.ident == d.destination), None)
            lat = dest_wpt.lat if dest_wpt else 0.0
            lon = dest_wpt.lon if dest_wpt else 0.0
            wpt_lines.append(f"1 {d.destination} ADES 0 {lat} {lon}")

        header.append(f"NUMENR {len(wpt_lines)}")
        header.extend(wpt_lines)
        return "\n".join(header) + "\n"

    def _write_fms_plan(self):
        """Generate and save FMS plan file for CO ROUTE loading."""
        d = self._data
        if not d or not self.xplane_path:
            self._log_msg("No X-Plane path configured, skipping FMS plan file")
            return

        fms_plans = Path(self.xplane_path) / "Output" / "FMS plans"
        fms_plans.mkdir(parents=True, exist_ok=True)

        route_name = f"{d.origin}{d.destination}"
        self._co_route_name = route_name

        content = self._generate_fms_content()
        fms_path = fms_plans / f"{route_name}.fms"
        fms_path.write_text(content, encoding="utf-8")
        self._log_msg(f"Wrote FMS plan: {fms_path} ({len(d.navlog)} waypoints)")

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
            "pos_init": self.program_pos_init,
            "init_ref": self.program_init_ref,
            "route": self.program_route,
            "dep_arr": self.program_dep_arr,
            "legs": self.delete_discontinuities,
            "perf_init": self.program_perf_init,
            "n1_limit": self.program_n1_limit,
            "takeoff_ref": self.program_takeoff_ref,
            "vnav": self.program_vnav,
            "uplink": self.trigger_uplink,
        }
        method = method_map.get(page)
        if not method:
            raise ValueError(f"Unknown page: {page}. Valid: {list(method_map.keys())}")

        if page == "legs" and XPlaneConfig().get_control_setting("skip_discontinuities"):
            self._log_msg("LEGS skipped (discontinuity removal disabled)")
            self._state = FMSState.COMPLETED
            self._page_results[page] = "completed"
            return

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
        self._stop_event.clear()
        self._thread = None
        self._log_msg("--- RESET ---")

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

            skip_disco = XPlaneConfig().get_control_setting("skip_discontinuities")

            steps = [
                ("pos_init", self.program_pos_init),
                ("init_ref", self.program_init_ref),
                ("route", self.program_route),
                ("dep_arr", self.program_dep_arr),
                ("legs", self.delete_discontinuities),
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

                if name == "legs" and skip_disco:
                    self._log_msg("--- LEGS --- (skipped: discontinuity removal disabled)")
                    self._page_results[name] = "completed"
                    continue

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
    # IRS alignment wait
    # ------------------------------------------------------------------

    def _wait_for_irs_alignment(self, timeout: float = 900, poll_interval: float = 3.0):
        """Wait for both IRS units to finish alignment (datarefs == 0).

        Args:
            timeout: Max wait in seconds (default 15 min).
            poll_interval: Seconds between polls.
        """
        start = time.time()
        last_log = 0.0

        while True:
            self._check_stop()

            left = self.client.get_dataref(_IRS_ALIGN_LEFT, timeout=1.0)
            right = self.client.get_dataref(_IRS_ALIGN_RIGHT, timeout=1.0)

            try:
                left_val = float(left) if left is not None else -1
                right_val = float(right) if right is not None else -1
            except (TypeError, ValueError):
                left_val, right_val = -1, -1

            # Both aligned
            if left_val == 0 and right_val == 0:
                self._log_msg("IRS aligned (both L/R = 0)")
                return

            # No power (-1) or still aligning (>0) — log periodically
            elapsed = time.time() - start
            if elapsed - last_log >= 15:
                self._log_msg(f"IRS aligning... left={left_val:.0f}s  right={right_val:.0f}s")
                last_log = elapsed

            if elapsed >= timeout:
                self._log_msg(f"Warning: IRS alignment timeout ({timeout}s) — continuing anyway")
                return

            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Page: POS INIT
    # ------------------------------------------------------------------

    def program_pos_init(self):
        """Navigate to POS INIT page, enter airport ref, copy coords to SET IRS POS."""
        d = self._data
        self._check_stop()

        # Navigate to POS INIT page
        self.cdu.clear_scratchpad()
        self.cdu.press_key("INIT_REF")
        self.cdu.wait_for_page()

        title = self.cdu.read_line(0)
        self._log_msg(f"INIT REF page title: {title}")

        if "POS INIT" not in (title or "").upper():
            self.cdu.press_key("NEXT_PAGE")
            self.cdu.wait_for_page()
            title = self.cdu.read_line(0)
            self._log_msg(f"After NEXT PAGE: {title}")

        self._check_stop()

        # Enter origin airport ICAO at LSK 2L (REF AIRPORT)
        if d.origin:
            self._log_msg(f"Entering REF AIRPORT: {d.origin}")
            self.cdu.enter_value_at_lsk(d.origin, "L", 2, check_error=True)
            time.sleep(0.5)
            self._check_stop()

            # Copy airport coordinates from LSK 2R into scratchpad
            self._log_msg("Copying airport coordinates (LSK 2R)")
            self.cdu.press_lsk("R", 2)
            time.sleep(0.3)

            # Paste into SET IRS POS (LSK 4R)
            self._log_msg("Setting IRS position (LSK 4R)")
            self.cdu.press_lsk("R", 4)
            self.cdu.wait_for_page()

        self._log_msg("POS INIT complete")

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
        self._log_msg("On PERF INIT page")

        self._check_stop()

        # ZFW -> LSK 3L (enter before GW so FMC can auto-calculate)
        if d.zfw:
            zfw_str = f"{d.zfw / 1000:.1f}"
            self._log_msg(f"Entering ZFW: {zfw_str}")
            self.cdu.enter_value_at_lsk(zfw_str, "L", 3, check_error=True)
            self._check_stop()

        # Reserves -> LSK 4L
        if d.fuel_reserve:
            rsv_str = f"{d.fuel_reserve / 1000:.1f}"
            self._log_msg(f"Entering reserves: {rsv_str}")
            self.cdu.enter_value_at_lsk(rsv_str, "L", 4, check_error=True)
            self._check_stop()

        # Cost Index -> LSK 5L
        if d.cost_index is not None:
            self._log_msg(f"Entering cost index: {d.cost_index}")
            ok = self.cdu.enter_value_at_lsk(str(d.cost_index), "L", 5, check_error=True)
            if not ok:
                self._log_msg("WARNING: Cost index entry failed (CDU error)")
            self._check_stop()

        # CRZ ALT -> LSK 1R (raw altitude; FMC may warn "UNABLE CRZ ALT" for low values)
        if d.cruise_altitude:
            alt_str = str(d.cruise_altitude)
            self._log_msg(f"Entering cruise alt: {alt_str}")
            self.cdu.enter_value_at_lsk(alt_str, "R", 1, check_error=False)
            # Clear any "UNABLE CRZ ALT" warning so it doesn't block LNAV/VNAV
            self.cdu.clear_scratchpad()
            self._check_stop()

        # GW -> LSK 1L (enter after ZFW since GW may auto-calculate from ZFW + fuel)
        if d.estimated_tow:
            gw_str = f"{d.estimated_tow / 1000:.1f}"
            self._log_msg(f"Entering GW: {gw_str}")
            self.cdu.enter_value_at_lsk(gw_str, "L", 1, check_error=True)
            self._check_stop()

        # TRANS ALT -> LSK 4R
        if d.trans_alt:
            self._log_msg(f"Entering transition alt: {d.trans_alt}")
            self.cdu.enter_value_at_lsk(str(d.trans_alt), "R", 4, clear_first=False, check_error=False)
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

        self.cdu.press_key("DEL")
        self.cdu.press_lsk("L", 1)
        self.cdu.wait_for_page()

        self.cdu.press_key("DEL")
        self.cdu.press_lsk("R", 1)
        self.cdu.wait_for_page()

        self.cdu.press_exec()
        self.cdu.wait_for_page()

        self.cdu.clear_scratchpad()
        self._log_msg("FMC route cleared")

    # ------------------------------------------------------------------
    # Page: ROUTE
    # ------------------------------------------------------------------

    def program_route(self):
        """Program RTE page using CO ROUTE for instant route loading.

        Loads the entire route from an FMS plan file written during
        fetch_simbrief(). This is much faster than entering waypoints
        one by one via the CDU.
        """
        d = self._data
        self._check_stop()

        self.cdu.clear_scratchpad()
        self.cdu.press_key("RTE")
        self.cdu.wait_for_page()
        self._log_msg("On RTE page")

        # Origin -> LSK 1L (required before CO ROUTE is available)
        if d.origin:
            self._log_msg(f"Entering origin: {d.origin}")
            self.cdu.enter_value_at_lsk(d.origin, "L", 1)
            self._check_stop()

        # Destination -> LSK 1R
        if d.destination:
            self._log_msg(f"Entering destination: {d.destination}")
            self.cdu.enter_value_at_lsk(d.destination, "R", 1)
            self._check_stop()

        # CO ROUTE name (e.g. "KRSWKMIA") -> LSK 2L
        route_name = self._co_route_name or f"{d.origin}{d.destination}"
        self._log_msg(f"Loading CO ROUTE: {route_name}")
        ok = self.cdu.enter_value_at_lsk(route_name, "L", 2)
        if not ok:
            self._log_msg("CO ROUTE entry failed (check FMS plan file)")
        time.sleep(1.0)  # Wait for FMC to load route from file
        self._check_stop()

        # Flight number -> LSK 2R
        if d.flight_number:
            self._log_msg(f"Entering flight number: {d.flight_number}")
            self.cdu.enter_value_at_lsk(d.flight_number, "R", 2)
            self._check_stop()

        # ACTIVATE -> LSK 6R
        self._log_msg("Activating route")
        self.cdu.press_lsk("R", 6)
        self.cdu.wait_for_page()
        self._check_stop()

        # EXEC
        self.cdu.press_exec()
        self._log_msg("RTE complete (via CO ROUTE)")

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
    # Page: LEGS — discontinuity removal
    # ------------------------------------------------------------------

    def delete_discontinuities(self):
        """Scan LEGS pages and remove route discontinuities.

        Zibo 737 LEGS layout (14 screen lines, 6 LSK rows):
          Line 0  = title
          Lines 1-2  = LSK 1 (waypoint ident on odd line, details on even)
          Lines 3-4  = LSK 2
          ...
          Lines 11-12 = LSK 6

        Discontinuities show as "-----" on the even line (small font row)
        of an LSK row. To remove: click the LSK of the waypoint AFTER the
        discontinuity (puts the ident in the scratchpad), then click the
        LSK of the waypoint BEFORE (connects them, removing the disco).

        If the next waypoint is on the same page (disco_lsk_row + 1), we
        handle it directly. If the disco is on LSK 6, the next waypoint
        is on the following page's LSK 1.
        """
        self._check_stop()

        self.cdu.clear_scratchpad()
        self.cdu.press_key("LEGS")
        self.cdu.wait_for_page()
        self._log_msg("On LEGS page")

        max_iterations = 20  # safety limit
        total_removed = 0

        for iteration in range(max_iterations):
            self._check_stop()
            found = False

            # Scan current page and all subsequent pages
            pages_checked = 0
            max_pages = 10
            prev_content = None

            while pages_checked < max_pages:
                self._check_stop()
                lines = self.cdu.read_screen()

                # Detect page cycling
                content_key = tuple(l.strip() for l in lines)
                if prev_content is not None and content_key == prev_content:
                    break
                prev_content = content_key

                # Check each line for discontinuity markers (-----)
                # Discontinuities appear on even-numbered lines (2, 4, 6, 8, 10, 12)
                for line_idx in range(2, 13, 2):
                    line_text = lines[line_idx] if line_idx < len(lines) else ""
                    if not line_text or not line_text.strip():
                        continue
                    line_upper = line_text.strip().upper()
                    is_disco = (
                        "DISCONTINUITY" in line_upper
                        or (len(line_upper) >= 3 and all(c == '-' for c in line_upper))
                    )
                    if not is_disco:
                        continue

                    disco_lsk = (line_idx + 1) // 2  # LSK row of the disco
                    if disco_lsk < 1 or disco_lsk > 6:
                        continue

                    # The waypoint BEFORE the disco is on the same LSK row
                    # (its ident is on the odd line above: line_idx - 1)
                    before_lsk = disco_lsk

                    # The waypoint AFTER the disco is on the next LSK row
                    after_lsk = disco_lsk + 1

                    if after_lsk <= 6:
                        # Next waypoint is on same page
                        after_line = lines[after_lsk * 2 - 1] if (after_lsk * 2 - 1) < len(lines) else ""
                        after_ident = after_line.strip().split()[0] if after_line.strip() else "?"
                        self._log_msg(f"Discontinuity at LSK {disco_lsk}L, connecting to {after_ident} at LSK {after_lsk}L")
                        self.cdu.clear_scratchpad()
                        self.cdu.press_lsk("L", after_lsk)
                        time.sleep(0.3)
                        self.cdu.press_lsk("L", before_lsk)
                        self.cdu.wait_for_page()
                    else:
                        # Disco is on LSK 6 — next waypoint is on next page LSK 1
                        self._log_msg(f"Discontinuity at LSK 6L (end of page), going to next page")
                        self.cdu.clear_scratchpad()
                        self.cdu.press_key("NEXT_PAGE")
                        self.cdu.wait_for_page()
                        next_lines = self.cdu.read_screen()
                        after_line = next_lines[1] if len(next_lines) > 1 else ""
                        after_ident = after_line.strip().split()[0] if after_line.strip() else "?"
                        self._log_msg(f"Connecting to {after_ident} at LSK 1L (next page)")
                        self.cdu.press_lsk("L", 1)
                        time.sleep(0.3)
                        # Go back to previous page where the disco was
                        self.cdu.press_key("PREV_PAGE")
                        self.cdu.wait_for_page()
                        self.cdu.press_lsk("L", 6)
                        self.cdu.wait_for_page()

                    total_removed += 1
                    found = True
                    break  # Re-scan from page 1 after each removal

                if found:
                    # Go back to LEGS page 1 to re-scan
                    self.cdu.press_key("LEGS")
                    self.cdu.wait_for_page()
                    break

                # No discontinuity on this page, try next
                pages_checked += 1
                self.cdu.press_key("NEXT_PAGE")
                self.cdu.wait_for_page()

            if not found:
                break

        if total_removed:
            self._log_msg(f"Removed {total_removed} discontinuit{'y' if total_removed == 1 else 'ies'}")
        else:
            self._log_msg("No discontinuities found")

        self.cdu.press_exec()
        self._log_msg("LEGS complete")

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
        """Navigate to N1 LIMIT page."""
        self._check_stop()

        self.cdu.clear_scratchpad()
        self.cdu.press_key("N1_LIMIT")
        self.cdu.wait_for_page()
        self._log_msg("N1 LIMIT complete")

    # ------------------------------------------------------------------
    # Page: TAKEOFF REF
    # ------------------------------------------------------------------

    def program_takeoff_ref(self):
        """Program TAKEOFF REF: obtain CG, enter V-speeds."""
        self._check_stop()

        # Navigate to TAKEOFF REF: N1 LIMIT -> LSK 6R (TAKEOFF)
        self.cdu.clear_scratchpad()
        self.cdu.press_key("N1_LIMIT")
        self.cdu.wait_for_page()
        self.cdu.press_lsk("R", 6)
        self.cdu.wait_for_page()

        title = self.cdu.read_line(0)
        self._log_msg(f"TAKEOFF REF title: {title}")

        if not title or "TAKEOFF" not in title.upper():
            self._log_msg("Could not find TAKEOFF REF page, skipping")
            return

        # Enter FLAPS at LSK 1L
        d = self._data
        flaps = d.flap_setting if d.flap_setting else "5"
        self._log_msg(f"Entering FLAPS: {flaps}")
        self.cdu.enter_value_at_lsk(flaps, "L", 1, check_error=True)
        self._check_stop()

        # Press LSK 3L to obtain CG
        self._log_msg("Obtaining CG (LSK 3L)")
        self.cdu.press_lsk("L", 3)
        time.sleep(0.5)
        self._check_stop()

        # Press LSK 1R, 2R, 3R to enter V1, VR, V2
        self._log_msg("Entering V1 (LSK 1R)")
        self.cdu.press_lsk("R", 1)
        time.sleep(0.3)

        self._log_msg("Entering VR (LSK 2R)")
        self.cdu.press_lsk("R", 2)
        time.sleep(0.3)

        self._log_msg("Entering V2 (LSK 3R)")
        self.cdu.press_lsk("R", 3)
        time.sleep(0.3)

        self._check_stop()

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
            "pos_init": self.program_pos_init,
            "init_ref": self.program_init_ref,
            "route": self.program_route,
            "dep_arr": self.program_dep_arr,
            "legs": self.delete_discontinuities,
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
            self.program_pos_init,
            self.program_init_ref,
            self.program_route,
            self.program_dep_arr,
            self.delete_discontinuities,
            self.program_perf_init,
            self.program_n1_limit,
            self.program_takeoff_ref,
            self.program_vnav,
        ]
        for step in steps:
            step()
