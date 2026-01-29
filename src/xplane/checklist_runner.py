"""
Checklist Copilot - Parses XChecklist Clist.txt and automatically satisfies
checklist items by writing datarefs via ExtPlane.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ItemType(Enum):
    SW_ITEM = "sw_item"
    SW_ITEMVOID = "sw_itemvoid"
    SW_REMARK = "sw_remark"
    SW_CONTINUE = "sw_continue"


class RunnerState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_CONFIRM = "waiting_confirm"
    COMPLETED = "completed"


@dataclass
class DatarefCondition:
    dataref: str
    operator: str  # =, >, <, >=, <=, !, |, +>, +<, etc.
    value: Any
    index: Optional[int] = None  # array index like [0]
    cross_ref: Optional[str] = None  # {other_dataref} reference


@dataclass
class ChecklistItem:
    item_type: ItemType
    label: str = ""
    checked_text: str = ""
    conditions: List[DatarefCondition] = field(default_factory=list)
    condition_logic: str = "&&"  # && or ||
    continue_target: str = ""  # for sw_continue
    raw_line: str = ""


@dataclass
class Checklist:
    name: str
    items: List[ChecklistItem] = field(default_factory=list)


def _parse_condition(cond_str: str) -> Optional[DatarefCondition]:
    """Parse a single dataref condition like 'sim/cockpit/electrical/battery_on[0]:1'."""
    cond_str = cond_str.strip()
    if not cond_str:
        return None

    # Check for change-detection prefix (+)
    is_trigger = False
    if cond_str.startswith('+'):
        is_trigger = True
        cond_str = cond_str[1:]

    # Split on the first operator character after the dataref path
    # Format: dataref_path[index]:operator_value or dataref_path:operator_value
    # Find the colon that separates dataref from condition (not inside {})
    colon_idx = -1
    brace_depth = 0
    for i, ch in enumerate(cond_str):
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
        elif ch == ':' and brace_depth == 0:
            colon_idx = i
            break

    if colon_idx == -1:
        return None

    dataref_part = cond_str[:colon_idx]
    value_part = cond_str[colon_idx + 1:]

    # Extract array index
    index = None
    idx_match = re.search(r'\[(\d+)\]', dataref_part)
    if idx_match:
        index = int(idx_match.group(1))
        dataref_part = dataref_part[:idx_match.start()]

    # Check for change-detection prefix in value_part too (e.g. dataref:+>0.5)
    if value_part.startswith('+'):
        is_trigger = True
        value_part = value_part[1:]

    # Determine operator and value
    operator = "="
    if value_part.startswith('>='):
        operator = ">=" if not is_trigger else "+>="
        value_part = value_part[2:]
    elif value_part.startswith('<='):
        operator = "<=" if not is_trigger else "+<="
        value_part = value_part[2:]
    elif value_part.startswith('>'):
        operator = ">" if not is_trigger else "+>"
        value_part = value_part[1:]
    elif value_part.startswith('<'):
        operator = "<" if not is_trigger else "+<"
        value_part = value_part[1:]
    elif value_part.startswith('!'):
        operator = "!"
        value_part = value_part[1:]
    elif value_part.startswith('|'):
        operator = "|"
        value_part = value_part[1:]
    elif '|' in value_part and not value_part.startswith('{'):
        # Range like 0.5|1.5 (pipe in the middle)
        operator = "|"
    else:
        if is_trigger:
            operator = "+="

    # Check for cross-dataref reference: {dataref} or {dataref}*multiplier
    cross_ref = None
    cross_match = re.match(r'^\{([^}]+)\}(.*)$', value_part)
    if cross_match:
        cross_ref = cross_match.group(1)
        # Store any expression suffix (e.g. "*3.28084") in value
        expr_suffix = cross_match.group(2).strip()
        value = expr_suffix if expr_suffix else None
    else:
        # Parse range for | operator
        if operator == "|" and '|' in value_part:
            # Range like 0.0|1.0 means between 0.0 and 1.0
            value = value_part  # keep as string, handle in runner
        else:
            try:
                value = float(value_part)
                if value == int(value):
                    value = int(value)
            except ValueError:
                value = value_part

    return DatarefCondition(
        dataref=dataref_part,
        operator=operator,
        value=value,
        index=index,
        cross_ref=cross_ref,
    )


def _parse_conditions(conditions_str: str) -> Tuple[List[DatarefCondition], str]:
    """Parse a conditions string, returning conditions and logic operator."""
    conditions_str = conditions_str.strip()
    if not conditions_str:
        return [], "&&"

    # Strip outer parentheses from individual conditions
    # e.g. (dataref:1)&&(dataref:2) -> dataref:1 && dataref:2

    # Determine logic: || or && (but not inside parentheses)
    if ')&&(' in conditions_str or '&&' in conditions_str:
        logic = "&&"
        parts = re.split(r'\)\s*&&\s*\(|\s*&&\s*', conditions_str)
    elif ')||(' in conditions_str or '||' in conditions_str:
        logic = "||"
        parts = re.split(r'\)\s*\|\|\s*\(|\s*\|\|\s*', conditions_str)
    else:
        logic = "&&"
        parts = [conditions_str]

    conditions = []
    for part in parts:
        # Strip surrounding parentheses
        part = part.strip().strip('()')
        cond = _parse_condition(part.strip())
        if cond:
            conditions.append(cond)

    return conditions, logic


def _parse_item_content(content: str) -> Tuple[str, str, str]:
    """
    Parse XChecklist item content: LABEL|CHECKED_TEXT:conditions

    The | separates label from checked_text. The first : after the
    checked_text (that looks like a dataref path) starts the conditions.
    Conditions contain their own : separators (dataref:value).

    Returns (label, checked_text, conditions_string).
    """
    content = content.strip()

    # Split on | to get label and rest
    pipe_idx = content.find('|')
    if pipe_idx == -1:
        # No pipe - entire content is just a label, no checked_text or conditions
        return content, "", ""

    label = content[:pipe_idx].strip()
    rest = content[pipe_idx + 1:]

    # rest = "CHECKED_TEXT:conditions" or just "CHECKED_TEXT"
    # The checked_text is human-readable (e.g. "ON", "ARMED", "SET")
    # Conditions start with a dataref path or ( character
    # Find the first : that starts a condition (preceded by what looks
    # like end of display text, followed by a dataref-like path or paren)
    # Strategy: look for :( or :[a-z] patterns which indicate conditions
    cond_start = _find_condition_start(rest)

    if cond_start == -1:
        return label, rest.strip(), ""
    else:
        checked_text = rest[:cond_start].strip()
        cond_str = rest[cond_start + 1:].strip()
        return label, checked_text, cond_str


def _find_condition_start(text: str) -> int:
    """
    Find the index of the : that separates checked_text from conditions.

    Returns -1 if no conditions found.
    """
    # Look for : followed by ( or a lowercase letter (start of dataref path)
    # or + (trigger prefix)
    # Skip : that are inside parentheses (those are dataref:value separators)
    for i, ch in enumerate(text):
        if ch == ':':
            remaining = text[i + 1:]
            # Check if what follows looks like a condition start
            if remaining and (remaining[0] == '(' or remaining[0] == '+'
                             or re.match(r'[a-z0-9]', remaining[0])):
                return i
    return -1


def parse_clist(text: str) -> List[Checklist]:
    """Parse Clist.txt content into structured checklists."""
    checklists: List[Checklist] = []
    current: Optional[Checklist] = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('#'):
            continue

        # Checklist header: sw_checklist:NAME or sw_checklist:NAME:DISPLAY_NAME
        if line.startswith('sw_checklist:'):
            raw_name = line[len('sw_checklist:'):].strip()
            # Format can be NAME:DISPLAY_NAME - use first part as key
            name = raw_name.split(':')[0] if ':' in raw_name else raw_name
            current = Checklist(name=name)
            checklists.append(current)
            continue

        if current is None:
            continue

        # XChecklist format: sw_item:LABEL|CHECKED_TEXT:conditions
        # The | separates label from checked_text, then the first :
        # after checked_text begins the dataref conditions.
        # Conditions themselves contain : (dataref:value), so we
        # find the split point by locating the | first.
        if line.startswith('sw_item:'):
            label, checked_text, cond_str = _parse_item_content(line[len('sw_item:'):])
            conditions, logic = _parse_conditions(cond_str)
            current.items.append(ChecklistItem(
                item_type=ItemType.SW_ITEM,
                label=label,
                checked_text=checked_text,
                conditions=conditions,
                condition_logic=logic,
                raw_line=line,
            ))

        elif line.startswith('sw_itemvoid:'):
            label, checked_text, cond_str = _parse_item_content(line[len('sw_itemvoid:'):])
            current.items.append(ChecklistItem(
                item_type=ItemType.SW_ITEMVOID,
                label=label,
                checked_text=checked_text,
                raw_line=line,
            ))

        elif line.startswith('sw_remark:'):
            content = line[len('sw_remark:'):]
            current.items.append(ChecklistItem(
                item_type=ItemType.SW_REMARK,
                label=content.strip(),
                raw_line=line,
            ))

        elif line.startswith('sw_continue:'):
            content = line[len('sw_continue:'):].strip()
            # Format: TARGET or TARGET:condition
            # Find where condition starts (lowercase dataref path after :)
            cond_start = _find_condition_start(content)
            if cond_start != -1:
                target = content[:cond_start].strip()
            else:
                target = content
            current.items.append(ChecklistItem(
                item_type=ItemType.SW_CONTINUE,
                continue_target=target,
                raw_line=line,
            ))

    return checklists


def _compute_write_value(cond: DatarefCondition) -> Optional[float]:
    """Determine the value to write to satisfy a condition."""
    # Skip trigger/change-detection operators
    if cond.operator.startswith('+'):
        return None

    val = cond.value
    op = cond.operator

    if op == "=":
        return float(val) if val is not None else None
    elif op == ">":
        v = float(val)
        return v + 1.0 if v != 0.0 else 1.0
    elif op == ">=":
        return float(val)
    elif op == "<":
        v = float(val)
        return v - 1.0
    elif op == "<=":
        return float(val)
    elif op == "!":
        # Not equal to val; if val is 0, write 1; otherwise write 0
        v = float(val)
        return 0.0 if v != 0.0 else 1.0
    elif op == "|":
        # Range: value is "low|high" string - write the midpoint
        if isinstance(val, str) and '|' in val:
            parts = val.split('|')
            try:
                low = float(parts[0])
                high = float(parts[1])
                return (low + high) / 2.0
            except (ValueError, IndexError):
                return None
        return float(val) if val is not None else None
    elif op == "-":
        return float(val) if val is not None else None

    return float(val) if val is not None else None


def _check_condition(cond: DatarefCondition, current_value: Any) -> bool:
    """Check if a single condition is satisfied by the current value."""
    if cond.operator.startswith('+'):
        # Trigger operators - can't check statically, assume satisfied
        return True

    if current_value is None:
        return False

    # Handle array indexing
    val = current_value
    if cond.index is not None and isinstance(val, list):
        if cond.index < len(val):
            val = val[cond.index]
        else:
            return False

    try:
        val = float(val)
    except (TypeError, ValueError):
        return False

    target = cond.value
    op = cond.operator

    if op == "=":
        return abs(val - float(target)) < 0.05
    elif op == ">":
        return val > float(target)
    elif op == ">=":
        return val >= float(target)
    elif op == "<":
        return val < float(target)
    elif op == "<=":
        return val <= float(target)
    elif op == "!":
        return abs(val - float(target)) > 0.05
    elif op == "|":
        if isinstance(target, str) and '|' in target:
            parts = target.split('|')
            try:
                low = float(parts[0])
                high = float(parts[1])
                return low <= val <= high
            except (ValueError, IndexError):
                return False
        return abs(val - float(target)) < 0.05

    return False


class ChecklistRunner:
    """
    Runs parsed checklists using the commands system and ExtPlane.

    Uses Clist.txt as a template for sequencing. For each sw_item:
    1. Execute the matching command (e.g. "BATTERY ON") via ScriptExecutor
    2. Poll the item's dataref conditions to detect completion
    3. Advance automatically when conditions are met

    Runs in a background thread so the API stays responsive.
    Uses threading.Event for pause/resume/confirm/skip signaling.
    """

    def __init__(self, client, commands_loader=None, poll_timeout: float = 90.0):
        self.client = client
        self.commands_loader = commands_loader
        self.checklists: List[Checklist] = []
        self._checklist_map: Dict[str, Checklist] = {}
        self.current_checklist: Optional[Checklist] = None
        self.current_index: int = 0
        self.state: RunnerState = RunnerState.IDLE
        self._log: List[str] = []
        self._poll_timeout: float = poll_timeout

        # History tracking — every processed item gets appended here
        self._history: List[Dict] = []
        self._current_section: str = ""
        self._progress_total: int = 0  # items processed across all sub-checklists

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = running, clear = paused
        self._confirm_event = threading.Event()
        self._skip_event = threading.Event()
        self._pause_event.set()  # start unpaused
        self._lock = threading.Lock()

    def load(self, clist_path: str) -> List[str]:
        self.stop()
        path = Path(clist_path)
        if not path.exists():
            raise FileNotFoundError(f"Clist.txt not found: {clist_path}")
        text = path.read_text(encoding='utf-8', errors='replace')
        self.checklists = parse_clist(text)
        self._checklist_map = {cl.name: cl for cl in self.checklists}
        self.state = RunnerState.IDLE
        self.current_checklist = None
        self.current_index = 0
        self._log = []
        self._history = []
        self._current_section = ""
        self._progress_total = 0
        return [cl.name for cl in self.checklists]

    def list_checklists(self) -> List[Dict]:
        return [{"name": cl.name, "item_count": len(cl.items)}
                for cl in self.checklists]

    def start(self, checklist_name: str, auto_run: bool = False) -> bool:
        """Start a checklist. If auto_run=True, begins executing in background."""
        cl = self._checklist_map.get(checklist_name)
        if not cl:
            return False
        self.stop()
        self.current_checklist = cl
        self.current_index = 0
        self.state = RunnerState.RUNNING
        self._current_section = checklist_name
        self._history = []
        self._progress_total = 0
        self._log.append(f"Started checklist: {checklist_name}")
        if auto_run:
            self._start_background()
        return True

    def stop(self):
        """Stop background execution."""
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused
        self._confirm_event.set()  # unblock if waiting
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._stop_event.clear()
        self._confirm_event.clear()
        self._skip_event.clear()
        self._pause_event.set()

    def restart(self):
        """Reset to START checklist and re-run from the beginning."""
        self.stop()
        self._history = []
        self._progress_total = 0
        self._current_section = ""
        self._log = []
        # Find the START checklist (first one whose name contains 'START', or just the first)
        start_cl = None
        for cl in self.checklists:
            if 'START' in cl.name.upper():
                start_cl = cl
                break
        if not start_cl and self.checklists:
            start_cl = self.checklists[0]
        if start_cl:
            self.current_checklist = start_cl
            self.current_index = 0
            self._current_section = start_cl.name
            self.state = RunnerState.RUNNING
            self._log.append(f"Restarted from: {start_cl.name}")
            self._start_background()

    def pause(self):
        if self.state == RunnerState.RUNNING:
            self._pause_event.clear()
            self.state = RunnerState.PAUSED
            self._log.append("Paused")

    def resume(self):
        if self.state in (RunnerState.PAUSED, RunnerState.WAITING_CONFIRM):
            self.state = RunnerState.RUNNING
            self._pause_event.set()
            self._confirm_event.set()
            self._log.append("Resumed")

    def confirm(self):
        """Confirm a manual item."""
        if self.state == RunnerState.WAITING_CONFIRM:
            self._log.append(f"Confirmed: {self._current_item_label()}")
            self._confirm_event.set()

    def skip_item(self):
        """Skip the current item."""
        self._log.append(f"Skipped: {self._current_item_label()}")
        self._skip_event.set()
        # Also unblock confirm wait
        self._confirm_event.set()

    def run_single(self) -> Dict:
        """Run just the current item (blocking). For step-by-step mode."""
        return self._process_current_item()

    def run_all(self):
        """Start auto-running all items in a background thread."""
        if self.state != RunnerState.RUNNING:
            if self.current_checklist:
                self.state = RunnerState.RUNNING
        self._start_background()

    def _start_background(self):
        """Launch the background runner thread."""
        self._stop_event.clear()
        self._pause_event.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="checklist-runner")
        self._thread.start()
        self._log.append("Auto-run started")

    def _run_loop(self):
        """Background thread: process items until done, paused, or stopped."""
        while not self._stop_event.is_set():
            # Wait if paused
            self._pause_event.wait(timeout=1.0)
            if self._stop_event.is_set():
                break

            if self.state not in (RunnerState.RUNNING,):
                if self.state == RunnerState.COMPLETED:
                    break
                time.sleep(0.5)
                continue

            try:
                result = self._process_current_item()
            except Exception as e:
                logger.error(f"Runner error: {e}")
                self._log.append(f"Error: {e}")
                self._advance()
                continue

            action = result.get("action", "")
            if action == "completed":
                break
            if action == "none":
                time.sleep(0.5)

        self._log.append("Auto-run finished")

    def _current_item_label(self) -> str:
        if self.current_checklist and self.current_index < len(self.current_checklist.items):
            item = self.current_checklist.items[self.current_index]
            return item.label or item.continue_target or "(unknown)"
        return ""

    def _append_history(self, item: ChecklistItem, status: str):
        """Append a processed item to the history list."""
        self._history.append({
            "label": item.label,
            "checked_text": item.checked_text,
            "type": item.item_type.value,
            "status": status,
            "section": self._current_section,
            "continue_target": item.continue_target if item.item_type == ItemType.SW_CONTINUE else "",
        })
        self._progress_total += 1

    def get_status(self) -> Dict:
        result: Dict[str, Any] = {
            "state": self.state.value,
            "current_checklist_name": self.current_checklist.name if self.current_checklist else None,
            # Keep legacy key for compat
            "checklist": self.current_checklist.name if self.current_checklist else None,
            "item_index": self.current_index,
            "total_items": len(self.current_checklist.items) if self.current_checklist else 0,
            "current_item": None,
            "progress": 0.0,
            "running_in_background": self._thread is not None and self._thread.is_alive(),
            "history": list(self._history),
            "progress_total": self._progress_total,
            "log": self._log[-100:],
        }
        if self.current_checklist and self.current_index < len(self.current_checklist.items):
            item = self.current_checklist.items[self.current_index]
            result["current_item"] = {
                "type": item.item_type.value,
                "label": item.label,
                "checked_text": item.checked_text,
                "has_datarefs": len(item.conditions) > 0,
                "continue_target": item.continue_target,
            }
            total = len(self.current_checklist.items)
            if total > 0:
                result["progress"] = round(self.current_index / total * 100, 1)
        return result

    # ----------------------------------------------------------------
    # Item processing
    # ----------------------------------------------------------------

    def _advance(self):
        self.current_index += 1
        if self.current_checklist and self.current_index >= len(self.current_checklist.items):
            self.state = RunnerState.COMPLETED

    def _process_current_item(self) -> Dict:
        """Process the current item. May block while polling conditions."""
        if self.state not in (RunnerState.RUNNING,):
            return {"action": "none", "reason": f"Runner is {self.state.value}"}

        if not self.current_checklist:
            return {"action": "none", "reason": "No checklist loaded"}

        if self.current_index >= len(self.current_checklist.items):
            self.state = RunnerState.COMPLETED
            return {"action": "completed"}

        item = self.current_checklist.items[self.current_index]

        # sw_itemvoid / sw_remark - display only, advance
        if item.item_type in (ItemType.SW_ITEMVOID, ItemType.SW_REMARK):
            self._log.append(f"[{item.item_type.value}] {item.label}")
            self._append_history(item, "void" if item.item_type == ItemType.SW_ITEMVOID else "remark")
            self._advance()
            return {"action": "skipped", "type": item.item_type.value,
                    "label": item.label}

        # sw_continue - chain to next checklist
        if item.item_type == ItemType.SW_CONTINUE:
            target = item.continue_target
            self._log.append(f">> {target}")
            self._append_history(item, "continue")
            cl = self._checklist_map.get(target)
            if cl:
                self.current_checklist = cl
                self.current_index = 0
                self._current_section = cl.name
                return {"action": "continue", "target": target}
            else:
                self._advance()
                return {"action": "error", "reason": f"Checklist '{target}' not found"}

        # sw_item - the main work
        if item.item_type == ItemType.SW_ITEM:
            if not item.conditions:
                return self._handle_manual_item(item)
            return self._handle_dataref_item(item)

        return {"action": "none", "reason": "Unknown item type"}

    def _handle_manual_item(self, item: ChecklistItem) -> Dict:
        """Manual item with no datarefs - wait for pilot confirmation."""
        self.state = RunnerState.WAITING_CONFIRM
        self._log.append(f"Manual: {item.label}|{item.checked_text}")
        self._confirm_event.clear()
        self._skip_event.clear()

        # Block until confirmed, skipped, or stopped
        while not self._stop_event.is_set():
            if self._confirm_event.wait(timeout=1.0):
                break
            if self._skip_event.is_set():
                break

        self._confirm_event.clear()
        skipped = self._skip_event.is_set()
        self._skip_event.clear()

        if self._stop_event.is_set():
            return {"action": "none", "reason": "Stopped"}

        self._append_history(item, "skipped" if skipped else "confirmed")
        self._advance()
        self.state = RunnerState.RUNNING
        return {"action": "confirmed" if not skipped else "skipped",
                "label": item.label, "checked_text": item.checked_text}

    def _handle_dataref_item(self, item: ChecklistItem) -> Dict:
        """Execute command for item, then poll conditions."""
        try:
            return self._handle_dataref_item_inner(item)
        except Exception as e:
            logger.error(f"Error on {item.label}: {e}")
            self._log.append(f"Error: {item.label} - {e}")
            self._append_history(item, "error")
            self._advance()
            return {"action": "error", "label": item.label, "reason": str(e)}

    def _handle_dataref_item_inner(self, item: ChecklistItem) -> Dict:
        # Check ExtPlane connection
        if not self.client.is_connected:
            self._log.append(f"No connection - skipping: {item.label}")
            self._append_history(item, "skipped")
            self._advance()
            return {"action": "skipped", "label": item.label,
                    "reason": "ExtPlane not connected"}

        # Skip trigger-only items
        non_trigger = [c for c in item.conditions if not c.operator.startswith('+')]
        if not non_trigger:
            self._log.append(f"Trigger-only: {item.label}|{item.checked_text}")
            self._append_history(item, "skipped")
            self._advance()
            return {"action": "skipped", "type": "trigger_only",
                    "label": item.label, "checked_text": item.checked_text}

        # Already satisfied?
        if self._check_all_conditions(item.conditions, item.condition_logic):
            self._log.append(f"Already OK: {item.label}|{item.checked_text}")
            self._append_history(item, "satisfied")
            self._advance()
            return {"action": "already_met", "label": item.label,
                    "checked_text": item.checked_text}

        # Execute matching command first (handles covers, spring switches, etc.)
        cmd_result = self._try_execute_command(item)
        actions = []
        errors = []
        if cmd_result:
            actions = cmd_result.get("commands_sent", []) + cmd_result.get("datarefs_set", [])
            errors = cmd_result.get("errors", [])

        # Brief pause to let command take effect
        time.sleep(0.3)

        # Also write datarefs directly for any conditions not yet satisfied
        # (commands may only handle part of the item, e.g. cover but not the switch)
        to_satisfy = item.conditions if item.condition_logic == "&&" else item.conditions[:1]
        if not self._check_all_conditions(item.conditions, item.condition_logic):
            extra_actions, extra_errors = self._write_conditions(to_satisfy)
            actions.extend(extra_actions)
            errors.extend(extra_errors)
            if extra_errors:
                self._log.append(f"Write errors: {extra_errors}")

        # Log condition state before polling
        self._log.append(f"Executing: {item.label}|{item.checked_text}")
        for cond in item.conditions:
            if not cond.operator.startswith('+'):
                raw = self._read_dataref(cond.dataref, cond.index)
                met = _check_condition(cond, raw)
                idx_str = f"[{cond.index}]" if cond.index is not None else ""
                self._log.append(f"  {cond.dataref}{idx_str} {cond.operator} {cond.value}: raw={raw} met={met}")

        # Early check: if checklist conditions already met, or command datarefs verify OK, skip polling
        if self._check_all_conditions(item.conditions, item.condition_logic):
            self._log.append(f"Done: {item.label}|{item.checked_text}")
            self._append_history(item, "satisfied")
            self._advance()
            return {"action": "satisfied", "label": item.label,
                    "checked_text": item.checked_text, "actions": actions}
        if cmd_result and self._verify_command_datarefs(cmd_result):
            self._log.append(f"Done (cmd datarefs OK): {item.label}|{item.checked_text}")
            self._append_history(item, "satisfied")
            self._advance()
            return {"action": "satisfied", "label": item.label,
                    "checked_text": item.checked_text, "actions": actions}

        # If direct writes didn't work for all conditions, try toggling
        # cover/switch commands for unsatisfied cover_position datarefs
        if not self._check_all_conditions(item.conditions, item.condition_logic):
            self._try_toggle_unsatisfied(item.conditions)

        # Poll for conditions
        elapsed = self._poll_until_satisfied(item)

        verified = self._check_all_conditions(item.conditions, item.condition_logic)

        # Fallback: if checklist conditions use wrong datarefs (common in Zibo),
        # check whether the command's own script datarefs were set successfully
        if not verified and cmd_result:
            verified = self._verify_command_datarefs(cmd_result)
            if verified:
                self._log.append(f"Verified via command datarefs (checklist condition dataref mismatch)")

        if self._skip_event.is_set():
            self._skip_event.clear()
            self._log.append(f"Skipped: {item.label}")
            self._append_history(item, "skipped")
            self._advance()
            self.state = RunnerState.RUNNING
            return {"action": "skipped", "label": item.label}

        if verified:
            wait_str = f" ({elapsed:.0f}s)" if elapsed > 1 else ""
            self._log.append(f"Done{wait_str}: {item.label}|{item.checked_text}")
            self._append_history(item, "satisfied")
            self._advance()
            return {"action": "satisfied", "label": item.label,
                    "checked_text": item.checked_text, "actions": actions,
                    "wait_time": round(elapsed, 1)}
        else:
            # Timeout - wait for pilot (manual_item will append history)
            self._log.append(f"Needs attention: {item.label}|{item.checked_text}")
            return self._handle_manual_item(item)

    def _verify_command_datarefs(self, cmd_result: Dict) -> bool:
        """Check if the command's own script datarefs were set to their intended values."""
        datarefs_set = cmd_result.get("datarefs_set", [])
        if not datarefs_set:
            self._log.append(f"  cmd fallback: no datarefs_set in command result")
            return False
        for entry in datarefs_set:
            dr = entry.get("dataref", "")
            target = entry.get("value")
            if not dr or target is None:
                continue
            # Strip array index for reading
            raw_dr = dr
            idx = None
            idx_match = re.search(r'\[(\d+)\]', dr)
            if idx_match:
                idx = int(idx_match.group(1))
                raw_dr = dr[:idx_match.start()]
            current = self._read_dataref(raw_dr, idx)
            self._log.append(f"  cmd fallback: {dr} target={target} current={current}")
            if current is None:
                return False
            try:
                if abs(float(current) - float(target)) > 0.05:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _poll_until_satisfied(self, item: ChecklistItem) -> float:
        """
        Poll conditions until met, skipped, paused, or timed out.
        Returns elapsed seconds.
        """
        max_wait = self._poll_timeout
        poll_interval = 0.5
        elapsed = 0.0

        time.sleep(0.5)
        if self._check_all_conditions(item.conditions, item.condition_logic):
            return 0.0

        while elapsed < max_wait:
            if self._stop_event.is_set() or self._skip_event.is_set():
                break
            # Respect pause
            self._pause_event.wait(timeout=1.0)
            if self.state == RunnerState.PAUSED:
                continue

            time.sleep(poll_interval)
            elapsed += poll_interval

            if self._check_all_conditions(item.conditions, item.condition_logic):
                return elapsed

            # Slow down after initial burst
            if elapsed >= 3.0:
                poll_interval = 2.0

        return elapsed

    # ----------------------------------------------------------------
    # Condition checking & command execution
    # ----------------------------------------------------------------

    def _check_all_conditions(self, conditions: List[DatarefCondition],
                              logic: str = "&&") -> bool:
        non_trigger = [c for c in conditions if not c.operator.startswith('+')]
        if not non_trigger:
            return True
        for cond in non_trigger:
            resolved = self._resolve_cross_ref(cond)
            current = self._read_dataref(cond.dataref, cond.index)
            met = _check_condition(resolved, current)
            logger.debug(f"Condition: {cond.dataref}[{cond.index}] {cond.operator} {cond.value} | raw={current} | met={met}")
            if logic == "||" and met:
                return True
            if logic == "&&" and not met:
                return False
        return logic == "&&"

    def _resolve_cross_ref(self, cond: DatarefCondition) -> DatarefCondition:
        if not cond.cross_ref:
            return cond
        ref_val = self._read_dataref(cond.cross_ref)
        if ref_val is None:
            return cond
        resolved = float(ref_val)
        if cond.value and isinstance(cond.value, str):
            try:
                expr = cond.value
                if expr.startswith('*'):
                    resolved *= float(expr[1:])
                elif expr.startswith('/'):
                    resolved /= float(expr[1:])
                elif expr.startswith('+'):
                    resolved += float(expr[1:])
                elif expr.startswith('-'):
                    resolved -= float(expr[1:])
            except (ValueError, ZeroDivisionError):
                pass
        return DatarefCondition(
            dataref=cond.dataref, operator=cond.operator,
            value=resolved, index=cond.index,
        )

    def _guess_command_text(self, item: ChecklistItem) -> List[str]:
        label = item.label.strip()
        checked = item.checked_text.strip()
        candidates = []
        # Most specific first: label + first word of checked_text
        if label and checked:
            first_word = checked.split(',')[0].split('or')[0].split('and')[0].strip()
            if first_word:
                candidates.append(f"{label} {first_word}")
        # Label + full checked_text
        if label and checked:
            candidates.append(f"{label} {checked}")
        # First word of label + checked_text
        # (e.g. "IRS mode selectors" + "NAV" -> "IRS NAV")
        if label and checked:
            label_first = label.split()[0]
            checked_first = checked.split(',')[0].split('or')[0].split('and')[0].strip()
            short = f"{label_first} {checked_first}"
            if short not in candidates:
                candidates.append(short)
        # Label without last word + checked_text
        # (e.g. "EMERGENCY EXIT LIGHTS" + "ARMED" -> "EMERGENCY EXIT ARMED")
        label_words = label.split()
        if label and checked and len(label_words) > 1:
            checked_first = checked.split(',')[0].split('or')[0].split('and')[0].strip()
            trimmed = ' '.join(label_words[:-1]) + ' ' + checked_first
            if trimmed not in candidates:
                candidates.append(trimmed)
        # Dataref-hint: extract keywords from condition datarefs to find better commands
        # e.g. condition "mixture_ratio1" -> try "MIXTURE ONE <checked_text>"
        if item.conditions and checked:
            checked_first = checked.split(',')[0].split('or')[0].split('and')[0].strip()
            for cond in item.conditions:
                if cond.operator.startswith('+'):
                    continue
                # Extract the last path segment and look for keywords
                parts = cond.dataref.rsplit('/', 1)[-1]
                # Map engine numbers: 1/2 -> ONE/TWO
                engine_map = {'1': 'ONE', '2': 'TWO', '3': 'THREE', '4': 'FOUR'}
                for digit, word in engine_map.items():
                    if digit in parts:
                        # e.g. "mixture_ratio1" -> "MIXTURE ONE IDLE"
                        base = re.sub(r'[_\d]+', ' ', parts).strip().upper()
                        base_words = base.split()
                        if base_words:
                            dr_candidate = f"{base_words[0]} {word} {checked_first}"
                            if dr_candidate not in candidates:
                                candidates.append(dr_candidate)
        # Bare label last (least specific - may match wrong variant)
        if label:
            candidates.append(label)
        return candidates

    def _try_execute_command(self, item: ChecklistItem) -> Optional[Dict]:
        if not self.commands_loader:
            return None
        from .script_executor import ScriptExecutor

        candidates = self._guess_command_text(item)
        for text in candidates:
            best_cmd, matched_tokens, operands = \
                self.commands_loader.find_command_for_input(text)
            if best_cmd and best_cmd.script:
                self._log.append(f"Command: {' '.join(matched_tokens)}")
                executor = ScriptExecutor(self.client)
                result = executor.execute(best_cmd.script, operands)
                return {
                    "command": ' '.join(matched_tokens),
                    "commands_sent": result.get("commands_sent", []),
                    "datarefs_set": result.get("datarefs_set", []),
                    "errors": result.get("errors", []),
                }
        return None

    def _try_toggle_unsatisfied(self, conditions: List[DatarefCondition]):
        """For unsatisfied conditions involving cover/switch positions, try toggling."""
        for cond in conditions:
            if cond.operator.startswith('+'):
                continue
            raw = self._read_dataref(cond.dataref, cond.index)
            if _check_condition(cond, raw):
                continue
            # cover_position datarefs need toggle commands, not direct writes
            if 'cover_position' in cond.dataref and cond.index is not None:
                # Zibo 737 cover commands follow pattern: laminar/B738/button_switch_coverNN
                # cover index mapping: index 2 -> cover02, index 3 -> cover03, etc.
                cover_cmd = f"laminar/B738/button_switch_cover{cond.index:02d}"
                self._log.append(f"Toggling cover: {cover_cmd}")
                try:
                    self.client.send_command(cover_cmd)
                    time.sleep(0.3)
                except Exception as e:
                    self._log.append(f"Cover toggle failed: {e}")

    def _write_conditions(self, conditions: List[DatarefCondition]) -> Tuple[List[Dict], List[str]]:
        actions = []
        errors = []
        for cond in conditions:
            if cond.operator.startswith('+'):
                continue
            actual = self._resolve_cross_ref(cond)
            target = _compute_write_value(actual)
            if target is None:
                continue
            try:
                if cond.index is not None:
                    ref = f"{cond.dataref}[{cond.index}]"
                    self.client.set_dataref(ref, target)
                    actions.append({"dataref": ref, "value": target})
                else:
                    self.client.set_dataref(cond.dataref, target)
                    actions.append({"dataref": cond.dataref, "value": target})
            except Exception as e:
                errors.append(f"Failed to set {cond.dataref}: {e}")
        return actions, errors

    def _read_dataref(self, dataref: str, index: Optional[int] = None) -> Any:
        val = self.client.get_dataref(dataref, timeout=1.0)
        if val is not None and index is not None and isinstance(val, list):
            if index < len(val):
                return val[index]
            return None
        return val
