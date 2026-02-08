"""
Shared command-script execution with dataref pre-subscription.

ExtPlane requires datarefs to be subscribed before set/get calls work.
This module wraps ScriptExecutor with the subscribe → execute → unsubscribe
pattern used by the Commands page, so any caller (Commands, Checklist, etc.)
gets reliable execution.
"""

import re
import time
from typing import Dict, List

from .extplane_client import ExtPlaneClient
from .script_executor import ScriptExecutor


def execute_command_script(client: ExtPlaneClient, script: str, operands: dict) -> Dict:
    """Execute a commands.xml script with proper dataref pre-subscription.

    Returns dict with keys: commands_sent, datarefs_set, errors.
    """
    # Extract all datarefs referenced in the script
    datarefs = set()
    for pattern in (
        r'setDataRefValue\s*\(\s*["\']([^"\']+)["\']',
        r'setDataRefArrayValue\s*\(\s*["\']([^"\']+)["\']',
        r'getDataRefValue\s*\(\s*["\']([^"\']+)["\']',
    ):
        for m in re.finditer(pattern, script):
            datarefs.add(m.group(1))

    datarefs = list(datarefs)[:10]

    # Pre-subscribe so ExtPlane tracks these datarefs
    for dr in datarefs:
        try:
            client.subscribe(dr)
        except Exception as e:
            print(f"[command_executor] Subscribe error for {dr}: {e}")

    time.sleep(0.2)

    # Execute
    executor = ScriptExecutor(client)
    result = executor.execute(script, operands)

    time.sleep(0.3)

    # Unsubscribe
    for dr in datarefs:
        try:
            client.unsubscribe(dr)
        except Exception:
            pass

    return result
