"""
Script executor for XPRemote command scripts.

Executes the JavaScript-like scripts from commands.xml using ExtPlane.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple
from .extplane_client import ExtPlaneClient


class ScriptExecutor:
    """
    Executes XPRemote command scripts.

    Supports:
    - cmd.sendCommand("path")
    - cmd.setDataRefValue("path", value)
    - cmd.getDataRefValue("path")
    - cmd.setDataRefArrayValue("path", index, value)
    - cmd.integerOperand, cmd.floatOperand
    - cmd.toIntegerOperand(value)
    - Variables: var x = value;
    - Conditionals: if/else
    - Loops: while
    - Basic math: +, -, *, /
    - Comparisons: ==, !=, <, >, <=, >=, lt, gt, eq
    """

    # Remap Zibo datarefs that ExtPlane writes don't stick reliably.
    # Values can be:
    #   - string: toggle command (sends if current != target)
    #   - tuple (on_cmd, off_cmd): directional commands based on target value
    _DATAREF_TO_COMMAND = {
        # Fuel pumps - true toggles
        "laminar/B738/fuel/fuel_tank_pos_lft1": "laminar/B738/toggle_switch/fuel_pump_lft1",
        "laminar/B738/fuel/fuel_tank_pos_lft2": "laminar/B738/toggle_switch/fuel_pump_lft2",
        "laminar/B738/fuel/fuel_tank_pos_ctr1": "laminar/B738/toggle_switch/fuel_pump_ctr1",
        "laminar/B738/fuel/fuel_tank_pos_ctr2": "laminar/B738/toggle_switch/fuel_pump_ctr2",
        "laminar/B738/fuel/fuel_tank_pos_rgt1": "laminar/B738/toggle_switch/fuel_pump_rgt1",
        "laminar/B738/fuel/fuel_tank_pos_rgt2": "laminar/B738/toggle_switch/fuel_pump_rgt2",
        # Window heat - true toggles
        "laminar/B738/ice/window_heat_l_side_pos": "laminar/B738/toggle_switch/window_heat_l_side",
        "laminar/B738/ice/window_heat_l_fwd_pos": "laminar/B738/toggle_switch/window_heat_l_fwd",
        "laminar/B738/ice/window_heat_r_side_pos": "laminar/B738/toggle_switch/window_heat_r_side",
        "laminar/B738/ice/window_heat_r_fwd_pos": "laminar/B738/toggle_switch/window_heat_r_fwd",
        # Landing lights - on/off pairs
        "laminar/B738/switch/land_lights_left_pos": ("laminar/B738/switch/land_lights_left_on", "laminar/B738/switch/land_lights_left_off"),
        "laminar/B738/switch/land_lights_right_pos": ("laminar/B738/switch/land_lights_right_on", "laminar/B738/switch/land_lights_right_off"),
        # Chocks - true toggle
        "laminar/B738/fms/chock_status": "laminar/B738/toggle_switch/chock",
    }

    def __init__(self, client: ExtPlaneClient):
        self.client = client
        self.variables: Dict[str, Any] = {}
        self.operands: Dict[str, Any] = {}
        self.commands_sent: List[str] = []
        self.datarefs_set: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def execute(self, script: str, operands: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a script and return results.

        Args:
            script: The script code to execute
            operands: Optional operands (number, etc.) from voice input

        Returns:
            Dict with commands_sent, datarefs_set, errors
        """
        # Reset state
        self.variables = {}
        self.operands = operands or {}
        self.commands_sent = []
        self.datarefs_set = []
        self.errors = []

        # Normalize script
        script = self._normalize_script(script)

        # Execute
        try:
            self._execute_block(script)
        except Exception as e:
            self.errors.append(f"Execution error: {str(e)}")

        return {
            "commands_sent": self.commands_sent,
            "datarefs_set": self.datarefs_set,
            "errors": self.errors,
            "success": len(self.errors) == 0
        }

    def _normalize_script(self, script: str) -> str:
        """Normalize script for easier parsing."""
        # Replace XML entities
        script = script.replace("&lt;", "<").replace("&gt;", ">")
        script = script.replace("&amp;", "&").replace("&quot;", '"')

        # Normalize line endings
        script = script.replace('\r\n', '\n').replace('\r', '\n')

        # Remove // style comments (but not inside strings)
        lines = script.split('\n')
        cleaned_lines = []
        for line in lines:
            # Simple comment removal - strip everything after // not in a string
            # For simplicity, just check if // appears before any quote
            if '//' in line:
                idx = line.find('//')
                quote_idx = min(
                    line.find('"') if '"' in line else len(line),
                    line.find("'") if "'" in line else len(line)
                )
                if idx < quote_idx:
                    line = line[:idx]
            cleaned_lines.append(line)
        script = '\n'.join(cleaned_lines)

        # Replace comparison keywords with symbols
        script = re.sub(r'\blt\b', '<', script)
        script = re.sub(r'\bgt\b', '>', script)
        script = re.sub(r'\ble\b', '<=', script)
        script = re.sub(r'\bge\b', '>=', script)
        script = re.sub(r'\beq\b', '==', script)
        script = re.sub(r'\bne\b', '!=', script)

        return script

    def _execute_block(self, code: str):
        """Execute a block of code."""
        # Split into statements
        statements = self._split_statements(code)

        i = 0
        while i < len(statements):
            stmt = statements[i].strip()
            if not stmt:
                i += 1
                continue

            # Skip comments
            if stmt.startswith('//') or stmt.startswith('\\'):
                i += 1
                continue

            # Handle if statement
            if stmt.startswith('if'):
                i = self._handle_if(statements, i)
            # Handle while loop
            elif stmt.startswith('while'):
                i = self._handle_while(statements, i)
            # Handle var declaration
            elif stmt.startswith('var '):
                self._handle_var(stmt)
                i += 1
            # Handle assignment
            elif '=' in stmt and not stmt.startswith('if') and not stmt.startswith('while'):
                self._handle_assignment(stmt)
                i += 1
            # Handle cmd calls
            elif 'cmd.' in stmt:
                self._handle_cmd(stmt)
                i += 1
            else:
                i += 1

    def _split_statements(self, code: str) -> List[str]:
        """Split code into statements, handling braces and parentheses."""
        statements = []
        current = ""
        brace_depth = 0
        paren_depth = 0

        for char in code:
            if char == '{':
                brace_depth += 1
                current += char
            elif char == '}':
                brace_depth -= 1
                current += char
            elif char == '(':
                paren_depth += 1
                current += char
            elif char == ')':
                paren_depth -= 1
                current += char
            elif char == ';' and brace_depth == 0 and paren_depth == 0:
                statements.append(current.strip())
                current = ""
            elif char == '\n' and brace_depth == 0 and paren_depth == 0:
                if current.strip():
                    statements.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            statements.append(current.strip())

        return statements

    def _extract_condition_and_block(self, text: str) -> Tuple[str, str, str]:
        """
        Extract condition and block from an if statement.
        Returns (condition, block, remaining_text).
        Handles nested parentheses in condition.
        """
        # Find opening paren for condition
        paren_start = text.find('(')
        if paren_start == -1:
            return "", "", text

        # Find matching closing paren (count nested parens)
        paren_depth = 1
        pos = paren_start + 1
        while pos < len(text) and paren_depth > 0:
            if text[pos] == '(':
                paren_depth += 1
            elif text[pos] == ')':
                paren_depth -= 1
            pos += 1

        condition = text[paren_start + 1:pos - 1]

        # Find opening brace for block
        brace_start = text.find('{', pos)
        if brace_start == -1:
            return condition, "", text[pos:]

        # Find matching closing brace
        brace_depth = 1
        pos = brace_start + 1
        while pos < len(text) and brace_depth > 0:
            if text[pos] == '{':
                brace_depth += 1
            elif text[pos] == '}':
                brace_depth -= 1
            pos += 1

        block = text[brace_start + 1:pos - 1]
        remaining = text[pos:].strip()

        return condition, block, remaining

    def _handle_if(self, statements: List[str], index: int) -> int:
        """Handle if/else if/else statement chains."""
        stmt = statements[index]
        remaining = stmt
        executed = False

        while remaining and not executed:
            remaining = remaining.strip()

            # Check for 'else if' or 'if'
            if remaining.startswith('else if'):
                remaining = remaining[4:].strip()  # Remove 'else'
                condition, block, remaining = self._extract_condition_and_block(remaining)
            elif remaining.startswith('if'):
                condition, block, remaining = self._extract_condition_and_block(remaining)
            elif remaining.startswith('else'):
                # Final else block
                brace_start = remaining.find('{')
                if brace_start != -1:
                    brace_depth = 1
                    pos = brace_start + 1
                    while pos < len(remaining) and brace_depth > 0:
                        if remaining[pos] == '{':
                            brace_depth += 1
                        elif remaining[pos] == '}':
                            brace_depth -= 1
                        pos += 1
                    else_block = remaining[brace_start + 1:pos - 1]
                    self._execute_block(else_block)
                    executed = True
                break
            else:
                break

            if condition:
                if self._evaluate_condition(condition):
                    self._execute_block(block)
                    executed = True
                # Otherwise continue to check else if / else
            else:
                break

        return index + 1

    def _handle_while(self, statements: List[str], index: int) -> int:
        """Handle while loop."""
        stmt = statements[index]

        # Extract condition
        match = re.match(r'while\s*\((.+?)\)\s*\{', stmt)
        if not match:
            return index + 1

        condition = match.group(1)

        # Find the loop block
        block_start = stmt.find('{')
        brace_count = 1
        pos = block_start + 1

        while pos < len(stmt) and brace_count > 0:
            if stmt[pos] == '{':
                brace_count += 1
            elif stmt[pos] == '}':
                brace_count -= 1
            pos += 1

        loop_block = stmt[block_start + 1:pos - 1]

        # Execute loop (with safety limit)
        iterations = 0
        max_iterations = 100

        while self._evaluate_condition(condition) and iterations < max_iterations:
            self._execute_block(loop_block)
            iterations += 1

        return index + 1

    def _handle_var(self, stmt: str):
        """Handle variable declaration."""
        # var x = value;
        match = re.match(r'var\s+(\w+)\s*=\s*(.+)', stmt.rstrip(';'))
        if match:
            name = match.group(1)
            value = self._evaluate_expression(match.group(2))
            self.variables[name] = value

    def _handle_assignment(self, stmt: str):
        """Handle variable assignment."""
        # x = value;
        match = re.match(r'(\w+)\s*=\s*(.+)', stmt.rstrip(';'))
        if match:
            name = match.group(1)
            value = self._evaluate_expression(match.group(2))
            self.variables[name] = value

    # Light commands that should be converted to dataref sets (workaround for ExtPlane)
    # These standard X-Plane commands don't work properly on some aircraft via ExtPlane
    LIGHT_COMMAND_TO_DATAREF = {
        "sim/lights/beacon_lights_on": ("sim/cockpit/electrical/beacon_lights_on", 1),
        "sim/lights/beacon_lights_off": ("sim/cockpit/electrical/beacon_lights_on", 0),
        "sim/lights/landing_lights_on": ("sim/cockpit/electrical/landing_lights_on", 1),
        "sim/lights/landing_lights_off": ("sim/cockpit/electrical/landing_lights_on", 0),
        "sim/lights/strobe_lights_on": ("sim/cockpit/electrical/strobe_lights_on", 1),
        "sim/lights/strobe_lights_off": ("sim/cockpit/electrical/strobe_lights_on", 0),
        "sim/lights/nav_lights_on": ("sim/cockpit/electrical/nav_lights_on", 1),
        "sim/lights/nav_lights_off": ("sim/cockpit/electrical/nav_lights_on", 0),
        "sim/lights/taxi_lights_on": ("sim/cockpit/electrical/taxi_light_on", 1),
        "sim/lights/taxi_lights_off": ("sim/cockpit/electrical/taxi_light_on", 0),
    }

    def _handle_cmd(self, stmt: str):
        """Handle cmd.* calls."""
        stmt = stmt.rstrip(';')

        # cmd.pause(ms) - sleep for milliseconds
        match = re.search(r'cmd\.pause\s*\(\s*(\d+)\s*\)', stmt)
        if match:
            ms = int(match.group(1))
            time.sleep(ms / 1000.0)
            return

        # cmd.sendCommandBegin("path") - begin holding a command
        match = re.search(r'cmd\.sendCommandBegin\s*\(\s*["\']([^"\']+)["\']\s*\)', stmt)
        if match:
            command = match.group(1)
            if self.client.send_command_begin(command):
                self.commands_sent.append(f"begin:{command}")
            return

        # cmd.sendCommandEnd("path") - end holding a command
        match = re.search(r'cmd\.sendCommandEnd\s*\(\s*["\']([^"\']+)["\']\s*\)', stmt)
        if match:
            command = match.group(1)
            if self.client.send_command_end(command):
                self.commands_sent.append(f"end:{command}")
            return

        # cmd.sendCommand("path") - static string
        match = re.search(r'cmd\.sendCommand\s*\(\s*["\']([^"\']+)["\']\s*\)', stmt)
        if match:
            command = match.group(1)
            # Check if this is a light command that needs dataref workaround
            if command in self.LIGHT_COMMAND_TO_DATAREF:
                dataref, value = self.LIGHT_COMMAND_TO_DATAREF[command]
                if self.client.set_dataref(dataref, value):
                    self.commands_sent.append(f"{command} -> set {dataref}={value}")
                    self.datarefs_set.append({"dataref": dataref, "value": value})
            else:
                if self.client.send_command(command):
                    self.commands_sent.append(command)
            return

        # cmd.setDataRefValue("path", value)
        # Use greedy (.+) to capture nested parens like cmd.toIntegerOperand(1000.0 * cmd.floatOperand)
        match = re.search(r'cmd\.setDataRefValue\s*\(\s*["\']([^"\']+)["\']\s*,\s*(.+)\s*\)', stmt)
        if match:
            dataref = match.group(1)
            value = self._evaluate_expression(match.group(2))
            # Remap datarefs to commands (toggle or on/off pairs)
            if dataref in self._DATAREF_TO_COMMAND:
                mapping = self._DATAREF_TO_COMMAND[dataref]
                current = self.client.get_dataref(dataref)
                try:
                    current_val = float(current) if current is not None else -1
                except (TypeError, ValueError):
                    current_val = -1
                target_val = float(value) if value is not None else 1

                if isinstance(mapping, tuple):
                    # ON/OFF command pair: (on_cmd, off_cmd)
                    on_cmd, off_cmd = mapping
                    if target_val > 0.5:
                        if current_val > 0.5:
                            self.commands_sent.append(f"SKIP {on_cmd} (already on)")
                            return
                        command = on_cmd
                    else:
                        if current_val < 0.5:
                            self.commands_sent.append(f"SKIP {off_cmd} (already off)")
                            return
                        command = off_cmd
                else:
                    # Toggle command: only send if state doesn't match
                    command = mapping
                    if abs(current_val - target_val) < 0.1:
                        self.commands_sent.append(f"SKIP {command} (already at {current_val})")
                        return

                if self.client.send_command(command):
                    self.commands_sent.append(f"{command} (remapped from {dataref})")
                    time.sleep(0.2)  # Let command take effect
                return
            if self.client.set_dataref(dataref, value):
                self.datarefs_set.append({"dataref": dataref, "value": value})
                time.sleep(0.15)  # let Zibo plugin process each write
            return

        # cmd.setDataRefArrayValue("path", index, value)
        # Use greedy match for value to handle nested parens
        match = re.search(r'cmd\.setDataRefArrayValue\s*\(\s*["\']([^"\']+)["\']\s*,\s*(\d+)\s*,\s*(.+)\s*\)', stmt)
        if match:
            dataref = match.group(1)
            index = int(self._evaluate_expression(match.group(2)))
            index = max(0, index - 1)  # XPRemote uses 1-based indices
            value = self._evaluate_expression(match.group(3))
            # For array values, we need to format as [index]=value
            # ExtPlane uses: set dataref[index] value
            array_dataref = f"{dataref}[{index}]"
            if self.client.set_dataref(array_dataref, value):
                self.datarefs_set.append({"dataref": array_dataref, "value": value})
                time.sleep(0.15)  # let Zibo plugin process each write
            return

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition expression."""
        condition = condition.strip()

        # Handle logical OR (||) - split and evaluate parts
        if '||' in condition:
            parts = condition.split('||', 1)
            return self._evaluate_condition(parts[0]) or self._evaluate_condition(parts[1])

        # Handle logical AND (&&) - split and evaluate parts
        if '&&' in condition:
            parts = condition.split('&&', 1)
            return self._evaluate_condition(parts[0]) and self._evaluate_condition(parts[1])

        # Handle 'and' keyword (with any whitespace around it)
        and_match = re.search(r'\s+and\s+', condition, re.IGNORECASE)
        if and_match:
            parts = re.split(r'\s+and\s+', condition, 1, re.IGNORECASE)
            return self._evaluate_condition(parts[0]) and self._evaluate_condition(parts[1])

        # Handle 'or' keyword (with any whitespace around it)
        or_match = re.search(r'\s+or\s+', condition, re.IGNORECASE)
        if or_match:
            parts = re.split(r'\s+or\s+', condition, 1, re.IGNORECASE)
            return self._evaluate_condition(parts[0]) or self._evaluate_condition(parts[1])

        # Handle comparisons
        for op, py_op in [('==', '=='), ('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>')]:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left = self._evaluate_expression(parts[0])
                    right = self._evaluate_expression(parts[1])
                    try:
                        if op == '==':
                            return float(left) == float(right)
                        elif op == '!=':
                            return float(left) != float(right)
                        elif op == '<':
                            return float(left) < float(right)
                        elif op == '>':
                            return float(left) > float(right)
                        elif op == '<=':
                            return float(left) <= float(right)
                        elif op == '>=':
                            return float(left) >= float(right)
                    except (ValueError, TypeError):
                        return False

        # Default to truthy evaluation
        val = self._evaluate_expression(condition)
        return bool(val)

    def _evaluate_expression(self, expr: str) -> Any:
        """Evaluate an expression and return its value."""
        expr = expr.strip()

        # Handle TOP-LEVEL arithmetic FIRST (operators outside parentheses)
        # This ensures "cmd.getDataRefValue(...) * 100" is evaluated as arithmetic
        pos, _ = self._find_top_level_operator(expr, [' + ', ' - ', ' * ', ' / ', ' % ', '+', '-', '*', '/', '%'])
        if pos >= 0:
            return self._evaluate_arithmetic(expr)

        # Handle cmd.toIntegerOperand(value) BEFORE getDataRefValue - need to properly extract nested parens
        # This ensures cmd.toIntegerOperand(cmd.getDataRefValue(...) * 100) works correctly
        if 'cmd.toIntegerOperand' in expr:
            start = expr.find('cmd.toIntegerOperand')
            paren_start = expr.find('(', start)
            if paren_start >= 0:
                # Find matching closing paren
                depth = 1
                pos = paren_start + 1
                while pos < len(expr) and depth > 0:
                    if expr[pos] == '(':
                        depth += 1
                    elif expr[pos] == ')':
                        depth -= 1
                    pos += 1
                inner = expr[paren_start + 1:pos - 1].strip()
                inner_val = self._evaluate_expression(inner)
                try:
                    return int(float(inner_val))
                except (ValueError, TypeError):
                    return 0

        # Handle cmd.getDataRefArrayValue("path", index) - read array element
        match = re.search(r'cmd\.getDataRefArrayValue\s*\(\s*["\']([^"\']+)["\']\s*,\s*([^)]+)\s*\)', expr)
        if match:
            dataref = match.group(1)
            index = int(self._evaluate_expression(match.group(2).strip()))
            index = max(0, index - 1)  # XPRemote uses 1-based indices
            value = self.client.get_dataref(dataref, timeout=1.0)
            if isinstance(value, list) and index < len(value):
                return value[index]
            return 0

        # Handle cmd.getDataRefValue("path") or cmd.getDataRefValue("path", default)
        match = re.search(r'cmd\.getDataRefValue\s*\(\s*["\']([^"\']+)["\']\s*(?:,\s*([^)]+))?\s*\)', expr)
        if match:
            dataref = match.group(1)
            default_str = match.group(2)
            default_val = 0
            if default_str:
                # Evaluate the default as an expression (could be "cmd.barometricFactor * cmd.floatOperand")
                default_val = self._evaluate_expression(default_str.strip())
            # Handle array index in path (e.g. "dataref[9]")
            idx_match = re.match(r'^(.+)\[(\d+)\]$', dataref)
            if idx_match:
                base = idx_match.group(1)
                index = int(idx_match.group(2))
                index = max(0, index - 1)  # XPRemote uses 1-based indices
                value = self.client.get_dataref(base, timeout=1.0)
                if isinstance(value, list) and index < len(value):
                    return value[index]
                return default_val
            value = self.client.get_dataref(dataref, timeout=1.0)
            return value if value is not None else default_val

        # Handle cmd.integerOperand (exact match only)
        if expr == 'cmd.integerOperand':
            return int(self.operands.get('number', 0))

        # Handle cmd.floatOperand (exact match only)
        if expr == 'cmd.floatOperand':
            return float(self.operands.get('number', 0.0))

        # Handle cmd.barometricFactor (1.0 for inHg, datarefs use inHg natively)
        if expr == 'cmd.barometricFactor':
            return 1.0

        # Handle variables
        if expr in self.variables:
            return self.variables[expr]

        # Handle numeric literals
        try:
            if '.' in expr:
                return float(expr)
            else:
                return int(expr)
        except ValueError:
            pass

        # Handle string literals
        match = re.match(r'^["\'](.+)["\']$', expr)
        if match:
            return match.group(1)

        return expr

    def _find_top_level_operator(self, expr: str, operators: List[str]) -> Tuple[int, str]:
        """Find the first operator at top level (not inside parentheses).
        Returns (position, operator) or (-1, '') if not found."""
        paren_depth = 0
        for i, char in enumerate(expr):
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif paren_depth == 0:
                for op in operators:
                    if expr[i:i+len(op)] == op:
                        return i, op
        return -1, ''

    def _split_at_top_level(self, expr: str, op: str, pos: int) -> Tuple[str, str]:
        """Split expression at the given operator position."""
        left = expr[:pos].strip()
        right = expr[pos+len(op):].strip()
        return left, right

    def _evaluate_arithmetic(self, expr: str) -> float:
        """Evaluate arithmetic expression, respecting parentheses."""
        expr = expr.strip()

        # Check operators in order of precedence (lowest first for left-to-right)
        # Addition and subtraction (lowest precedence)
        pos, op = self._find_top_level_operator(expr, [' + ', ' - ', '+', '-'])
        if pos >= 0 and op in [' + ', '+']:
            left, right = self._split_at_top_level(expr, op, pos)
            if left:  # Avoid splitting on unary minus
                return float(self._evaluate_expression(left)) + float(self._evaluate_expression(right))
        if pos >= 0 and op in [' - ', '-']:
            left, right = self._split_at_top_level(expr, op, pos)
            if left:  # Avoid splitting on unary minus
                return float(self._evaluate_expression(left)) - float(self._evaluate_expression(right))

        # Multiplication, division, modulo (higher precedence)
        pos, op = self._find_top_level_operator(expr, [' * ', ' / ', ' % ', '*', '/', '%'])
        if pos >= 0:
            left, right = self._split_at_top_level(expr, op, pos)
            left_val = float(self._evaluate_expression(left))
            right_val = float(self._evaluate_expression(right))
            if op in [' * ', '*']:
                return left_val * right_val
            elif op in [' / ', '/']:
                return left_val / right_val if right_val != 0 else 0
            elif op in [' % ', '%']:
                return left_val % right_val if right_val != 0 else 0

        # No top-level operator found, evaluate as expression
        return float(self._evaluate_expression(expr))
