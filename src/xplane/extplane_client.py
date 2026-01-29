"""
ExtPlane Client for X-Plane communication.

Connects to the ExtPlane plugin via TCP to:
- Send commands
- Read/write datarefs
- Subscribe to dataref changes
"""

import socket
import threading
import time
from typing import Callable, Dict, Optional, Any
from dataclasses import dataclass
from queue import Queue, Empty


@dataclass
class DatarefValue:
    """Holds a dataref value with metadata."""
    name: str
    value: Any
    timestamp: float


class ExtPlaneClient:
    """
    Client for ExtPlane plugin communication.

    ExtPlane protocol:
    - Commands: cmd <command_path>
    - Subscribe: sub <dataref> [accuracy]
    - Unsubscribe: unsub <dataref>
    - Set dataref: set <dataref> <value>
    - Responses: u<type> <dataref> <value>
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 51000):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        # Dataref subscriptions and values
        self._subscriptions: Dict[str, DatarefValue] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()

        # Response queue for synchronous operations
        self._response_queue: Queue = Queue()

    def connect(self) -> bool:
        """Connect to ExtPlane."""
        if self._connected:
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(None)
            self._connected = True

            # Start reader thread
            self._running = True
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()

            # Read welcome message
            time.sleep(0.1)

            return True
        except Exception as e:
            print(f"ExtPlane connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from ExtPlane."""
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
        self._socket = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _send(self, message: str) -> bool:
        """Send a message to ExtPlane."""
        if not self._connected or not self._socket:
            return False
        try:
            self._socket.sendall((message + "\n").encode('utf-8'))
            return True
        except Exception as e:
            print(f"ExtPlane send error: {e}")
            self._connected = False
            return False

    def _read_loop(self):
        """Background thread to read responses from ExtPlane."""
        buffer = ""
        while self._running and self._socket:
            try:
                self._socket.settimeout(0.5)
                data = self._socket.recv(4096)
                if not data:
                    break
                buffer += data.decode('utf-8')

                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process_line(line.strip())

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"ExtPlane read error: {e}")
                break

        self._connected = False

    def _process_line(self, line: str):
        """Process a response line from ExtPlane."""
        if not line:
            return

        # Dataref update: u<type> <dataref> <value>
        # Types: i (int), f (float), d (double), ia (int array), fa (float array), b (data)
        if line.startswith('u'):
            parts = line.split(' ', 2)
            if len(parts) >= 3:
                dtype = parts[0]
                dataref = parts[1]
                value_str = parts[2]

                # Parse value based on type
                try:
                    if dtype in ('ui', 'uf', 'ud'):
                        value = float(value_str) if '.' in value_str else int(value_str)
                    elif dtype in ('uia', 'ufa'):
                        value = [float(v) if '.' in v else int(v) for v in value_str.strip('[]').split(',')]
                    else:
                        value = value_str
                except:
                    value = value_str

                # Update stored value
                with self._lock:
                    self._subscriptions[dataref] = DatarefValue(
                        name=dataref,
                        value=value,
                        timestamp=time.time()
                    )

                    # Call callback if registered
                    if dataref in self._callbacks:
                        try:
                            self._callbacks[dataref](dataref, value)
                        except Exception as e:
                            print(f"Callback error for {dataref}: {e}")

    def send_command(self, command: str) -> bool:
        """
        Send a command to X-Plane.

        Args:
            command: Command path (e.g., "sim/lights/beacon_lights_on")
        """
        # For push_button commands, use begin/end instead of once
        if "push_button" in command:
            import time
            result = self._send(f"cmd begin {command}")
            time.sleep(0.1)  # Brief hold
            self._send(f"cmd end {command}")
            return result
        return self._send(f"cmd once {command}")

    def send_command_begin(self, command: str) -> bool:
        """Begin holding a command (for continuous commands)."""
        return self._send(f"cmd begin {command}")

    def send_command_end(self, command: str) -> bool:
        """End holding a command."""
        return self._send(f"cmd end {command}")

    def subscribe(self, dataref: str, accuracy: float = 0.0, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to a dataref.

        Args:
            dataref: Dataref path (e.g., "sim/cockpit/electrical/beacon_lights_on")
            accuracy: Only send updates when value changes by this amount
            callback: Optional callback function(dataref, value) for updates
        """
        if callback:
            with self._lock:
                self._callbacks[dataref] = callback

        if accuracy > 0:
            return self._send(f"sub {dataref} {accuracy}")
        else:
            return self._send(f"sub {dataref}")

    def unsubscribe(self, dataref: str) -> bool:
        """Unsubscribe from a dataref."""
        with self._lock:
            self._callbacks.pop(dataref, None)
            self._subscriptions.pop(dataref, None)
        return self._send(f"unsub {dataref}")

    def set_dataref(self, dataref: str, value: Any) -> bool:
        """
        Set a dataref value.

        Args:
            dataref: Dataref path
            value: Value to set (int, float, or list for arrays)
        """
        if isinstance(value, list):
            value_str = '[' + ','.join(str(v) for v in value) + ']'
        else:
            value_str = str(value)
        return self._send(f"set {dataref} {value_str}")

    def get_dataref(self, dataref: str, timeout: float = 1.0) -> Optional[Any]:
        """
        Get current value of a subscribed dataref.

        Args:
            dataref: Dataref path
            timeout: How long to wait for value

        Returns:
            Current value or None if not available
        """
        with self._lock:
            if dataref in self._subscriptions:
                return self._subscriptions[dataref].value

        # If not subscribed, subscribe temporarily and wait
        self.subscribe(dataref)
        time.sleep(0.2)  # Give ExtPlane time to send initial value
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if dataref in self._subscriptions:
                    value = self._subscriptions[dataref].value
                    # Don't unsubscribe - keep it for potential re-reads
                    return value
            time.sleep(0.05)

        return None

    def get_subscribed_value(self, dataref: str) -> Optional[DatarefValue]:
        """Get the stored value for a subscribed dataref."""
        with self._lock:
            return self._subscriptions.get(dataref)

    def execute_script_command(self, script_line: str) -> bool:
        """
        Execute a single command from an XPRemote script.

        Handles:
        - cmd.sendCommand("path")
        - cmd.setDataRefValue("path", value)
        """
        script_line = script_line.strip()

        # cmd.sendCommand("...")
        if 'sendCommand' in script_line:
            import re
            match = re.search(r'sendCommand\s*\(\s*["\']([^"\']+)["\']\s*\)', script_line)
            if match:
                return self.send_command(match.group(1))

        # cmd.setDataRefValue("...", value)
        if 'setDataRefValue' in script_line:
            import re
            match = re.search(r'setDataRefValue\s*\(\s*["\']([^"\']+)["\']\s*,\s*([^)]+)\)', script_line)
            if match:
                dataref = match.group(1)
                value_str = match.group(2).strip()
                try:
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except:
                    value = value_str
                return self.set_dataref(dataref, value)

        return False


# Global client instance
_client: Optional[ExtPlaneClient] = None


def get_client() -> ExtPlaneClient:
    """Get or create the global ExtPlane client."""
    global _client
    if _client is None:
        _client = ExtPlaneClient()
    return _client
