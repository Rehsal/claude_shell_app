"""
Command categorization for the X-Plane Copilot.

Groups commands by functional category for the web UI.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from .commands_loader import Command, CommandsLoader


# Category definitions with keyword patterns
CATEGORY_PATTERNS = {
    "Lights": [
        "BEACON", "LANDING_LIGHT", "TAXI_LIGHT", "NAV_LIGHTS", "STROBE",
        "LOGO", "WING LIGHTS", "RUNWAY LIGHTS", "POSITION LIGHTS",
        "NOSE LIGHTS", "ICE LIGHTS", "CABIN LIGHTS", "DIMMER LIGHTS",
        "ACCESS LIGHTS", "WHEEL_WELL LIGHTS", "COCKPIT_LIGHTING",
        "WHITE_DOME_LIGHT", "LIGHTS"
    ],
    "Autopilot": [
        "AUTOPILOT", "HEADING", "ALTITUDE", "SPEED", "VERTICAL_SPEED",
        "VNAV", "LNAV", "APPROACH", "LOCALIZER", "GLIDE_SLOPE",
        "FLIGHT_DIRECTOR", "FLIGHT_LEVEL", "AUTOTHROTTLE", "EXPEDITE",
        "BACK_COURSE", "TOGA", "GO_AROUND"
    ],
    "Engines": [
        "ENGINE", "MIXTURE", "FUEL_PUMP", "FUEL_SELECTOR", "STARTER",
        "IGNITION", "BOOST_PUMP", "COWL_FLAPS", "CARB_HEAT",
        "AUTO_FEATHER", "AUTOIGNITION", "FUEL", "THROTTLE", "POWER"
    ],
    "Electrical": [
        "BATTERY", "GENERATOR", "ALTERNATOR", "APU", "EXTERNAL_POWER",
        "SOURCE", "AVIONICS_MASTER", "BUS"
    ],
    "Radios / Navigation": [
        "COM_ONE", "COM_TWO", "NAV_ONE", "NAV_TWO", "ADF", "TRANSPONDER",
        "ILS", "VOR", "NDB", "DME", "SQUAWK", "TCAS", "FREQUENCY",
        "RADIO", "FMS", "COURSE", "OBS"
    ],
    "Flight Controls": [
        "FLAPS", "GEAR", "LANDING_GEAR", "TRIM", "ELEVATOR_TRIM",
        "SPEED BRAKE", "SPOILERS", "SPEEDBRAKE", "YAW_DAMPER"
    ],
    "Environmental": [
        "PACKS", "A/C", "BLEED", "CROSS_BLEED", "PRESSURE", "OXYGEN",
        "SEAT_BELTS", "ICING", "WINDOW", "WIPERS"
    ],
    "Ground": [
        "BRAKE", "PARKING", "AUTOBRAKES", "PUSHBACK", "TOWING", "CHOCKS",
        "DOOR", "GPU", "GROUND"
    ],
    "Procedures": [
        "PROCEDURE", "CHECKLIST", "POWER_UP", "SHUTDOWN", "TAKEOFF",
        "LANDING", "TAXI", "STARTUP", "CLEANUP"
    ],
    "Display / Map": [
        "MAP", "ECAM", "ND", "PFD", "MFD", "EFIS", "RANGE", "MODE",
        "OVERLAY", "TERRAIN", "WEATHER"
    ],
}


@dataclass
class CategorizedCommand:
    """A command with its category and metadata."""
    command: Command
    category: str
    requires_value: bool = False
    value_type: Optional[str] = None  # 'number', 'frequency', 'identifier', etc.


def categorize_commands(loader: CommandsLoader) -> Dict[str, List[CategorizedCommand]]:
    """
    Categorize all commands by functional group.

    Args:
        loader: CommandsLoader instance with loaded commands

    Returns:
        Dict mapping category name to list of CategorizedCommand
    """
    categories: Dict[str, List[CategorizedCommand]] = {
        cat: [] for cat in CATEGORY_PATTERNS.keys()
    }
    categories["Other"] = []  # For uncategorized commands

    for cmd in loader.commands.values():
        category = _get_command_category(cmd)
        requires_value, value_type = _check_value_requirement(cmd)

        cat_cmd = CategorizedCommand(
            command=cmd,
            category=category,
            requires_value=requires_value,
            value_type=value_type
        )
        categories[category].append(cat_cmd)

    # Sort commands within each category
    for cat in categories:
        categories[cat].sort(key=lambda c: c.command.tokens)

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}

    return categories


def _get_command_category(cmd: Command) -> str:
    """Determine the category for a command based on its tokens."""
    tokens_upper = cmd.tokens.upper()

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in tokens_upper:
                return category

    return "Other"


def _check_value_requirement(cmd: Command) -> tuple:
    """
    Check if a command requires a value input.

    Returns:
        Tuple of (requires_value: bool, value_type: str or None)
    """
    tokens = cmd.tokens.upper()

    # Check for value placeholders in token string
    if "NUMBER" in tokens:
        return True, "number"
    if "DEGREES" in tokens:
        return True, "degrees"
    if "FREQUENCY" in tokens or "NAV_FREQUENCY" in tokens or "COM_FREQUENCY" in tokens:
        return True, "frequency"
    if "IDENTIFIER" in tokens:
        return True, "identifier"
    if "FLIGHT_LEVEL" in tokens:
        return True, "flight_level"
    if "ALTIMETER_SETTING" in tokens:
        return True, "altimeter"

    return False, None


def get_category_summary(loader: CommandsLoader) -> List[dict]:
    """
    Get a summary of all categories with command counts.

    Returns:
        List of dicts with category name and count.
    """
    categories = categorize_commands(loader)
    return [
        {"name": cat, "count": len(cmds)}
        for cat, cmds in sorted(categories.items(), key=lambda x: x[0])
    ]
