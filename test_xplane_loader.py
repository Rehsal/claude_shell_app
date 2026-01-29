"""
Test script for X-Plane commands loader.

Run from project root:
    python test_xplane_loader.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.xplane import XPlaneConfig, CommandsLoader


def main():
    print("=" * 60)
    print("X-Plane Commands Loader Test")
    print("=" * 60)

    # Load config
    print("\n1. Loading configuration...")
    config = XPlaneConfig()
    print(f"   Config path: {config._config_path}")
    print(f"   Commands XML: {config.commands_xml_path}")
    print(f"   Active profile: {config.active_profile}")

    # Validate
    print("\n2. Validating configuration...")
    validation = config.validate()
    if validation["valid"]:
        print("   Configuration is valid!")
    else:
        print("   Configuration errors:")
        for err in validation["errors"]:
            print(f"   - {err}")
        return 1

    # Load commands
    print("\n3. Loading commands.xml...")
    loader = CommandsLoader(config)
    stats = loader.get_stats()

    print(f"   Loaded from: {stats['xml_path']}")
    print(f"   Last loaded: {stats['last_loaded']}")
    print(f"   Profiles: {stats['profiles_count']}")
    print(f"   Profile names: {stats['profile_names']}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Merged tokens: {stats['merged_tokens']}")
    print(f"   Total commands: {stats['total_commands']}")
    print(f"   Merged commands: {stats['merged_commands']}")

    # Test token lookup
    print("\n4. Testing token lookup...")
    test_tokens = ["BEACON", "ON", "OFF", "LANDING_LIGHT", "AUTOPILOT", "FLAPS"]
    for name in test_tokens:
        token = loader.get_token(name)
        if token:
            pattern_preview = token.pattern[:50] + "..." if len(token.pattern) > 50 else token.pattern
            print(f"   {name}: phrase='{token.phrase}', pattern='{pattern_preview}'")
        else:
            print(f"   {name}: NOT FOUND")

    # Test command lookup
    print("\n5. Testing command lookup...")
    test_commands = ["BEACON ON", "BEACON OFF", "LANDING_LIGHT ON", "FLAPS NUMBER"]
    for cmd_tokens in test_commands:
        cmd = loader.get_command(cmd_tokens)
        if cmd:
            script_preview = cmd.script[:80].replace('\n', ' ')
            print(f"   '{cmd_tokens}' [{cmd.profile}]: {script_preview}...")
        else:
            print(f"   '{cmd_tokens}': NOT FOUND")

    # Test natural language matching
    print("\n6. Testing natural language matching...")
    test_inputs = [
        "beacon on",
        "turn the beacon on",
        "landing light off",
        "landing lights off",
        "flaps five",
        "flaps 5",
        "set heading one two zero",
        "heading 120",
        "altitude three five thousand",
    ]
    for text in test_inputs:
        cmd, matched, operands = loader.find_command_for_input(text)
        cmd_str = cmd.tokens if cmd else "NO MATCH"
        op_str = f", operands={operands}" if operands else ""
        print(f"   '{text}' -> tokens={matched}{op_str} -> '{cmd_str}'")

    # Show available profiles
    print("\n7. Available profiles:")
    for name in loader.profile_names:
        profile = loader.get_profile(name)
        is_active = " (ACTIVE)" if name == loader.active_profile_name else ""
        print(f"   - {name}{is_active}: {len(profile.tokens)} tokens, {len(profile.commands)} commands")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
