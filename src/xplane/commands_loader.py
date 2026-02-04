"""
XPRemote Commands.xml Loader

Parses and provides structured access to:
- Token definitions (with synonyms for voice recognition)
- Command definitions (with script bodies)
- Aircraft profiles

The commands.xml file is subject to external changes and is loaded
dynamically from the configured path.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from .config_loader import XPlaneConfig


@dataclass
class Token:
    """Represents a token definition for voice/text recognition."""
    name: str
    phrase: str
    pattern: str = ""  # The raw pattern from XML (regex or synonyms)

    def matches(self, word: str) -> Tuple[bool, int]:
        """
        Check if a word matches this token.

        Returns:
            Tuple of (matches: bool, score: int)
            Higher score = better match quality.
            Score breakdown:
            - 100: Exact match to phrase
            - 90: Exact match to a simple synonym
            - 50-80: Regex match (higher = more specific)
            - 0: No match
        """
        word_lower = word.lower().strip()
        word_norm = word_lower.replace('_', ' ')

        # Exact match to phrase is best
        if word_lower == self.phrase.lower() or word_norm == self.phrase.lower():
            return (True, 100)

        # Try regex match
        if self.pattern:
            try:
                match = re.fullmatch(self.pattern, word_lower, re.IGNORECASE) or \
                       re.fullmatch(self.pattern, word_norm, re.IGNORECASE)
                if match:
                    # Score based on how specific the match is
                    # Shorter patterns = more specific = higher score
                    specificity = max(50, 90 - len(self.pattern) // 5)
                    return (True, specificity)
            except re.error:
                pass

        return (False, 0)

    def matches_bool(self, word: str) -> bool:
        """Simple boolean match check."""
        return self.matches(word)[0]


@dataclass
class Command:
    """Represents a command definition with its script body."""
    tokens: str  # Space-separated token names, e.g., "BEACON ON"
    script: str  # The script body to execute
    profile: str  # Which profile this command belongs to

    @property
    def token_list(self) -> List[str]:
        """Get tokens as a list."""
        return self.tokens.split()


@dataclass
class Profile:
    """Represents an aircraft profile with its commands."""
    name: str
    authors: str = ""
    description: str = ""
    tokens: Dict[str, Token] = field(default_factory=dict)
    commands: List[Command] = field(default_factory=list)


class CommandsLoader:
    """
    Loads and manages XPRemote commands.xml data.

    Supports:
    - Dynamic loading from configured path
    - File change detection and reloading
    - Profile-based command resolution (Zibo overrides X-Plane defaults)
    - Token synonym lookup
    """

    def __init__(self, config: Optional[XPlaneConfig] = None):
        """
        Initialize the commands loader.

        Args:
            config: Optional XPlaneConfig instance. If not provided,
                    creates a new one from default config file.
        """
        self.config = config or XPlaneConfig()
        self._xml_path: Optional[Path] = None
        self._last_modified: float = 0
        self._last_loaded: Optional[datetime] = None

        # Parsed data
        self._global_tokens: Dict[str, Token] = {}
        self._profiles: Dict[str, Profile] = {}
        self._all_commands: List[Command] = []

        # Merged view for active profile
        self._merged_tokens: Dict[str, Token] = {}
        self._merged_commands: Dict[str, Command] = {}  # keyed by token string

        # Initial load
        self._load()

    def _load(self) -> bool:
        """
        Load commands.xml from configured path.

        Returns:
            True if loaded successfully, False otherwise.
        """
        xml_path = self.config.commands_xml_path
        if not xml_path:
            return False

        self._xml_path = Path(xml_path)
        if not self._xml_path.exists():
            return False

        return self._parse_xml()

    def _parse_xml(self) -> bool:
        """
        Parse the commands.xml file.

        Returns:
            True if parsed successfully.
        """
        if not self._xml_path or not self._xml_path.exists():
            return False

        try:
            tree = ET.parse(self._xml_path)
            root = tree.getroot()

            # Reset state
            self._global_tokens = {}
            self._profiles = {}
            self._all_commands = []

            current_profile: Optional[Profile] = None
            current_profile_name = "X-Plane"  # Default profile before first <profile>

            # Create default X-Plane profile
            self._profiles["X-Plane"] = Profile(name="X-Plane")
            current_profile = self._profiles["X-Plane"]

            for elem in root:
                if elem.tag == "token":
                    token = self._parse_token(elem)
                    if token:
                        if current_profile:
                            current_profile.tokens[token.name] = token
                        self._global_tokens[token.name] = token

                elif elem.tag == "command":
                    cmd = self._parse_command(elem, current_profile_name)
                    if cmd:
                        if current_profile:
                            current_profile.commands.append(cmd)
                        self._all_commands.append(cmd)

                elif elem.tag == "profile":
                    # Start new profile
                    profile_name = elem.get("name", "Unknown")
                    authors = elem.get("authors", "")
                    desc = elem.get("descriptions", "")

                    current_profile = Profile(
                        name=profile_name,
                        authors=authors,
                        description=desc
                    )
                    self._profiles[profile_name] = current_profile
                    current_profile_name = profile_name

                    # Parse children of profile
                    for child in elem:
                        if child.tag == "token":
                            token = self._parse_token(child)
                            if token:
                                current_profile.tokens[token.name] = token
                                self._global_tokens[token.name] = token

                        elif child.tag == "command":
                            cmd = self._parse_command(child, profile_name)
                            if cmd:
                                current_profile.commands.append(cmd)
                                self._all_commands.append(cmd)

            self._last_modified = self._xml_path.stat().st_mtime
            self._last_loaded = datetime.now()

            # Build merged view for active profile
            self._build_merged_view()

            return True

        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return False
        except Exception as e:
            print(f"Error loading commands.xml: {e}")
            return False

    def _parse_token(self, elem: ET.Element) -> Optional[Token]:
        """Parse a <token> element."""
        name = elem.get("name")
        phrase = elem.get("phrase", "")
        text = (elem.text or "").strip()

        if not name:
            return None

        # The text content is the regex pattern for matching
        # It might be simple synonyms like "on|icon" or complex regex
        pattern = text if text else phrase.lower() if phrase else name.lower()

        return Token(
            name=name,
            phrase=phrase,
            pattern=pattern
        )

    def _parse_command(self, elem: ET.Element, profile_name: str) -> Optional[Command]:
        """Parse a <command> element."""
        tokens = elem.get("tokens")
        if not tokens:
            return None

        # Get script body (everything inside the command tag)
        script = elem.text or ""

        return Command(
            tokens=tokens,
            script=script.strip(),
            profile=profile_name
        )

    def _build_merged_view(self) -> None:
        """
        Build merged tokens and commands for the active profile.

        Priority: Active Profile > Fallback Profile (X-Plane)
        """
        active_profile = self.config.active_profile
        fallback_profile = self.config.fallback_profile

        # Start with global/fallback tokens
        self._merged_tokens = dict(self._global_tokens)

        # Start with fallback profile commands
        self._merged_commands = {}
        if fallback_profile in self._profiles:
            for cmd in self._profiles[fallback_profile].commands:
                self._merged_commands[cmd.tokens] = cmd

        # Override with active profile
        if active_profile in self._profiles and active_profile != fallback_profile:
            profile = self._profiles[active_profile]

            # Override tokens
            self._merged_tokens.update(profile.tokens)

            # Override commands
            for cmd in profile.commands:
                self._merged_commands[cmd.tokens] = cmd

    def reload(self) -> bool:
        """
        Force reload of commands.xml.

        Returns:
            True if reloaded successfully.
        """
        return self._load()

    def check_for_changes(self) -> bool:
        """
        Check if commands.xml has changed and reload if needed.

        Returns:
            True if file was reloaded.
        """
        if not self._xml_path or not self._xml_path.exists():
            return False

        mtime = self._xml_path.stat().st_mtime
        if mtime > self._last_modified:
            return self._load()

        return False

    def set_active_profile(self, profile_name: str) -> bool:
        """
        Set the active aircraft profile.

        Args:
            profile_name: Name of profile (e.g., "Zibo 737")

        Returns:
            True if profile exists and was set.
        """
        if profile_name in self._profiles:
            self.config.active_profile = profile_name
            self._build_merged_view()
            return True
        return False

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    @property
    def tokens(self) -> Dict[str, Token]:
        """Get merged tokens for active profile."""
        return self._merged_tokens

    @property
    def commands(self) -> Dict[str, Command]:
        """Get merged commands for active profile (keyed by token string)."""
        return self._merged_commands

    @property
    def profile_names(self) -> List[str]:
        """Get list of available profile names."""
        return list(self._profiles.keys())

    @property
    def active_profile_name(self) -> str:
        """Get current active profile name."""
        return self.config.active_profile

    def get_profile(self, name: str) -> Optional[Profile]:
        """Get a specific profile by name."""
        return self._profiles.get(name)

    def get_token(self, name: str) -> Optional[Token]:
        """Get a token by name from merged view."""
        return self._merged_tokens.get(name)

    def get_command(self, token_string: str) -> Optional[Command]:
        """Get a command by its token string (e.g., 'BEACON ON')."""
        return self._merged_commands.get(token_string)

    def find_commands_by_token(self, token_name: str) -> List[Command]:
        """Find all commands that use a specific token."""
        results = []
        for cmd in self._merged_commands.values():
            if token_name in cmd.token_list:
                results.append(cmd)
        return results

    def search_commands(self, query: str) -> List[Command]:
        """
        Search commands by partial token match.

        Args:
            query: Search string (matched against token names)

        Returns:
            List of matching commands.
        """
        query_upper = query.upper()
        results = []
        for cmd in self._merged_commands.values():
            if query_upper in cmd.tokens:
                results.append(cmd)
        return results

    # Special operand placeholder tokens that aren't in the token list
    OPERAND_PLACEHOLDERS = {
        'NUMBER', 'DEGREES', 'IDENTIFIER', 'NAV_FREQUENCY', 'COM_FREQUENCY',
        'ADF_FREQUENCY', 'ALTIMETER_SETTING', 'FLIGHT_LEVEL', 'STRING'
    }

    # Digit tokens that should be treated as number operands
    DIGIT_TOKENS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    def match_input_to_tokens(self, input_text: str) -> Tuple[List[str], dict]:
        """
        Match input text to token names and extract operands.

        Args:
            input_text: Natural language input (e.g., "flaps five")

        Returns:
            Tuple of (matched_token_names, operands_dict)
            operands_dict may contain: 'number', 'identifier', etc.
        """
        # Replace underscores with spaces so "FUEL_PUMP ON" matches "FUEL PUMP ON"
        words = input_text.lower().strip().replace('_', ' ').split()
        matched_tokens = []
        operands = {}
        i = 0

        while i < len(words):
            word = words[i]

            # Check for floating point number (e.g., "121.5" or "250")
            float_num = self._parse_float(word)
            if float_num is not None:
                operands['number'] = float_num
                matched_tokens.append('NUMBER')
                i += 1
                continue

            # Check for spoken number phrase (e.g., "one hundred thirty", "fifteen thousand")
            spoken_num, words_consumed = self._parse_spoken_number(words, i)
            if spoken_num is not None and words_consumed > 1:
                # Only use spoken number if it consumed multiple words
                operands['number'] = spoken_num
                matched_tokens.append('NUMBER')
                i += words_consumed
                continue

            # Check for decimal point word ("point", "decimal")
            if word in ('point', 'decimal', 'dot') and 'number_parts' in operands:
                operands['has_decimal'] = True
                operands['decimal_parts'] = []
                i += 1
                continue

            # Check if it's a number word/digit (before matching tokens)
            num = self._parse_number_word(word)
            if num is not None:
                if operands.get('has_decimal'):
                    # This is a decimal digit
                    operands['decimal_parts'].append(num)
                else:
                    if 'number_parts' not in operands:
                        operands['number_parts'] = []
                    operands['number_parts'].append(num)
                matched_tokens.append('NUMBER')
                i += 1
                continue

            best_match = None
            best_score = 0
            words_consumed = 1

            # Try matching multiple words first (for multi-word tokens like "landing light")
            # Start with longest phrases - prefer multi-word matches over single word
            for length in range(min(3, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i + length])

                for name, token in self._merged_tokens.items():
                    # Skip digit tokens - we handle numbers specially above
                    if name in self.DIGIT_TOKENS:
                        continue

                    matches, score = token.matches(phrase)
                    if matches:
                        # Add bonus for longer matches (more specific)
                        adjusted_score = score + (length * 20)
                        if adjusted_score > best_score:
                            best_match = name
                            best_score = adjusted_score
                            words_consumed = length

            if best_match:
                matched_tokens.append(best_match)
                i += words_consumed
            else:
                # Unknown word - skip it
                i += 1

        # Combine consecutive NUMBER tokens into a single number
        # e.g., ['FLAPS', 'NUMBER', 'NUMBER', 'NUMBER'] with parts [1,2,0] -> ['FLAPS', 'NUMBER'] with number=120
        # Also handles decimals: "one two one point five" -> 121.5
        if 'number_parts' in operands and 'number' not in operands:
            integer_part = self._combine_number_parts(operands['number_parts'])
            if operands.get('has_decimal') and operands.get('decimal_parts'):
                # Combine decimal digits: [5] -> 0.5, [2, 5] -> 0.25
                decimal_str = ''.join(str(d) for d in operands['decimal_parts'])
                decimal_part = float('0.' + decimal_str)
                operands['number'] = float(integer_part) + decimal_part
            else:
                operands['number'] = integer_part
            del operands['number_parts']
            operands.pop('has_decimal', None)
            operands.pop('decimal_parts', None)

        # Collapse consecutive NUMBER tokens into one
        if matched_tokens:
            collapsed = []
            for t in matched_tokens:
                if t == 'NUMBER':
                    if not collapsed or collapsed[-1] != 'NUMBER':
                        collapsed.append('NUMBER')
                else:
                    collapsed.append(t)
            matched_tokens = collapsed

        return matched_tokens, operands

    def _parse_float(self, word: str) -> Optional[float]:
        """Parse a word as a floating point number (e.g., '121.5')."""
        word = word.strip()
        try:
            # Check if it looks like a number with decimal
            if '.' in word or word.replace('-', '').isdigit():
                return float(word)
        except ValueError:
            pass
        return None

    def _parse_number_word(self, word: str) -> Optional[int]:
        """Parse a word as a single digit number (digit or word form)."""
        word = word.lower().strip()

        # Single digit
        if len(word) == 1 and word.isdigit():
            return int(word)

        # Multi-digit number - return None, let _parse_float handle it
        if word.isdigit():
            return None

        # Word forms for single digits
        number_words = {
            'zero': 0, 'oh': 0,
            'one': 1,
            'two': 2, 'too': 2, 'to': 2,
            'three': 3, 'tree': 3,
            'four': 4, 'for': 4,
            'five': 5, 'fife': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9, 'niner': 9,
        }
        return number_words.get(word)

    def _parse_spoken_number(self, words: List[str], start_idx: int) -> Tuple[Optional[int], int]:
        """
        Parse spoken English numbers. Two modes:
        1. Multiplier mode: "one hundred thirty" → 130, "fifteen thousand" → 15000
        2. Concatenation mode: "two fifty" → 250, "one eight zero" → 180

        If "hundred" or "thousand" is found, use multiplier mode (standard spoken numbers).
        Otherwise, concatenate numeric values as strings (aviation style).

        Returns (number, words_consumed) or (None, 0) if not a number.
        """
        ones = {
            'zero': 0, 'oh': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19
        }
        tens = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        multipliers = {'hundred': 100, 'thousand': 1000}

        # First pass: collect all numeric parts and check for multipliers
        parts = []  # List of (value, is_multiplier)
        has_multiplier = False
        i = start_idx

        while i < len(words):
            word = words[i].lower().strip()

            # Skip "and" in numbers like "one hundred and thirty"
            if word == 'and':
                i += 1
                continue

            if word in ones:
                parts.append((ones[word], False))
                i += 1
            elif word in tens:
                parts.append((tens[word], False))
                i += 1
            elif word in multipliers:
                parts.append((multipliers[word], True))
                has_multiplier = True
                i += 1
            else:
                break

        consumed = i - start_idx

        if not parts:
            return None, 0

        if has_multiplier:
            # Use standard multiplier-based parsing
            # "one hundred thirty" → 1 * 100 + 30 = 130
            # "fifteen thousand" → 15 * 1000 = 15000
            result = 0
            current = 0
            for value, is_mult in parts:
                if is_mult:
                    if current == 0:
                        current = 1
                    if value == 1000:
                        result += current * 1000
                        current = 0
                    else:  # hundred
                        current *= 100
                else:
                    current += value
            result += current
        else:
            # Concatenation mode: "two fifty" → "2" + "50" = "250"
            # "one eight zero" → "1" + "8" + "0" = "180"
            result_str = ''.join(str(v) for v, _ in parts)
            result = int(result_str) if result_str else 0

        if consumed > 0 and result >= 0:
            return result, consumed
        return None, 0

    def _combine_number_parts(self, parts: List[int]) -> int:
        """Combine individual digits into a number (e.g., [1, 2, 0] -> 120)."""
        if not parts:
            return 0
        result = 0
        for digit in parts:
            result = result * 10 + digit
        return result

    def find_command_for_input(self, input_text: str) -> Tuple[Optional[Command], List[str], dict]:
        """
        Find the best matching command for natural language input.

        Args:
            input_text: Natural language input (e.g., "beacon on")

        Returns:
            Tuple of (best_command, matched_tokens, operands)
        """
        matched_tokens, operands = self.match_input_to_tokens(input_text)
        if not matched_tokens:
            return None, [], {}

        # Filter out NUMBER placeholders for command matching
        # (they're operands, not part of the command token string)
        filter_tokens = [t for t in matched_tokens if t not in self.OPERAND_PLACEHOLDERS]
        token_string = " ".join(filter_tokens)

        # If there's a number operand, try placeholder commands FIRST
        # This ensures "heading 270" matches "HEADING DEGREES" not just "HEADING"
        if 'number' in operands:
            # Try each placeholder type that could represent a number
            for placeholder in ['DEGREES', 'NUMBER', 'NAV_FREQUENCY', 'COM_FREQUENCY',
                                'ADF_FREQUENCY', 'ALTIMETER_SETTING', 'FLIGHT_LEVEL']:
                # Try appending placeholder
                token_with_placeholder = token_string + " " + placeholder
                if token_with_placeholder in self._merged_commands:
                    return self._merged_commands[token_with_placeholder], matched_tokens, operands

            # Also try replacing last token with placeholder (if NUMBER was matched separately)
            if filter_tokens:
                for placeholder in ['NUMBER', 'DEGREES']:
                    token_with_placeholder = " ".join(filter_tokens[:-1] + [placeholder])
                    if token_with_placeholder in self._merged_commands:
                        return self._merged_commands[token_with_placeholder], matched_tokens, operands

        # Try exact match (no placeholders)
        if token_string in self._merged_commands:
            return self._merged_commands[token_string], matched_tokens, operands

        # Try partial matches (find commands that contain all matched tokens)
        candidates = []
        for cmd_tokens, cmd in self._merged_commands.items():
            cmd_token_list = cmd.token_list
            cmd_token_set = set(cmd_token_list)
            filter_set = set(filter_tokens)

            # Check if all filtered tokens are in the command (ignoring operand placeholders)
            cmd_without_operands = {t for t in cmd_token_set if t not in self.OPERAND_PLACEHOLDERS}

            if filter_set.issubset(cmd_without_operands):
                # Score by how many extra tokens the command has
                extra_tokens = len(cmd_without_operands) - len(filter_set)
                candidates.append((extra_tokens, cmd))

        if candidates:
            # Return command with fewest extra tokens
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1], matched_tokens, operands

        return None, matched_tokens, operands

    def get_stats(self) -> dict:
        """Get statistics about loaded data."""
        return {
            "xml_path": str(self._xml_path) if self._xml_path else None,
            "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
            "profiles_count": len(self._profiles),
            "profile_names": list(self._profiles.keys()),
            "active_profile": self.config.active_profile,
            "total_tokens": len(self._global_tokens),
            "merged_tokens": len(self._merged_tokens),
            "total_commands": len(self._all_commands),
            "merged_commands": len(self._merged_commands),
        }

    def to_dict(self) -> dict:
        """
        Export all data as dictionary for serialization.

        Returns:
            dict with tokens, commands, and profiles.
        """
        return {
            "stats": self.get_stats(),
            "tokens": {
                name: {
                    "name": t.name,
                    "phrase": t.phrase,
                    "pattern": t.pattern
                }
                for name, t in self._merged_tokens.items()
            },
            "commands": [
                {
                    "tokens": cmd.tokens,
                    "profile": cmd.profile,
                    "script": cmd.script[:200] + "..." if len(cmd.script) > 200 else cmd.script
                }
                for cmd in self._merged_commands.values()
            ],
            "profiles": {
                name: {
                    "name": p.name,
                    "authors": p.authors,
                    "description": p.description,
                    "token_count": len(p.tokens),
                    "command_count": len(p.commands)
                }
                for name, p in self._profiles.items()
            }
        }
