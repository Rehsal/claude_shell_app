"""
Claude Shell App - A development shell for managing projects with persistent state.

Includes X-Plane Copilot integration for XPRemote commands.
Now uses ScriptExecutor for full script execution.
"""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# X-Plane integration
from src.xplane import XPlaneConfig, CommandsLoader, ChecklistRunner, AnnunciatorMonitor
from src.xplane.command_categories import categorize_commands
from src.xplane.extplane_client import get_client, ExtPlaneClient
from src.xplane.script_executor import ScriptExecutor
from src.xplane.command_executor import execute_command_script
from src.xplane.fms_programmer import FMSProgrammer

load_dotenv()

# Server version — changes every restart so UI can detect stale cache
SERVER_VERSION = datetime.now().strftime("%H:%M:%S")
print(f"[server] Starting version {SERVER_VERSION}")

app = FastAPI(title="Claude Shell App", version="1.0.0")


# ---------------------------------------------------------------------------
# Background ExtPlane health monitor — detects stale connections and recovers
# ---------------------------------------------------------------------------
_health_check_task = None


async def _extplane_health_loop():
    """Periodically check ExtPlane connection health and auto-reconnect."""
    while True:
        await asyncio.sleep(5)
        try:
            client = get_client()
            if client.is_connected:
                if not client.health_check():
                    print("[health] Stale ExtPlane connection detected, reconnecting...")
                    client.reconnect()
        except Exception as e:
            print(f"[health] Error in health check: {e}")


@app.on_event("startup")
async def start_health_monitor():
    global _health_check_task
    _health_check_task = asyncio.create_task(_extplane_health_loop())


@app.on_event("shutdown")
async def stop_health_monitor():
    global _health_check_task
    if _health_check_task:
        _health_check_task.cancel()

# Initialize X-Plane commands loader (singleton)
xplane_config = XPlaneConfig()
commands_loader = CommandsLoader(xplane_config)

# Configuration
APP_DATA_DIR = Path(os.getenv("APP_DATA_DIR", "./data"))
PROJECTS_DIR = APP_DATA_DIR / "projects"
UPLOADS_DIR = APP_DATA_DIR / "uploads"

# Ensure directories exist
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_project_path(project_id: str) -> Path:
    """Get the path to a project's JSON file."""
    return PROJECTS_DIR / f"{project_id}.json"


def load_project(project_id: str) -> dict:
    """Load a project from disk."""
    path = get_project_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(project_id: str, data: dict) -> None:
    """Save a project to disk atomically."""
    path = get_project_path(project_id)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def add_log_entry(project: dict, kind: str, message: str) -> None:
    """Add an entry to the project's session log."""
    if "log" not in project:
        project["log"] = []
    project["log"].append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "kind": kind,
        "message": message
    })


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/commands", response_class=HTMLResponse)
async def copilot():
    """Serve the X-Plane Copilot command browser."""
    with open("templates/commands.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/projects")
async def list_projects():
    """List all projects."""
    projects = []
    for path in PROJECTS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                projects.append({
                    "id": data["id"],
                    "name": data["name"],
                    "description": data.get("description", ""),
                    "created_at": data["created_at"]
                })
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(projects, key=lambda x: x["created_at"], reverse=True)


@app.post("/api/projects")
async def create_project(name: str = Form(...), description: str = Form("")):
    """Create a new project."""
    project_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    project = {
        "id": project_id,
        "name": name,
        "description": description,
        "created_at": now,
        "updated_at": now,
        "notes": "",
        "pins": [],
        "uploads": [],
        "log": []
    }

    add_log_entry(project, "project_created", f"Project '{name}' created")
    save_project(project_id, project)

    # Create uploads directory for this project
    (UPLOADS_DIR / project_id).mkdir(parents=True, exist_ok=True)

    return project


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get a single project."""
    return load_project(project_id)


@app.post("/api/projects/{project_id}/notes")
async def update_notes(project_id: str, notes: str = Form(...)):
    """Update project notes."""
    project = load_project(project_id)
    project["notes"] = notes
    project["updated_at"] = datetime.utcnow().isoformat() + "Z"
    add_log_entry(project, "notes_updated", "Notes updated")
    save_project(project_id, project)
    return {"status": "ok", "notes": notes}


@app.post("/api/projects/{project_id}/pin")
async def add_pin(project_id: str, title: str = Form(...), content: str = Form(...)):
    """Add a pinned prompt to the project."""
    project = load_project(project_id)

    pin_id = str(uuid.uuid4())
    pin = {
        "id": pin_id,
        "title": title,
        "content": content,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    if "pins" not in project:
        project["pins"] = []
    project["pins"].append(pin)
    project["updated_at"] = datetime.utcnow().isoformat() + "Z"
    add_log_entry(project, "pin_added", f"Pin '{title}' added")
    save_project(project_id, project)

    return pin


@app.delete("/api/projects/{project_id}/pin/{pin_id}")
async def delete_pin(project_id: str, pin_id: str):
    """Delete a pinned prompt."""
    project = load_project(project_id)

    pins = project.get("pins", [])
    original_len = len(pins)
    project["pins"] = [p for p in pins if p["id"] != pin_id]

    if len(project["pins"]) == original_len:
        raise HTTPException(status_code=404, detail="Pin not found")

    project["updated_at"] = datetime.utcnow().isoformat() + "Z"
    add_log_entry(project, "pin_deleted", f"Pin deleted")
    save_project(project_id, project)

    return {"status": "ok"}


@app.post("/api/projects/{project_id}/upload")
async def upload_file(
    project_id: str,
    file: UploadFile = File(...),
    label: str = Form("")
):
    """Upload a file to the project."""
    project = load_project(project_id)

    # Create upload directory for project if needed
    upload_dir = UPLOADS_DIR / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    file_id = str(uuid.uuid4())
    original_name = file.filename or "unnamed"
    ext = Path(original_name).suffix
    stored_name = f"{file_id}{ext}"
    file_path = upload_dir / stored_name

    # Save file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Record metadata
    upload_meta = {
        "id": file_id,
        "original_name": original_name,
        "stored_name": stored_name,
        "label": label or original_name,
        "size": len(content),
        "path": str(file_path.relative_to(APP_DATA_DIR)),
        "uploaded_at": datetime.utcnow().isoformat() + "Z"
    }

    if "uploads" not in project:
        project["uploads"] = []
    project["uploads"].append(upload_meta)
    project["updated_at"] = datetime.utcnow().isoformat() + "Z"
    add_log_entry(project, "file_uploaded", f"File '{original_name}' uploaded")
    save_project(project_id, project)

    return upload_meta


@app.get("/api/projects/{project_id}/export")
async def export_project(project_id: str):
    """Export project as JSON with upload paths."""
    project = load_project(project_id)

    # Add full upload paths
    export_data = project.copy()
    export_data["upload_paths"] = [
        str(UPLOADS_DIR / project_id / u["stored_name"])
        for u in project.get("uploads", [])
    ]

    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="{project["name"]}_export.json"'
        }
    )


@app.post("/api/projects/{project_id}/log")
async def add_log(project_id: str, kind: str = Form(...), message: str = Form(...)):
    """Add a custom log entry."""
    project = load_project(project_id)
    add_log_entry(project, kind, message)
    project["updated_at"] = datetime.utcnow().isoformat() + "Z"
    save_project(project_id, project)
    return {"status": "ok"}


# =============================================================================
# X-Plane Copilot API Endpoints
# =============================================================================

@app.get("/api/xplane/status")
async def xplane_status():
    """Get X-Plane configuration and commands loader status."""
    validation = xplane_config.validate()
    stats = commands_loader.get_stats()
    return {
        "config_valid": validation["valid"],
        "config_errors": validation["errors"],
        "stats": stats,
        "extplane": {
            "host": xplane_config.extplane_host,
            "port": xplane_config.extplane_port
        }
    }


@app.post("/api/xplane/reload")
async def xplane_reload():
    """Reload commands.xml from disk."""
    success = commands_loader.reload()
    if success:
        return {
            "status": "ok",
            "message": "Commands reloaded successfully",
            "stats": commands_loader.get_stats()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload commands.xml")


@app.get("/api/xplane/profiles")
async def xplane_profiles():
    """List available aircraft profiles."""
    profiles = []
    for name in commands_loader.profile_names:
        profile = commands_loader.get_profile(name)
        if profile:
            profiles.append({
                "name": profile.name,
                "authors": profile.authors,
                "description": profile.description,
                "token_count": len(profile.tokens),
                "command_count": len(profile.commands),
                "is_active": name == commands_loader.active_profile_name
            })
    return profiles


@app.post("/api/xplane/profiles/{profile_name}/activate")
async def xplane_activate_profile(profile_name: str):
    """Set the active aircraft profile."""
    if commands_loader.set_active_profile(profile_name):
        return {
            "status": "ok",
            "active_profile": profile_name
        }
    else:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found")


@app.get("/api/xplane/tokens")
async def xplane_tokens(limit: int = 100, offset: int = 0):
    """List tokens with their patterns."""
    tokens = list(commands_loader.tokens.values())
    total = len(tokens)
    tokens = tokens[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "tokens": [
            {
                "name": t.name,
                "phrase": t.phrase,
                "pattern": t.pattern
            }
            for t in tokens
        ]
    }


@app.get("/api/xplane/tokens/{token_name}")
async def xplane_get_token(token_name: str):
    """Get a specific token by name."""
    token = commands_loader.get_token(token_name)
    if not token:
        raise HTTPException(status_code=404, detail=f"Token '{token_name}' not found")

    # Also find commands that use this token
    related_commands = commands_loader.find_commands_by_token(token_name)

    return {
        "name": token.name,
        "phrase": token.phrase,
        "pattern": token.pattern,
        "used_in_commands": [cmd.tokens for cmd in related_commands[:20]]
    }


@app.get("/api/xplane/commands")
async def xplane_commands(limit: int = 100, offset: int = 0, q: str = ""):
    """List commands with optional search."""
    if q:
        cmds = commands_loader.search_commands(q)
    else:
        cmds = list(commands_loader.commands.values())

    total = len(cmds)
    cmds = cmds[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "query": q,
        "commands": [
            {
                "tokens": cmd.tokens,
                "profile": cmd.profile,
                "script_preview": cmd.script[:200] + "..." if len(cmd.script) > 200 else cmd.script
            }
            for cmd in cmds
        ]
    }


@app.get("/api/xplane/commands/{token_string:path}")
async def xplane_get_command(token_string: str):
    """Get a specific command by its token string (e.g., 'BEACON ON')."""
    # URL-decode and normalize
    token_string = token_string.replace("/", " ").strip().upper()

    cmd = commands_loader.get_command(token_string)
    if not cmd:
        raise HTTPException(status_code=404, detail=f"Command '{token_string}' not found")

    return {
        "tokens": cmd.tokens,
        "token_list": cmd.token_list,
        "profile": cmd.profile,
        "script": cmd.script
    }


@app.post("/api/xplane/match")
async def xplane_match_input(text: str = Form(...)):
    """
    Match natural language input to tokens and find the best command.

    This is the main entry point for voice/text command processing.
    """
    # Check for file changes
    commands_loader.check_for_changes()

    # Find best command (includes token matching)
    best_cmd, matched_tokens, operands = commands_loader.find_command_for_input(text)

    result = {
        "input": text,
        "matched_tokens": matched_tokens,
        "operands": operands,
        "command_found": best_cmd is not None
    }

    if best_cmd:
        result["command"] = {
            "tokens": best_cmd.tokens,
            "profile": best_cmd.profile,
            "script": best_cmd.script
        }

    return result


@app.get("/api/xplane/export")
async def xplane_export():
    """Export all commands data as JSON."""
    return commands_loader.to_dict()


@app.get("/api/xplane/categories")
async def xplane_categories():
    """Get commands organized by category."""
    categories = categorize_commands(commands_loader)

    # Convert to JSON-serializable format
    result = {}
    for cat_name, cat_commands in categories.items():
        result[cat_name] = [
            {
                "command": {
                    "tokens": cmd.command.tokens,
                    "profile": cmd.command.profile,
                    "script_preview": cmd.command.script[:200] if cmd.command.script else ""
                },
                "category": cmd.category,
                "requires_value": cmd.requires_value,
                "value_type": cmd.value_type
            }
            for cmd in cat_commands
        ]

    return result


@app.post("/api/commands/reload")
async def reload_commands():
    """Reload commands.xml from disk."""
    success = commands_loader.reload()
    if success:
        return {
            "status": "reloaded",
            "total_commands": len(commands_loader.commands),
            "profiles": commands_loader.profile_names
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload commands.xml")


@app.get("/api/commands")
async def copilot_commands():
    """
    Get all commands organized by profile and category for the Copilot page.
    Returns commands grouped by profile, then by category, with test values.
    """
    # Sample test values for different value types
    TEST_VALUES = {
        "number": "100",
        "degrees": "180",
        "frequency": "121.5",
        "identifier": "KLAX",
        "flight_level": "350",
        "altimeter": "29.92",
    }

    # Value placeholder tokens - these get replaced with test values
    VALUE_TOKENS = {"NUMBER", "DEGREES", "FREQUENCY", "NAV_FREQUENCY", "COM_FREQUENCY",
                    "ADF_FREQUENCY", "IDENTIFIER", "FLIGHT_LEVEL", "ALTIMETER_SETTING"}

    categories = categorize_commands(commands_loader)
    profiles = {}

    for cat_name, cat_commands in categories.items():
        for cat_cmd in cat_commands:
            cmd = cat_cmd.command
            profile_name = cmd.profile

            if profile_name not in profiles:
                profiles[profile_name] = {}

            if cat_name not in profiles[profile_name]:
                profiles[profile_name][cat_name] = []

            # Generate test phrase by looking up each token's phrase
            test_phrases = []
            token_names = cmd.tokens.split()
            for token_name in token_names:
                if token_name in VALUE_TOKENS:
                    # Skip value placeholders - they'll be added separately
                    continue
                token = commands_loader.get_token(token_name)
                if token and token.phrase:
                    test_phrases.append(token.phrase.lower())
                else:
                    # Fallback: convert underscore to space
                    test_phrases.append(token_name.replace('_', ' ').lower())

            test_phrase = ' '.join(test_phrases)

            test_value = ""
            if cat_cmd.requires_value and cat_cmd.value_type:
                test_value = TEST_VALUES.get(cat_cmd.value_type, "100")

            profiles[profile_name][cat_name].append({
                "tokens": cmd.tokens,
                "test_phrase": test_phrase,
                "profile": profile_name,
                "category": cat_name,
                "requires_value": cat_cmd.requires_value,
                "value_type": cat_cmd.value_type,
                "test_value": test_value,
                "script_preview": cmd.script[:100] if cmd.script else ""
            })

    # Sort categories within each profile
    for profile_name in profiles:
        profiles[profile_name] = dict(sorted(profiles[profile_name].items()))

    # Always return all available profiles from the loader (not just from commands found)
    all_profiles = commands_loader.profile_names

    return {
        "profiles": all_profiles,
        "active_profile": commands_loader.active_profile_name,
        "commands_by_profile": profiles,
        "total_commands": len(commands_loader.commands)
    }


# =============================================================================
# ExtPlane Connection & Execution API
# =============================================================================

@app.get("/api/extplane/status")
async def extplane_status():
    """Get ExtPlane connection status."""
    client = get_client()
    return {
        "connected": client.is_connected,
        "host": client.host,
        "port": client.port
    }


@app.post("/api/extplane/connect")
async def extplane_connect():
    """Connect to ExtPlane."""
    client = get_client()
    if client.is_connected:
        return {"status": "already_connected"}

    success = client.connect()
    if success:
        return {"status": "connected"}
    else:
        raise HTTPException(status_code=503, detail="Failed to connect to ExtPlane")


@app.post("/api/extplane/disconnect")
async def extplane_disconnect():
    """Disconnect from ExtPlane."""
    client = get_client()
    client.disconnect()
    return {"status": "disconnected"}


@app.post("/api/extplane/command")
async def extplane_send_command(command: str = Form(...)):
    """Send a raw command to X-Plane."""
    client = get_client()
    if not client.is_connected:
        raise HTTPException(status_code=503, detail="Not connected to ExtPlane")

    success = client.send_command(command)
    return {"status": "sent" if success else "failed", "command": command}


@app.post("/api/extplane/dataref/set")
async def extplane_set_dataref(dataref: str = Form(...), value: str = Form(...)):
    """Set a dataref value."""
    client = get_client()
    if not client.is_connected:
        raise HTTPException(status_code=503, detail="Not connected to ExtPlane")

    # Parse value
    try:
        if '.' in value:
            parsed_value = float(value)
        else:
            parsed_value = int(value)
    except:
        parsed_value = value

    success = client.set_dataref(dataref, parsed_value)
    return {"status": "sent" if success else "failed", "dataref": dataref, "value": parsed_value}


@app.get("/api/extplane/dataref/get")
async def extplane_get_dataref(dataref: str):
    """Get a dataref value (subscribes temporarily if needed)."""
    client = get_client()
    if not client.is_connected:
        raise HTTPException(status_code=503, detail="Not connected to ExtPlane")

    value = client.get_dataref(dataref, timeout=2.0)
    return {"dataref": dataref, "value": value}


@app.post("/api/extplane/dataref/subscribe")
async def extplane_subscribe(dataref: str = Form(...)):
    """Subscribe to a dataref for continuous updates."""
    client = get_client()
    if not client.is_connected:
        raise HTTPException(status_code=503, detail="Not connected to ExtPlane")

    success = client.subscribe(dataref)
    return {"status": "subscribed" if success else "failed", "dataref": dataref}


@app.post("/api/extplane/dataref/unsubscribe")
async def extplane_unsubscribe(dataref: str = Form(...)):
    """Unsubscribe from a dataref."""
    client = get_client()
    if not client.is_connected:
        raise HTTPException(status_code=503, detail="Not connected to ExtPlane")

    success = client.unsubscribe(dataref)
    return {"status": "unsubscribed" if success else "failed", "dataref": dataref}


@app.post("/api/extplane/execute")
async def extplane_execute_command(text: str = Form(...)):
    """
    Execute a natural language command by running the script from commands.xml.

    This:
    1. Matches input text to a command's tokens
    2. Gets the script code from commands.xml
    3. Executes the full script using ScriptExecutor (variables, conditionals, loops, etc.)
    4. Verifies execution by monitoring dataref changes
    """
    import re
    import time

    client = get_client()

    # Check connection
    if not client.is_connected:
        if not client.connect():
            return {
                "status": "error",
                "detail": "Not connected to ExtPlane. Is X-Plane running with ExtPlane plugin?",
                "input": text,
                "matched_tokens": [],
                "connected": False
            }

    # Find the command
    best_cmd, matched_tokens, operands = commands_loader.find_command_for_input(text)

    if not best_cmd:
        return {
            "status": "no_match",
            "input": text,
            "matched_tokens": matched_tokens,
            "connected": client.is_connected
        }

    script = best_cmd.script

    # Extract datarefs from script for verification (before execution)
    datarefs_to_watch = list(set(
        m.group(1)
        for pat in (
            r'setDataRefValue\s*\(\s*["\']([^"\']+)["\']',
            r'setDataRefArrayValue\s*\(\s*["\']([^"\']+)["\']',
            r'getDataRefValue\s*\(\s*["\']([^"\']+)["\']',
        )
        for m in re.finditer(pat, script)
    ))[:10]

    # Capture initial values (execute_command_script will subscribe for us,
    # but we need pre-values for change detection, so subscribe early)
    initial_values = {}
    for dr in datarefs_to_watch:
        try:
            client.subscribe(dr)
        except Exception as e:
            print(f"Subscribe error for {dr}: {e}")
    time.sleep(0.2)
    for dr in datarefs_to_watch:
        val = client.get_subscribed_value(dr)
        initial_values[dr] = val.value if val else None

    # Execute via shared function (handles subscribe/execute/unsubscribe)
    result = execute_command_script(client, script, operands)

    # Check final values for verification
    # Re-subscribe briefly since execute_command_script unsubscribed
    for dr in datarefs_to_watch:
        try:
            client.subscribe(dr)
        except Exception:
            pass
    time.sleep(0.1)

    final_values = {}
    changes = []
    for dr in datarefs_to_watch:
        try:
            val = client.get_subscribed_value(dr)
            final_values[dr] = val.value if val else None

            if initial_values.get(dr) != final_values.get(dr):
                changes.append({
                    "dataref": dr,
                    "before": initial_values.get(dr),
                    "after": final_values.get(dr)
                })
        except Exception as e:
            print(f"Error getting final value for {dr}: {e}")
            final_values[dr] = None

    # Also add any datarefs that were set by the executor to the changes list
    for dr_set in result.get("datarefs_set", []):
        dr = dr_set.get("dataref", "")
        if dr and dr not in [c["dataref"] for c in changes]:
            try:
                client.subscribe(dr)
                time.sleep(0.1)
                val = client.get_subscribed_value(dr)
                if val:
                    changes.append({
                        "dataref": dr,
                        "before": None,
                        "after": val.value
                    })
            except:
                pass

    # Unsubscribe from all
    for dr in datarefs_to_watch:
        try:
            client.unsubscribe(dr)
        except:
            pass

    # Determine status
    has_errors = len(result.get("errors", [])) > 0
    did_something = len(result.get("commands_sent", [])) > 0 or len(result.get("datarefs_set", [])) > 0
    has_changes = len(changes) > 0

    if has_errors and not did_something:
        status = "failed"
    elif has_changes:
        status = "verified"
    elif did_something:
        status = "sent"
    else:
        status = "failed"

    return {
        "status": status,
        "input": text,
        "matched_tokens": matched_tokens,
        "operands": operands,
        "command": {
            "tokens": best_cmd.tokens,
            "profile": best_cmd.profile
        },
        "commands_sent": result.get("commands_sent", []),
        "datarefs_set": result.get("datarefs_set", []),
        "datarefs_watched": datarefs_to_watch[:10],
        "dataref_changes": changes,
        "initial_values": initial_values,
        "final_values": final_values,
        "errors": result.get("errors", []),
        "verified": status == "verified",
        "connected": client.is_connected
    }


# =============================================================================
# AI Copilot - Claude-powered natural language command interpretation
# =============================================================================

# Initialize Anthropic client (lazy loading)
_anthropic_client = None

def get_anthropic_client():
    """Get or create Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            _anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            return None
    return _anthropic_client


@app.get("/api/ai/status")
async def ai_status():
    """Check if AI Copilot is available."""
    client = get_anthropic_client()
    return {
        "available": client is not None,
        "reason": "API key not configured" if client is None else None
    }


@app.post("/api/ai/interpret")
async def ai_interpret_command(text: str = Form(...)):
    """
    Use Claude to interpret natural language and find the best matching command.

    Takes natural speech like "turn on the landing lights" and returns
    the best matching command phrase like "landing lights on".
    """
    client = get_anthropic_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="AI Copilot not available. Set ANTHROPIC_API_KEY in .env file."
        )

    # Build list of available commands with their phrases
    categories = categorize_commands(commands_loader)
    command_list = []

    for cat_name, cat_commands in categories.items():
        for cat_cmd in cat_commands:
            cmd = cat_cmd.command
            # Get the spoken phrase for this command
            tokens = cmd.tokens.split()
            phrases = []
            for token_name in tokens:
                # Skip value placeholders
                if token_name in {'NUMBER', 'DEGREES', 'FREQUENCY', 'NAV_FREQUENCY',
                                  'COM_FREQUENCY', 'ADF_FREQUENCY', 'IDENTIFIER',
                                  'FLIGHT_LEVEL', 'ALTIMETER_SETTING'}:
                    phrases.append(f"[{token_name}]")
                    continue
                token = commands_loader.get_token(token_name)
                if token and token.phrase:
                    phrases.append(token.phrase.lower())
                else:
                    phrases.append(token_name.replace('_', ' ').lower())

            phrase = ' '.join(phrases)
            command_list.append({
                "tokens": cmd.tokens,
                "phrase": phrase,
                "category": cat_name,
                "requires_value": cat_cmd.requires_value,
                "value_type": cat_cmd.value_type
            })

    # Build the prompt for Claude
    commands_text = "\n".join([
        f"- {c['phrase']}" + (f" (requires {c['value_type']})" if c['requires_value'] else "")
        for c in command_list
    ])

    system_prompt = """You are an aircraft copilot assistant. Your job is to interpret pilot commands and map them to the available aircraft control commands.

Given a pilot's spoken instruction, determine the best matching command from the available commands list.

Rules:
1. Return ONLY the exact command phrase(s) that should be executed, nothing else
2. If the command requires a value (like a number), include it in your response
3. If the pilot's intent is unclear or doesn't match any command, respond with "NO_MATCH"
4. Be flexible with phrasing - "turn on beacon" matches "beacon on", "activate landing lights" matches "landing lights on"
5. For numerical values, extract the number from the speech (e.g., "set flaps to 5" -> "flaps 5")
6. You can return multiple commands separated by semicolons if the pilot clearly wants multiple actions
7. IGNORE timing instructions like "wait", "pause", "after X seconds" - these cannot be executed. Only return the actual aircraft commands.
8. For "turn off all lights" or similar, return each individual light command separated by semicolons

Available commands:
""" + commands_text

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": f"Pilot says: \"{text}\""}
            ]
        )

        interpreted = message.content[0].text.strip()

        if interpreted == "NO_MATCH":
            return {
                "success": False,
                "input": text,
                "interpreted": None,
                "reason": "Could not match to any available command"
            }

        # Handle multiple commands (separated by semicolons)
        commands = [cmd.strip() for cmd in interpreted.split(';') if cmd.strip()]

        return {
            "success": True,
            "input": text,
            "interpreted": commands[0] if len(commands) == 1 else commands,
            "multiple": len(commands) > 1
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI interpretation failed: {str(e)}")


@app.post("/api/ai/execute")
async def ai_execute_command(text: str = Form(...)):
    """
    Interpret natural language with Claude and execute the matched command.

    Combines AI interpretation with command execution in one call.
    """
    # First interpret with AI
    client = get_anthropic_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="AI Copilot not available. Set ANTHROPIC_API_KEY in .env file."
        )

    # Get interpretation
    import json as json_module
    interpret_result = await ai_interpret_command(text)

    if not interpret_result["success"]:
        return {
            "status": "no_match",
            "input": text,
            "interpreted": None,
            "reason": interpret_result.get("reason", "No matching command found")
        }

    # Execute the interpreted command(s)
    interpreted = interpret_result["interpreted"]

    # Handle single or multiple commands
    if isinstance(interpreted, list):
        # Multiple commands - execute each and track results
        results = []
        succeeded = []
        failed = []

        for cmd in interpreted:
            # Execute each command
            result = await extplane_execute_command(cmd)
            result["command"] = cmd
            results.append(result)

            if result.get("status") in ["verified", "sent"]:
                succeeded.append(cmd)
            else:
                failed.append(cmd)

        # Aggregate status
        all_verified = all(r.get("status") == "verified" for r in results)
        any_success = len(succeeded) > 0

        if all_verified:
            status = "verified"
        elif any_success:
            status = "partial" if failed else "sent"
        else:
            status = "failed"

        return {
            "status": status,
            "input": text,
            "interpreted": interpreted,
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
            "ai_assisted": True
        }
    else:
        # Single command
        result = await extplane_execute_command(interpreted)
        result["input"] = text
        result["interpreted"] = interpreted
        result["ai_assisted"] = True
        return result


# =============================================================================
# Checklist Copilot API Endpoints
# =============================================================================

# Global checklist runner instance
_checklist_runner: Optional[ChecklistRunner] = None


def get_checklist_runner() -> ChecklistRunner:
    """Get or create the global ChecklistRunner."""
    global _checklist_runner
    if _checklist_runner is None:
        _checklist_runner = ChecklistRunner(get_client(), commands_loader=commands_loader)
    return _checklist_runner


@app.get("/api/version")
async def api_version():
    """Return server version for cache validation."""
    return {"version": SERVER_VERSION}


@app.get("/control", response_class=HTMLResponse)
async def control_page():
    """Serve the unified Control page."""
    with open("templates/control.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</head>", f'<meta name="server-version" content="{SERVER_VERSION}"></head>', 1)
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/checklist", response_class=HTMLResponse)
async def checklist_page():
    """Serve the Checklist Copilot UI with cache-busting version."""
    with open("templates/checklist.html", "r", encoding="utf-8") as f:
        html = f.read()
    # Inject version so browser always gets fresh content
    html = html.replace("</head>", f'<meta name="server-version" content="{SERVER_VERSION}"></head>', 1)
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.post("/api/checklist/load")
async def checklist_load(path: str = Form("")):
    """Load a checklist file (.xlsx, .csv, or .txt) from the given path or auto-detect."""
    runner = get_checklist_runner()

    # Ensure ExtPlane is connected
    client = get_client()
    if not client.is_connected:
        client.connect()
    runner.client = client

    if not path:
        # Auto-detect: check app directory first, then aircraft directory
        search_dirs = [Path(__file__).parent]
        aircraft_path = xplane_config.zibo_737_path
        if aircraft_path:
            search_dirs.append(Path(aircraft_path))
        filenames = ["CopilotAI_Checklists.xlsx", "CopilotAI_Checklists.csv", "Clist.txt", "clist.txt"]
        for directory in search_dirs:
            for name in filenames:
                candidate = directory / name
                if candidate.exists():
                    path = str(candidate)
                    break
            if path:
                break

    if not path:
        raise HTTPException(status_code=400, detail="No path provided and could not auto-detect checklist file")

    try:
        names = runner.load(path)
        return {"status": "ok", "checklists": names, "path": path,
                "summary": runner._load_summary}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/checklist/release")
async def checklist_release():
    """Stop the runner and clear all state so the file can be edited externally."""
    global _checklist_runner
    if _checklist_runner:
        _checklist_runner.stop()
    _checklist_runner = None
    return {"status": "ok", "message": "Checklist released for editing"}


@app.get("/api/checklist/list")
async def checklist_list():
    """List available checklists."""
    runner = get_checklist_runner()
    return {"checklists": runner.list_checklists()}


@app.post("/api/checklist/start")
async def checklist_start(name: str = Form(...), auto_run: bool = Form(False)):
    """Start a checklist by name. Set auto_run=true to begin executing immediately."""
    runner = get_checklist_runner()
    if runner.start(name, auto_run=auto_run):
        return {"status": "ok", "checklist": name, "auto_run": auto_run}
    else:
        raise HTTPException(status_code=404, detail=f"Checklist '{name}' not found")


@app.post("/api/checklist/run")
async def checklist_run(mode: str = Form("single")):
    """
    Run checklist items.

    mode=single: run current item (blocking, returns when done)
    mode=all: start auto-running in background thread (non-blocking)
    """
    runner = get_checklist_runner()
    if mode == "all":
        runner.run_all()
        return {"status": "ok", "message": "Auto-run started in background",
                "state": runner.get_status()}
    else:
        result = runner.run_single()
        return {"status": "ok", "result": result, "state": runner.get_status()}


@app.post("/api/checklist/pause")
async def checklist_pause():
    """Pause checklist automation."""
    runner = get_checklist_runner()
    runner.pause()
    return {"status": "ok", "state": runner.get_status()}


@app.post("/api/checklist/resume")
async def checklist_resume():
    """Resume checklist automation."""
    runner = get_checklist_runner()
    runner.resume()
    return {"status": "ok", "state": runner.get_status()}


@app.post("/api/checklist/confirm")
async def checklist_confirm():
    """Confirm a manual checklist item."""
    runner = get_checklist_runner()
    runner.confirm()
    return {"status": "ok", "state": runner.get_status()}


@app.post("/api/checklist/skip")
async def checklist_skip():
    """Skip the current checklist item."""
    runner = get_checklist_runner()
    runner.skip_item()
    return {"status": "ok", "state": runner.get_status()}


@app.post("/api/checklist/restart")
async def checklist_restart():
    """Restart the checklist from START, resetting all history."""
    runner = get_checklist_runner()
    runner.restart()
    return {"status": "ok", "state": runner.get_status()}


@app.post("/api/checklist/speech")
async def checklist_speech(enabled: bool = Form(False)):
    """Toggle speech (TTS) for checklist items."""
    runner = get_checklist_runner()
    runner.speech_enabled = enabled
    return {"status": "ok", "speech_enabled": runner.speech_enabled}


@app.post("/api/checklist/speech_done")
async def checklist_speech_done():
    """Signal that client-side TTS has finished speaking."""
    runner = get_checklist_runner()
    runner.speech_done()
    return {"status": "ok"}


@app.post("/api/checklist/auto_continue")
async def checklist_auto_continue(enabled: bool = Form(False)):
    """Toggle auto-continue behavior for sw_continue items."""
    runner = get_checklist_runner()
    runner.auto_continue = enabled
    return {"status": "ok", "auto_continue": runner.auto_continue}


@app.get("/api/checklist/status")
async def checklist_status():
    """Get current checklist state."""
    runner = get_checklist_runner()
    return runner.get_status()


@app.get("/api/checklist/completed")
async def checklist_completed():
    """Return list of completed checklist names."""
    runner = get_checklist_runner()
    return {"completed": runner.get_completed()}


@app.post("/api/checklist/clear_completed")
async def checklist_clear_completed():
    """Reset all completion tracking."""
    runner = get_checklist_runner()
    runner.clear_completed()
    return {"status": "ok", "message": "Completion states cleared"}


@app.post("/api/checklist/run_series")
async def checklist_run_series(start: str = Form(...)):
    """Run checklists in sequence starting from the specified checklist."""
    runner = get_checklist_runner()
    if runner.run_series(start):
        return {"status": "ok", "message": f"Running series from {start}",
                "state": runner.get_status()}
    else:
        raise HTTPException(status_code=404, detail=f"Checklist '{start}' not found")


# =============================================================================
# Annunciator Monitor API Endpoints
# =============================================================================

_annunciator_monitor: Optional[AnnunciatorMonitor] = None


def get_annunciator_monitor() -> AnnunciatorMonitor:
    """Get or create the global AnnunciatorMonitor."""
    global _annunciator_monitor
    if _annunciator_monitor is None:
        _annunciator_monitor = AnnunciatorMonitor(get_client())
    return _annunciator_monitor


@app.post("/api/annunciator/load")
async def annunciator_load(path: str = Form("")):
    """Load annunciators from an XLSX file."""
    monitor = get_annunciator_monitor()

    if not path:
        # Auto-detect annunciator file in app directory
        candidate = Path(__file__).parent / "CopilotAI_Annunciators.xlsx"
        if candidate.exists():
            path = str(candidate)

    if not path:
        raise HTTPException(status_code=400, detail="No path provided and could not find CopilotAI_Annunciators.xlsx")

    try:
        count = monitor.load_from_xlsx(path)
        return {"status": "ok", "count": count, "path": path}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/annunciator/start")
async def annunciator_start():
    """Start annunciator monitoring."""
    monitor = get_annunciator_monitor()

    # Ensure ExtPlane is connected
    client = get_client()
    if not client.is_connected:
        client.connect()
    monitor.extplane = client

    if monitor.start():
        return {"status": "ok", "message": "Annunciator monitoring started"}
    else:
        raise HTTPException(status_code=400, detail="Could not start monitor - check ExtPlane connection and loaded annunciators")


@app.post("/api/annunciator/stop")
async def annunciator_stop():
    """Stop annunciator monitoring."""
    monitor = get_annunciator_monitor()
    monitor.stop()
    return {"status": "ok", "message": "Annunciator monitoring stopped"}


@app.get("/api/annunciator/status")
async def annunciator_status():
    """Get annunciator monitor status."""
    monitor = get_annunciator_monitor()
    return monitor.get_status()


@app.get("/api/annunciator/list")
async def annunciator_list():
    """List all loaded annunciators."""
    monitor = get_annunciator_monitor()
    return {"annunciators": monitor.get_annunciators()}


@app.get("/api/annunciator/alert")
async def annunciator_alert():
    """Get pending alert (if any). Returns null if no alert pending."""
    monitor = get_annunciator_monitor()
    alert = monitor.get_pending_alert()
    return {"alert": alert}


@app.get("/api/annunciator/alerts")
async def annunciator_alerts(since: float = 0):
    """Get all alerts since a timestamp."""
    monitor = get_annunciator_monitor()
    return {"alerts": monitor.get_alerts(since)}


@app.post("/api/annunciator/clear")
async def annunciator_clear():
    """Clear all alerts."""
    monitor = get_annunciator_monitor()
    monitor.clear_alerts()
    return {"status": "ok"}


@app.post("/api/annunciator/test")
async def annunciator_test(message: str = Form("TEST ANNUNCIATOR")):
    """Manually trigger a test alert to verify the system works."""
    import time
    from src.xplane.annunciator_monitor import AnnunciatorAlert

    monitor = get_annunciator_monitor()

    # Create test alert directly
    alert = AnnunciatorAlert(
        name="TEST",
        message=message,
        timestamp=time.time()
    )

    # Add to pending queue
    with monitor._pending_lock:
        monitor._pending_alerts.append(alert)

    # Add to history
    with monitor._alerts_lock:
        monitor._alerts.append(alert)

    return {"status": "ok", "message": f"Test alert triggered: {message}"}


@app.post("/api/annunciator/toggle")
async def annunciator_toggle(enabled: str = Form("")):
    """
    Toggle annunciator master switch.

    enabled=true: Set rockets_armed=0 (annunciators ON)
    enabled=false/empty: Set rockets_armed=1 (annunciators OFF)
    """
    monitor = get_annunciator_monitor()

    # Parse enabled - treat 'true', '1', 'on' as True
    is_enabled = enabled.lower() in ('true', '1', 'on', 'yes')

    # Ensure ExtPlane is connected
    client = get_client()
    if not client.is_connected:
        client.connect()
    monitor.extplane = client

    if monitor.set_enabled(is_enabled):
        return {"status": "ok", "enabled": is_enabled}
    else:
        raise HTTPException(status_code=500, detail="Failed to set annunciator switch")


# =============================================================================
# FMS Programmer API Endpoints
# =============================================================================

_fms_programmer: Optional[FMSProgrammer] = None


def get_fms_programmer() -> FMSProgrammer:
    """Get or create the global FMSProgrammer."""
    global _fms_programmer
    if _fms_programmer is None:
        client = get_client()
        _fms_programmer = FMSProgrammer(
            client,
            keypress_delay=xplane_config.fms_keypress_delay,
            page_delay=xplane_config.fms_page_delay,
            verify_retries=xplane_config.fms_verify_retries,
            xplane_path=xplane_config.xplane_install_path,
        )
    return _fms_programmer


@app.get("/fms", response_class=HTMLResponse)
async def fms_page():
    """Serve the FMS Programmer UI."""
    with open("templates/fms.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</head>", f'<meta name="server-version" content="{SERVER_VERSION}"></head>', 1)
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.post("/api/fms/simbrief/fetch")
async def fms_simbrief_fetch(pilot_id: str = Form("")):
    """Fetch SimBrief OFP."""
    import asyncio
    fms = get_fms_programmer()
    pid = pilot_id or xplane_config.simbrief_pilot_id
    if not pid:
        raise HTTPException(status_code=400, detail="No SimBrief pilot ID provided")
    try:
        data = await asyncio.to_thread(fms.fetch_simbrief, pid)
        # Save pilot ID to config for next time
        if pilot_id and pilot_id != xplane_config.simbrief_pilot_id:
            xplane_config.simbrief_pilot_id = pilot_id
            xplane_config.save()
        return {
            "status": "ok",
            "origin": data.origin,
            "destination": data.destination,
            "route": data.route,
            "flight_number": data.flight_number,
            "cruise_altitude": data.cruise_altitude,
            "cost_index": data.cost_index,
            "zfw": data.zfw,
            "fuel_block": data.fuel_block,
            "pax_count": data.pax_count,
            "sid": data.sid,
            "star": data.star,
            "flap_setting": data.flap_setting,
            "v1": data.v1,
            "vr": data.vr,
            "v2": data.v2,
            "trim": data.trim,
            "assumed_temp": data.assumed_temp,
            "navlog_count": len(data.navlog),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fms/simbrief/data")
async def fms_simbrief_data():
    """Return cached SimBrief data."""
    fms = get_fms_programmer()
    data = fms.simbrief_data
    if not data:
        return {"status": "no_data"}
    return {
        "status": "ok",
        "origin": data.origin,
        "destination": data.destination,
        "route": data.route,
        "flight_number": data.flight_number,
        "cruise_altitude": data.cruise_altitude,
        "cost_index": data.cost_index,
        "zfw": data.zfw,
        "fuel_block": data.fuel_block,
        "fuel_taxi": data.fuel_taxi,
        "fuel_trip": data.fuel_trip,
        "fuel_reserve": data.fuel_reserve,
        "pax_count": data.pax_count,
        "cargo": data.cargo,
        "estimated_tow": data.estimated_tow,
        "sid": data.sid,
        "star": data.star,
        "origin_runway": data.origin_runway,
        "dest_runway": data.dest_runway,
        "flap_setting": data.flap_setting,
        "v1": data.v1,
        "vr": data.vr,
        "v2": data.v2,
        "trim": data.trim,
        "assumed_temp": data.assumed_temp,
        "trans_alt": data.trans_alt,
        "trans_level": data.trans_level,
        "navlog_count": len(data.navlog),
        "navlog": [
            {"ident": w.ident, "airway": w.airway, "altitude": w.altitude, "type": w.type}
            for w in data.navlog[:50]
        ],
    }


@app.post("/api/fms/clear")
async def fms_clear():
    """Clear the current FMC route."""
    fms = get_fms_programmer()
    client = get_client()
    if not client.is_connected:
        client.connect()
    await asyncio.to_thread(fms.clear_route)
    return {"status": "ok", "message": "FMC route cleared"}


@app.post("/api/fms/program")
async def fms_program():
    """Start full FMS programming sequence in background."""
    fms = get_fms_programmer()
    # Ensure ExtPlane connected
    client = get_client()
    if not client.is_connected:
        client.connect()
    fms.client = client
    fms.cdu.client = client
    try:
        fms.program_all()
        return {"status": "ok", "message": "Programming started"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/fms/program/{page}")
async def fms_program_page(page: str):
    """Program a single FMS page in background."""
    fms = get_fms_programmer()
    client = get_client()
    if not client.is_connected:
        client.connect()
    fms.client = client
    fms.cdu.client = client
    try:
        fms.program_page(page)
        return {"status": "ok", "page": page}
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/fms/uplink")
async def fms_uplink():
    """Trigger Zibo UPLINK (write b738x.xml and press REQUEST on PERF INIT)."""
    import asyncio
    fms = get_fms_programmer()
    client = get_client()
    if not client.is_connected:
        client.connect()
    fms.client = client
    try:
        await asyncio.to_thread(fms.trigger_uplink)
        return {"status": "ok", "message": "UPLINK triggered"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/fms/stop")
async def fms_stop():
    """Stop FMS programming."""
    fms = get_fms_programmer()
    fms.stop()
    return {"status": "ok"}


@app.post("/api/fms/reset")
async def fms_reset():
    """Reset FMS programmer state (clear page results, log, and state)."""
    fms = get_fms_programmer()
    fms.reset()
    return {"status": "ok", "message": "Programmer state reset"}


@app.post("/api/fms/release")
async def fms_release():
    """Destroy FMS programmer completely for a fresh start."""
    global _fms_programmer
    if _fms_programmer:
        _fms_programmer.stop()
    _fms_programmer = None
    return {"status": "ok", "message": "FMS programmer released"}


@app.get("/api/fms/status")
async def fms_status():
    """Get FMS programmer status."""
    fms = get_fms_programmer()
    return fms.get_status()


@app.get("/api/fms/cdu/screen")
async def fms_cdu_screen():
    """Read current CDU screen."""
    import asyncio
    fms = get_fms_programmer()
    return await asyncio.to_thread(fms.read_cdu_screen)


# =============================================================================
# Persistent Settings API
# =============================================================================

@app.get("/api/settings")
async def get_settings():
    """Return all control page settings."""
    xplane_config.reload()
    return xplane_config.control_settings


@app.post("/api/settings")
async def update_settings(request: Request):
    """Update one or more control page settings."""
    body = await request.json()
    xplane_config.update_control_settings(body)
    xplane_config.save()
    return xplane_config.control_settings


# =============================================================================
# Cached SimBrief Plans API
# =============================================================================

@app.get("/api/fms/simbrief/cached")
async def fms_simbrief_cached():
    """List cached SimBrief plans."""
    fms = get_fms_programmer()
    plans = fms._simbrief_client.list_cached()
    return {"plans": plans}


@app.post("/api/fms/simbrief/delete_cached")
async def fms_simbrief_delete_cached(filename: str = Form(...)):
    """Delete a cached SimBrief plan and its corresponding X-Plane .fms file."""
    fms = get_fms_programmer()
    # Read origin+dest before deleting so we can remove the .fms file
    route_key = None
    try:
        cache_path = Path(fms._simbrief_client._CACHE_DIR) / filename
        if cache_path.exists():
            import json as _json
            meta = _json.loads(cache_path.read_text(encoding="utf-8"))
            orig = meta.get("origin", "")
            dest = meta.get("destination", "")
            if orig and dest:
                route_key = f"{orig}{dest}"
    except Exception:
        pass
    deleted = fms._simbrief_client.delete_cached(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Cached plan not found")
    # Also remove the .fms file from X-Plane if no other cache uses this route
    if route_key and fms.xplane_path:
        remaining = [p for p in fms._simbrief_client.list_cached()
                     if f"{p['origin']}{p['destination']}" == route_key]
        if not remaining:
            fms_file = Path(fms.xplane_path) / "Output" / "FMS plans" / f"{route_key}.fms"
            if fms_file.exists():
                fms_file.unlink()
    return {"status": "ok"}


@app.post("/api/fms/simbrief/load_cached")
async def fms_simbrief_load_cached(filename: str = Form(...)):
    """Load a cached SimBrief plan by filename."""
    fms = get_fms_programmer()
    try:
        data = fms._simbrief_client.load_cached(filename)
        fms._data = data
        # Also write the FMS plan file for CO ROUTE
        try:
            fms._write_fms_plan()
        except Exception as e:
            pass
        return {
            "status": "ok",
            "origin": data.origin,
            "destination": data.destination,
            "route": data.route,
            "flight_number": data.flight_number,
            "cruise_altitude": data.cruise_altitude,
            "cost_index": data.cost_index,
            "zfw": data.zfw,
            "fuel_block": data.fuel_block,
            "pax_count": data.pax_count,
            "sid": data.sid,
            "star": data.star,
            "flap_setting": data.flap_setting,
            "v1": data.v1,
            "vr": data.vr,
            "v2": data.v2,
            "trim": data.trim,
            "assumed_temp": data.assumed_temp,
            "navlog_count": len(data.navlog),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Hardware PTT via ExtPlane (joystick button monitoring)
# =============================================================================

import threading
import time as _time

_ptt_lock = threading.Lock()
_ptt_pressed = False
_ptt_monitoring = False
_ptt_discover_mode = False
_ptt_discovered_index = -1
_ptt_monitor_thread: Optional[threading.Thread] = None
_ptt_stop_event = threading.Event()


def _ptt_monitor_loop():
    """Poll joystick button values from ExtPlane to detect PTT presses."""
    global _ptt_pressed
    button_index = xplane_config.get_control_setting("hardware_ptt_button_index")
    if button_index < 0:
        return

    client = get_client()
    dataref = "sim/joystick/joystick_button_values"

    while not _ptt_stop_event.is_set():
        try:
            if not client.is_connected:
                _time.sleep(0.5)
                continue
            val = client.get_dataref(dataref, timeout=0.3)
            if val is not None:
                # val is an array of ints — parse it
                if isinstance(val, (list, tuple)):
                    buttons = val
                elif isinstance(val, str) and val.startswith("["):
                    buttons = json.loads(val)
                else:
                    buttons = []
                if 0 <= button_index < len(buttons):
                    with _ptt_lock:
                        _ptt_pressed = bool(int(buttons[button_index]))
        except Exception:
            pass
        _time.sleep(0.05)  # ~20Hz poll rate


def _start_ptt_monitor():
    """Start the PTT monitor thread if not already running."""
    global _ptt_monitor_thread, _ptt_monitoring
    if _ptt_monitoring:
        return
    _ptt_stop_event.clear()
    _ptt_monitor_thread = threading.Thread(target=_ptt_monitor_loop, daemon=True, name="ptt-monitor")
    _ptt_monitor_thread.start()
    _ptt_monitoring = True


def _stop_ptt_monitor():
    """Stop the PTT monitor thread."""
    global _ptt_monitoring, _ptt_pressed
    _ptt_stop_event.set()
    if _ptt_monitor_thread and _ptt_monitor_thread.is_alive():
        _ptt_monitor_thread.join(timeout=2)
    _ptt_monitoring = False
    with _ptt_lock:
        _ptt_pressed = False


@app.get("/api/ptt/status")
async def ptt_status():
    """Get current hardware PTT state."""
    enabled = xplane_config.get_control_setting("hardware_ptt_enabled")
    with _ptt_lock:
        pressed = _ptt_pressed
    return {"pressed": pressed, "enabled": enabled, "monitoring": _ptt_monitoring}


@app.post("/api/ptt/enable")
async def ptt_enable(enabled: str = Form("true")):
    """Enable or disable hardware PTT monitoring."""
    is_enabled = enabled.lower() in ('true', '1', 'on', 'yes')
    xplane_config.set_control_setting("hardware_ptt_enabled", is_enabled)
    xplane_config.save()

    if is_enabled:
        btn_idx = xplane_config.get_control_setting("hardware_ptt_button_index")
        if btn_idx >= 0:
            _start_ptt_monitor()
    else:
        _stop_ptt_monitor()

    return {"status": "ok", "enabled": is_enabled}


@app.post("/api/ptt/discover")
async def ptt_discover():
    """Start button discovery mode — watches for any joystick button press."""
    global _ptt_discover_mode, _ptt_discovered_index
    _ptt_discover_mode = True
    _ptt_discovered_index = -1

    def discover():
        global _ptt_discovered_index, _ptt_discover_mode
        client = get_client()
        dataref = "sim/joystick/joystick_button_values"
        baseline = None
        timeout = _time.time() + 10  # 10 second timeout

        while _time.time() < timeout and _ptt_discover_mode:
            try:
                if not client.is_connected:
                    _time.sleep(0.2)
                    continue
                val = client.get_dataref(dataref, timeout=0.3)
                if val is None:
                    _time.sleep(0.1)
                    continue
                if isinstance(val, (list, tuple)):
                    buttons = [int(b) for b in val]
                elif isinstance(val, str) and val.startswith("["):
                    buttons = [int(b) for b in json.loads(val)]
                else:
                    _time.sleep(0.1)
                    continue

                if baseline is None:
                    baseline = buttons[:]
                    _time.sleep(0.1)
                    continue

                # Check for any button that went from 0 to 1
                for i in range(min(len(baseline), len(buttons))):
                    if baseline[i] == 0 and buttons[i] == 1:
                        _ptt_discovered_index = i
                        _ptt_discover_mode = False
                        return

                baseline = buttons[:]
            except Exception:
                pass
            _time.sleep(0.05)
        _ptt_discover_mode = False

    threading.Thread(target=discover, daemon=True, name="ptt-discover").start()
    return {"status": "ok", "message": "Listening for button press (10s timeout)"}


@app.get("/api/ptt/discover_result")
async def ptt_discover_result():
    """Check if a button has been discovered."""
    return {
        "discovering": _ptt_discover_mode,
        "button_index": _ptt_discovered_index,
    }


@app.post("/api/ptt/save_button")
async def ptt_save_button(button_index: int = Form(...)):
    """Save the discovered button index to settings."""
    xplane_config.set_control_setting("hardware_ptt_button_index", button_index)
    xplane_config.save()
    return {"status": "ok", "button_index": button_index}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
