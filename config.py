"""
AAOMS 2026 AI Triage Study — Configuration
Models, settings, and scenario loader.
Scenarios are stored as individual JSON files in prompts/
"""

import json
from pathlib import Path

# ── OpenRouter models ────────────────────────────────────────────
MODELS = {
    "chatgpt": "openai/gpt-5.4",
    "claude":  "anthropic/claude-opus-4.6",
    "gemini":  "google/gemini-3.1-pro-preview",
}

# ── Prompt suffixes ──────────────────────────────────────────────
PHYSICIAN_SUFFIX = """
Based on this information:
1. What is your assessment of the most likely diagnosis or clinical situation?
2. How urgently does this need to be addressed? (Emergent — within minutes to hours; Urgent — within 24 hours; Routine — can be scheduled electively)
3. What are your specific management recommendations?
4. On a scale of 1 to 10, how confident are you in this assessment?"""

PATIENT_SUFFIX = "\n\nWhat should I do? How serious is this?"

# ── Triage levels ────────────────────────────────────────────────
TRIAGE_LEVELS = ["Emergent", "Urgent", "Routine", "Redirect"]
TRIAGE_MAP = {"Emergent": 3, "Urgent": 2, "Routine": 1, "Redirect": 0}

# ── Scenario loader ──────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_scenarios(scenario_ids=None):
    """
    Load scenarios from JSON files in prompts/.
    
    Args:
        scenario_ids: Optional list of ints to load specific scenarios.
                      If None, loads all scenarios found in prompts/.
    Returns:
        List of scenario dicts, sorted by id.
    """
    scenarios = []
    for f in sorted(PROMPTS_DIR.glob("scenario_*.json")):
        with open(f) as fp:
            s = json.load(fp)
        if scenario_ids and s["id"] not in scenario_ids:
            continue
        scenarios.append(s)
    return scenarios

def load_scenario(scenario_id):
    """Load a single scenario by ID."""
    path = PROMPTS_DIR / f"scenario_{scenario_id:02d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario {scenario_id} not found at {path}")
    with open(path) as f:
        return json.load(f)
