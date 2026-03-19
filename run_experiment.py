"""
AAOMS 2026 AI Triage Study — Experiment Runner
Loads scenarios from prompts/*.json, sends to models via OpenRouter, saves results.

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python run_experiment.py
    python run_experiment.py --model chatgpt --frame physician --scenarios 1,2,3
    python run_experiment.py --dry-run
"""

import asyncio
import csv
import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from config import MODELS, PHYSICIAN_SUFFIX, load_scenarios

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable or add to .env file")
        sys.exit(1)
    return key


async def query_model(client, api_key, model_id, prompt, run_id):
    """Send a single prompt to a model via OpenRouter."""
    import httpx
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nisargpatel/oms-triage",
        "X-Title": "AAOMS AI Triage Study",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    try:
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return {
            "run_id": run_id,
            "status": "success",
            "response": content,
            "model_used": data.get("model", model_id),
            "usage": data.get("usage", {}),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response'):
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
        return {
            "run_id": run_id,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }


def build_runs(scenarios, model_filter=None, frame_filter=None):
    """Build the list of all experiment runs from loaded scenarios."""
    runs = []
    run_counter = 0

    for model_name, model_id in MODELS.items():
        if model_filter and model_name != model_filter:
            continue
        for scenario in scenarios:
            frames_to_run = []
            if not frame_filter or frame_filter == "physician":
                frames_to_run.append("physician")
            if scenario["has_patient_frame"] and (not frame_filter or frame_filter == "patient"):
                frames_to_run.append("patient")

            for frame in frames_to_run:
                run_counter += 1
                if frame == "physician":
                    prompt = scenario["physician_prompt"]
                    # Category E prompts already have questions baked in
                    if scenario["category"] != "E":
                        prompt = prompt + PHYSICIAN_SUFFIX
                else:
                    prompt = scenario["patient_prompt"]

                runs.append({
                    "run_id": f"R{run_counter:03d}",
                    "scenario_id": scenario["id"],
                    "scenario_name": scenario["name"],
                    "category": scenario["category"],
                    "correct_triage": scenario["correct_triage"],
                    "model_name": model_name,
                    "model_id": model_id,
                    "frame": frame,
                    "prompt": prompt,
                })
    return runs


async def run_experiment(model_filter=None, frame_filter=None, scenario_ids=None):
    """Run all experiment conditions and save results."""
    import httpx

    api_key = get_api_key()
    scenarios = load_scenarios(scenario_ids)
    if not scenarios:
        print("ERROR: No scenarios found in prompts/. Run extract_scenarios.py first.")
        sys.exit(1)

    runs = build_runs(scenarios, model_filter, frame_filter)
    total = len(runs)
    results = []

    print(f"\n{'='*60}")
    print(f"AAOMS 2026 AI Triage Study — Experiment Runner")
    print(f"{'='*60}")
    print(f"Scenarios loaded: {len(scenarios)} (from prompts/*.json)")
    print(f"Total runs: {total}")
    print(f"Models: {', '.join(m for m in MODELS if not model_filter or m == model_filter)}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient() as client:
        for i, run in enumerate(runs):
            print(f"[{i+1}/{total}] {run['run_id']}: S{run['scenario_id']:02d} "
                  f"({run['scenario_name'][:30]}...) | {run['model_name']} | {run['frame']}")

            result = await query_model(client, api_key, run["model_id"], run["prompt"], run["run_id"])

            full_result = {**run, **result}
            full_result["prompt_text"] = full_result.pop("prompt")
            results.append(full_result)

            if result["status"] == "success":
                print(f"         ✓ ({result['usage'].get('total_tokens', '?')} tokens)")
            else:
                print(f"         ✗ {result.get('error', 'unknown')[:80]}")

            await asyncio.sleep(1.5)

    # Save JSON (full results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = DATA_DIR / f"experiment_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV (for spreadsheet import)
    csv_file = DATA_DIR / f"experiment_outputs_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run ID", "Scenario #", "Category", "Scenario Name", "Model",
                         "Frame", "Correct Triage", "Model Response", "Status"])
        for r in results:
            writer.writerow([
                r["run_id"], r["scenario_id"], r["category"], r["scenario_name"],
                r["model_name"], r["frame"], r["correct_triage"],
                r.get("response", r.get("error", "")), r["status"],
            ])

    # Save ground truth reference (from scenario JSONs, for analysis)
    ground_truth_file = DATA_DIR / f"ground_truth_{timestamp}.json"
    gt = []
    for s in scenarios:
        gt.append({
            "id": s["id"],
            "category": s["category"],
            "name": s["name"],
            "correct_triage": s["correct_triage"],
            "has_patient_frame": s["has_patient_frame"],
            "critical_elements": s["critical_elements"],
            "dangerous_recs": s["dangerous_recs"],
            "num_critical_elements": len(s["critical_elements"]),
        })
    with open(ground_truth_file, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results:      {json_file}")
    print(f"CSV export:   {csv_file}")
    print(f"Ground truth: {ground_truth_file}")
    print(f"Total: {len(results)} | Success: {sum(1 for r in results if r['status'] == 'success')} | "
          f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AAOMS AI Triage Study — Run Experiment")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Run only this model")
    parser.add_argument("--frame", choices=["physician", "patient"], help="Run only this frame")
    parser.add_argument("--scenarios", type=str, help="Comma-separated scenario IDs (e.g., 1,2,3)")
    parser.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    args = parser.parse_args()

    scenario_ids = [int(x) for x in args.scenarios.split(",")] if args.scenarios else None
    scenarios = load_scenarios(scenario_ids)

    if not scenarios:
        print("ERROR: No scenarios found. Check that prompts/*.json files exist.")
        sys.exit(1)

    if args.dry_run:
        runs = build_runs(scenarios, args.model, args.frame)
        print(f"DRY RUN — {len(runs)} runs planned:\n")
        for i, r in enumerate(runs, 1):
            print(f"  {i:3d}. S{r['scenario_id']:02d} ({r['scenario_name'][:35]:35s}) | "
                  f"{r['model_name']:8s} | {r['frame']}")
        print(f"\nTotal: {len(runs)} runs")
        print(f"Scenarios: {len(scenarios)} loaded from prompts/*.json")
        return

    asyncio.run(run_experiment(
        model_filter=args.model,
        frame_filter=args.frame,
        scenario_ids=scenario_ids,
    ))


if __name__ == "__main__":
    main()
