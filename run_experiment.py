"""
AAOMS 2026 AI Triage Study — Experiment Runner
Sends all prompts to models via OpenRouter and saves raw outputs.

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python run_experiment.py
    
    # Or run a single model/frame for testing:
    python run_experiment.py --model chatgpt --frame physician --scenarios 1,2,3
"""

import asyncio
import httpx
import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from config import MODELS, SCENARIOS

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


async def query_model(client: httpx.AsyncClient, api_key: str, model_id: str, prompt: str, run_id: str) -> dict:
    """Send a single prompt to a model via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nisargpatel/aaoms-triage-study",
        "X-Title": "AAOMS AI Triage Study",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.0,  # Deterministic for reproducibility
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
    except httpx.HTTPStatusError as e:
        return {
            "run_id": run_id,
            "status": "error",
            "error": f"HTTP {e.response.status_code}: {e.response.text[:500]}",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def run_experiment(model_filter=None, frame_filter=None, scenario_filter=None):
    """Run all experiment conditions and save results."""
    api_key = get_api_key()
    results = []
    run_counter = 0
    
    # Build the run list
    runs = []
    for model_name, model_id in MODELS.items():
        if model_filter and model_name != model_filter:
            continue
        for scenario in SCENARIOS:
            if scenario_filter and scenario["id"] not in scenario_filter:
                continue
            
            frames_to_run = []
            if (not frame_filter or frame_filter == "physician"):
                frames_to_run.append("physician")
            if scenario["has_patient_frame"] and (not frame_filter or frame_filter == "patient"):
                frames_to_run.append("patient")
            
            for frame in frames_to_run:
                run_counter += 1
                prompt = scenario["physician_prompt"] if frame == "physician" else scenario["patient_prompt"]
                
                # Add the appropriate suffix for physician frames
                if frame == "physician" and scenario["category"] != "E":
                    from config import PHYSICIAN_SUFFIX
                    prompt = prompt + PHYSICIAN_SUFFIX
                # Patient frames already have their suffix baked into the prompt text
                # Category E prompts already have their questions baked in
                
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

    total = len(runs)
    print(f"\n{'='*60}")
    print(f"AAOMS 2026 AI Triage Study — Experiment Runner")
    print(f"{'='*60}")
    print(f"Total runs: {total}")
    print(f"Models: {', '.join(m for m in MODELS if not model_filter or m == model_filter)}")
    print(f"Scenarios: {len(set(r['scenario_id'] for r in runs))}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient() as client:
        for i, run in enumerate(runs):
            print(f"[{i+1}/{total}] {run['run_id']}: Scenario {run['scenario_id']} "
                  f"({run['scenario_name'][:30]}...) | {run['model_name']} | {run['frame']}")
            
            result = await query_model(client, api_key, run["model_id"], run["prompt"], run["run_id"])
            
            # Merge run metadata with result
            full_result = {**run, **result}
            del full_result["prompt"]  # Don't duplicate the prompt in results (it's in the run metadata)
            full_result["prompt_text"] = run["prompt"]  # Keep for reference
            results.append(full_result)
            
            if result["status"] == "success":
                print(f"         ✓ Success ({result['usage'].get('total_tokens', '?')} tokens)")
            else:
                print(f"         ✗ Error: {result.get('error', 'unknown')[:80]}")
            
            # Rate limiting — be gentle with OpenRouter
            await asyncio.sleep(1.5)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = DATA_DIR / f"experiment_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"{'='*60}")

    # Also save a simplified CSV for quick import to the scoring spreadsheet
    csv_file = DATA_DIR / f"experiment_outputs_{timestamp}.csv"
    import csv
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run ID", "Scenario #", "Category", "Scenario Name", "Model", "Frame",
                         "Model Response", "Status"])
        for r in results:
            writer.writerow([
                r["run_id"], r["scenario_id"], r["category"], r["scenario_name"],
                r["model_name"], r["frame"],
                r.get("response", r.get("error", "")),
                r["status"],
            ])
    print(f"CSV export saved to {csv_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AAOMS AI Triage Study — Run Experiment")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Run only this model")
    parser.add_argument("--frame", choices=["physician", "patient"], help="Run only this frame")
    parser.add_argument("--scenarios", type=str, help="Comma-separated scenario IDs (e.g., 1,2,3)")
    parser.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    args = parser.parse_args()

    scenario_filter = None
    if args.scenarios:
        scenario_filter = [int(x) for x in args.scenarios.split(",")]

    if args.dry_run:
        print("DRY RUN — would execute:")
        count = 0
        for model_name in MODELS:
            if args.model and model_name != args.model:
                continue
            for s in SCENARIOS:
                if scenario_filter and s["id"] not in scenario_filter:
                    continue
                frames = []
                if not args.frame or args.frame == "physician":
                    frames.append("physician")
                if s["has_patient_frame"] and (not args.frame or args.frame == "patient"):
                    frames.append("patient")
                for frame in frames:
                    count += 1
                    print(f"  {count:3d}. Scenario {s['id']:2d} ({s['name'][:35]:35s}) | {model_name:8s} | {frame}")
        print(f"\nTotal: {count} runs")
        return

    asyncio.run(run_experiment(
        model_filter=args.model,
        frame_filter=args.frame,
        scenario_filter=scenario_filter,
    ))


if __name__ == "__main__":
    main()
