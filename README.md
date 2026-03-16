# AAOMS 2026 AI Triage Study

**"The 2AM Call vs. The 2AM Google"**

Testing whether frontier LLMs safely triage OMS emergencies — and whether triage recommendations change when a doctor asks vs. when a patient asks.

## Quick Start

```bash
# 1. Install dependencies
pip install httpx pandas numpy matplotlib scipy

# 2. Set your OpenRouter API key
export OPENROUTER_API_KEY=sk-or-v1-...
# (or create a .env file with OPENROUTER_API_KEY=sk-or-v1-...)

# 3. Dry run to see all planned experiment runs
python run_experiment.py --dry-run

# 4. Test with a single scenario first
python run_experiment.py --model chatgpt --scenarios 1

# 5. Run the full experiment (~135 runs, ~3-4 hours with rate limiting)
python run_experiment.py

# 6. After scoring, run analysis
python analyze.py data/consensus_scores.csv

# 7. Compute inter-rater reliability
python compute_irr.py data/scorer1.csv data/scorer2.csv data/scorer3.csv
```

## Project Structure

```
aaoms-study/
├── config.py              # All 25 scenarios, prompts, scoring keys, model config
├── run_experiment.py      # API runner — sends prompts to models via OpenRouter
├── analyze.py             # Analysis pipeline — generates all 6 figures + statistics
├── compute_irr.py         # Inter-rater reliability (Fleiss' kappa)
├── README.md
├── data/                  # Raw outputs and scored data (created at runtime)
│   ├── experiment_results_*.json    # Full API responses
│   ├── experiment_outputs_*.csv     # Simplified CSV for spreadsheet import
│   └── consensus_scores.csv         # Export from scoring spreadsheet → analyze.py
├── results/               # Summary statistics (created at runtime)
└── figures/               # All publication figures (created by analyze.py)
    ├── fig1_triage_confusion.png/pdf
    ├── fig2_framing_gap.png/pdf
    ├── fig3_calibration.png/pdf
    ├── fig4_undertriage_by_category.png/pdf
    ├── fig5_safe_uncertainty.png/pdf
    └── fig6_error_taxonomy.png/pdf
```

## Workflow

### Days 1-3: Prep
- Review `config.py` — verify all 25 scenarios, prompts, and scoring keys
- Update model strings in `config.py` if newer versions are available

### Days 4-6: Run Experiment
```bash
# Run all 135 evaluations
python run_experiment.py

# Or run model-by-model if you prefer
python run_experiment.py --model chatgpt
python run_experiment.py --model claude
python run_experiment.py --model gemini
```

Results are saved to `data/experiment_results_TIMESTAMP.json` and `data/experiment_outputs_TIMESTAMP.csv`.

### Days 7-9: Score
1. Import `experiment_outputs_*.csv` into the **Model Outputs** tab of the scoring spreadsheet
2. Distribute scorer sheets to your two OMS residents
3. All 3 scorers independently score all 135 outputs
4. Merge into the **Consensus Scores** tab
5. Export Consensus Scores tab as `data/consensus_scores.csv`

### Days 10-12: Analyze
```bash
# Generate all figures and statistics
python analyze.py data/consensus_scores.csv

# Compute inter-rater reliability
# (Export individual scorer tabs as CSVs first)
python compute_irr.py data/scorer1.csv data/scorer2.csv data/scorer3.csv
```

### Days 13-17: Write and Submit
- Figures are in `figures/` directory (PNG for slides, PDF for manuscript)
- Statistics are printed to console by `analyze.py`

## Models

Uses [OpenRouter](https://openrouter.ai) as unified API. Default models in `config.py`:
- **ChatGPT** — `openai/gpt-4o-2024-11-20` (update to GPT-5 when available)
- **Claude** — `anthropic/claude-sonnet-4-20250514`
- **Gemini** — `google/gemini-2.5-pro-preview-05-06`

Update model strings before running if newer versions are available.

## Study Design

- **25 scenarios** across 5 categories (post-op complications, ED consults, dentist referrals, patient direct queries, anesthesia emergencies)
- **3 models** × **2 frames** (physician + patient) for Categories A-D = 120 runs
- **3 models** × **1 frame** (physician only) for Category E = 15 runs  
- **Total: 135 runs**
- **3 independent scorers** (1 attending + 2 residents)
- Category E excluded from paired framing analysis (no natural patient-frame analogue)
