# Research Replay

Exploratory POC. Testing an idea.

## The idea

Take a published paper + its dataset + a research question. Ask an LLM to "replay" the investigation step by step in two modes:

- **Privileged**: the model has the paper (knows how it ends)
- **Base**: the model only has the data and the question

Compare the trajectories. Where they diverge, extract preference pairs. The hypothesis is that the privileged model makes better research decisions — but we don't know if that's true yet.

## Status

Early stage. Pipeline scaffolded, not tested end-to-end. Starting with 3 tasks from [DiscoveryBench](https://github.com/allenai/discoverybench).

## Setup

```bash
conda activate research-taste
pip install -r requirements.txt
git clone https://github.com/allenai/discoverybench data/discoverybench
cp .env.example .env  # fill in Azure AI Foundry credentials
```

## Usage

```bash
python src/extract.py --task biology_fish --output data/tasks/biology_fish.json
python src/generate_privi.py --task data/tasks/biology_fish.json --paper data/papers/cerezer2023.txt --output trajectories/biology_fish/privi_1.json
python src/generate_base.py --task data/tasks/biology_fish.json --output trajectories/biology_fish/base_1.json
python src/extract_forks.py --privi trajectories/biology_fish/privi_1.json --base trajectories/biology_fish/base_1.json --output trajectories/biology_fish/forks.json
python src/format_eval.py --forks trajectories/biology_fish/forks.json --output eval/biology_fish_pairs.json
```
