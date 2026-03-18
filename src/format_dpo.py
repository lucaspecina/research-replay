"""Convert anchored pairs to DPO training format (TRL-compatible JSONL).

Usage:
    python src/format_dpo.py \
        --input trajectories/biology_fish/anchored_1.json \
        --output training_data/biology_fish.jsonl

    # Multiple inputs:
    python src/format_dpo.py \
        --input trajectories/*/anchored_*.json \
        --output training_data/all_pairs.jsonl \
        --min-divergence 0.3
"""

import argparse
import glob
import json
import os


def anchored_to_dpo(anchored_data, min_divergence=0.2):
    """Convert anchored pairs to DPO format.

    Each pair becomes one record:
    - prompt: shared context (research question + previous steps + results)
    - chosen: privi's response (reasoning + action + code)
    - rejected: base's response (reasoning + action + code)
    """
    records = []
    task_id = anchored_data.get("task_id", "unknown")
    trajectory = anchored_data.get("privi_trajectory", {})
    steps = trajectory.get("steps", [])

    for pair in anchored_data.get("pairs", []):
        if pair.get("divergence_score", 0) < min_divergence:
            continue

        step_num = pair["step_number"]

        # Build prompt: shared context up to this step
        prompt_parts = [f"Task: {task_id}"]
        prompt_parts.append(f"Step {step_num} of investigation.")

        # Include previous steps as context
        for prev_step in steps[:step_num - 1]:
            prompt_parts.append(
                f"\nStep {prev_step['step_number']} [{prev_step['action_type']}]: "
                f"{prev_step['action']}\n"
                f"Result: {prev_step.get('actual_outcome', '')[:500]}"
            )

        prompt_parts.append("\nWhat analysis should be done next?")
        prompt = "\n".join(prompt_parts)

        # Format chosen/rejected as structured text
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        chosen_text = (
            f"Action type: {chosen.get('action_type', '')}\n"
            f"Reasoning: {chosen.get('reasoning', '')}\n"
            f"Action: {chosen.get('action', '')}\n"
            f"Code:\n```python\n{chosen.get('code', '')}\n```"
        )

        rejected_text = (
            f"Action type: {rejected.get('action_type', '')}\n"
            f"Reasoning: {rejected.get('reasoning', '')}\n"
            f"Action: {rejected.get('action', '')}\n"
            f"Code:\n```python\n{rejected.get('code', '')}\n```"
        )

        records.append({
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "task_id": task_id,
            "step_number": step_num,
            "divergence_score": pair.get("divergence_score", 0),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Convert anchored pairs to DPO JSONL")
    parser.add_argument("--input", required=True, nargs="+", help="Anchored JSON file(s) or glob")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--min-divergence", type=float, default=0.2,
                        help="Min divergence score to include pair")
    args = parser.parse_args()

    # Expand globs
    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(pattern))

    if not input_files:
        print(f"No input files found for: {args.input}")
        return

    all_records = []
    for path in input_files:
        print(f"Processing: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = anchored_to_dpo(data, args.min_divergence)
        all_records.extend(records)
        print(f"  {len(records)} pairs extracted")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_records)} DPO pairs written to {args.output}")


if __name__ == "__main__":
    main()
