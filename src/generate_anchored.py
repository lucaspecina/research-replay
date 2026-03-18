"""Generate privi-anchored DPO pairs.

Two-pass process:
1. Run privi trajectory (with paper, real code execution)
2. At each checkpoint, query base (no paper) from the SAME state

This produces N clean DPO pairs where both decisions come from identical context.

Usage:
    python src/generate_anchored.py \
        --task data/tasks/biology_fish.json \
        --paper data/papers/cerezer2023.txt \
        --output trajectories/biology_fish/anchored_1.json \
        --steps 6
"""

import argparse
import asyncio

from common import load_task, load_paper, load_prompt, save_json
from llm import call_messages, extract_json, get_model
from trajectory_runner import TrajectoryRunner
from verifier import VerifierTool


async def query_base_at_checkpoint(snapshot_messages, base_system_prompt, step_prompt):
    """Query the base model from a privi checkpoint.

    Takes the exact messages the privi saw, swaps system prompt,
    removes paper, and asks the base what it would do.
    """
    # Replace system prompt
    base_messages = [{"role": "system", "content": base_system_prompt}]

    # Copy user/assistant messages, removing paper from first user message
    for msg in snapshot_messages[1:]:  # skip original system prompt
        if msg["role"] == "user" and "PRIVILEGED INFORMATION" in msg.get("content", ""):
            # Remove paper section from context
            content = msg["content"]
            paper_start = content.find("## Published Paper (PRIVILEGED INFORMATION)")
            if paper_start >= 0:
                paper_end = content.find("\n## ", paper_start + 1)
                if paper_end < 0:
                    paper_end = content.find("\nI will now ask you", paper_start)
                if paper_end >= 0:
                    content = content[:paper_start] + content[paper_end:]
                else:
                    content = content[:paper_start]
            base_messages.append({"role": msg["role"], "content": content})
        else:
            base_messages.append(dict(msg))

    # Add the step prompt
    base_messages.append({"role": "user", "content": step_prompt})

    text = await call_messages(base_messages, max_tokens=4096)
    return extract_json(text)


async def generate_anchored(task, paper_text, num_steps=6):
    """Two-pass: run privi, then query base at each checkpoint."""
    model = get_model()
    privi_system = load_prompt("privi_system.txt")
    base_system = load_prompt("base_system.txt")

    # === Pass 1: Run privi trajectory ===
    print(f"=== Pass 1: Privi trajectory ({model}) ===")
    runner = TrajectoryRunner(task, paper_text, privi_system, num_steps)
    await runner.run_all_steps()
    await runner.run_final_hypothesis()

    print(f"  Privi done: {len(runner.steps)} steps")
    print(f"  Hypothesis: {runner.final_hypothesis[:100]}...")

    # === Pass 2: Query base at each checkpoint ===
    print(f"\n=== Pass 2: Base counterfactuals ({model}) ===")
    pairs = []

    for step in runner.steps:
        step_num = step.step_number
        print(f"  Base counterfactual for step {step_num}...")

        # Get the exact messages snapshot from before this step
        snapshot = runner.get_snapshot(step_num)
        if not snapshot:
            print(f"    WARNING: no snapshot for step {step_num}, skipping")
            continue

        # Build the step prompt that privi saw
        # We need to reconstruct it — it's the last user message before the LLM response
        step_prompt = None
        for msg in reversed(snapshot):
            if msg["role"] == "user" and "step" in msg.get("content", "").lower():
                step_prompt = msg["content"]
                break

        if not step_prompt:
            print(f"    WARNING: could not find step prompt for step {step_num}")
            continue

        # Remove the step prompt from snapshot (we'll add it separately)
        snapshot_without_prompt = snapshot[:-1]

        # Query base
        base_step = await query_base_at_checkpoint(
            snapshot_without_prompt, base_system, step_prompt
        )

        if base_step is None:
            print(f"    WARNING: could not parse base response for step {step_num}")
            continue

        # Compute divergence score
        divergence = VerifierTool.score_pair_divergence(
            step.to_dict(), base_step
        )

        pair = {
            "step_number": step_num,
            "divergence_score": round(divergence, 3),
            "chosen": {
                "source": "privi",
                "action_type": step.action_type,
                "reasoning": step.reasoning,
                "action": step.action,
                "code": step.code,
                "expected_outcome": step.expected_outcome,
            },
            "rejected": {
                "source": "base",
                "action_type": base_step.get("action_type", ""),
                "reasoning": base_step.get("reasoning", ""),
                "action": base_step.get("action", ""),
                "code": base_step.get("code", ""),
                "expected_outcome": base_step.get("expected_outcome", ""),
            },
        }
        pairs.append(pair)
        print(f"    OK (divergence: {divergence:.2f})")

    return runner, pairs


def main():
    parser = argparse.ArgumentParser(description="Generate privi-anchored DPO pairs")
    parser.add_argument("--task", required=True, help="Path to task JSON")
    parser.add_argument("--paper", required=True, help="Path to paper text file")
    parser.add_argument("--output", required=True, help="Output anchored pairs JSON")
    parser.add_argument("--steps", type=int, default=6, help="Number of steps (5-8)")
    args = parser.parse_args()

    task = load_task(args.task)
    paper_text = load_paper(args.paper)

    runner, pairs = asyncio.run(generate_anchored(task, paper_text, args.steps))

    # Score the privi trajectory
    score = VerifierTool.score_trajectory(
        runner.to_dict(),
        task["queries"][0].get("true_hypothesis", "")
    )

    output = {
        "task_id": task["task_id"],
        "model": get_model(),
        "mode": "anchored",
        "anchor": "privi",
        "generation_mode": "agentic",
        "privi_trajectory": runner.to_dict(),
        "pairs": pairs,
        "num_pairs": len(pairs),
        "score": score.to_dict(),
    }

    save_json(output, args.output)
    print(f"\nAnchored pairs saved: {args.output}")
    print(f"Total pairs: {len(pairs)}")
    usable = [p for p in pairs if p["divergence_score"] > 0.2]
    print(f"Usable pairs (divergence > 0.2): {len(usable)}")
    print(f"Trajectory score: {score.total():.2f}")


if __name__ == "__main__":
    main()
