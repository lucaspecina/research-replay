"""Trajectory state manager (like SREG's EpisodeRunner).

Manages the full state of a research trajectory:
- Persistent Python namespace for code execution
- Step history with execution results
- LLM conversation messages
- State snapshots for anchored mode
"""

import copy

from python_exec import make_namespace, python_exec
from llm import extract_json, call_messages, get_model
from common import load_prompt, build_dataset_summary, build_df_description


class StepResult:
    """Immutable record of one trajectory step."""

    def __init__(self, step_number, action_type, reasoning, action, code,
                 expected_outcome, execution_result, actual_outcome):
        self.step_number = step_number
        self.action_type = action_type
        self.reasoning = reasoning
        self.action = action
        self.code = code
        self.expected_outcome = expected_outcome
        self.execution_result = execution_result
        self.actual_outcome = actual_outcome

    def to_dict(self):
        return {
            "step_number": self.step_number,
            "action_type": self.action_type,
            "reasoning": self.reasoning,
            "action": self.action,
            "code": self.code,
            "expected_outcome": self.expected_outcome,
            "execution_result": self.execution_result,
            "actual_outcome": self.actual_outcome,
        }


class TrajectoryRunner:
    """Manages trajectory state, code execution, and step tracking."""

    def __init__(self, task, paper_text=None, system_prompt=None, num_steps=6):
        self.task = task
        self.paper_text = paper_text
        self.system_prompt = system_prompt or ""
        self.num_steps = num_steps
        self.namespace = make_namespace(task)
        self.steps = []
        self.messages = []
        self.snapshots = {}  # step_number -> messages snapshot (before LLM call)
        self.final_hypothesis = ""
        self.finished = False

        self._init_messages()

    def _init_messages(self):
        """Build initial system + context messages."""
        query = self.task["queries"][0]
        domain_knowledge = self.task.get("domain_knowledge") or f"Domain: {self.task['domain']}"
        dataset_summary = build_dataset_summary(self.task)

        context = f"""## Research Question
{query['question']}

## Domain Context
{domain_knowledge}

## Available Data
{dataset_summary}
"""
        if self.paper_text:
            context += f"\n## Published Paper (PRIVILEGED INFORMATION)\n{self.paper_text}\n"

        context += "\nI will now ask you to generate steps one at a time. For each step, propose an analysis AND write executable Python code. I will run your code on the real data and show you the results."

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context},
        ]

    def _build_step_prompt(self):
        """Build the user prompt for the next step."""
        step_num = len(self.steps) + 1
        df_desc = build_df_description(self.task)
        history = self._format_history()

        return f"""You are on step {step_num} of your investigation.

## Available DataFrames (pre-loaded)
{df_desc}

## Previous Steps and Results
{history}

## Instructions
Propose the next research step. Return ONLY a JSON object:

```json
{{
  "step_number": {step_num},
  "action_type": "explore|analyze|interpret|decide|verify",
  "reasoning": "Why this step makes sense given what you've seen so far...",
  "action": "Natural language description of what to do...",
  "code": "Python code that performs the analysis. Print results to stdout.",
  "expected_outcome": "What you expect to find before running the code..."
}}
```

IMPORTANT:
- Your code will be executed on REAL data. You will see real results.
- Available: pandas (pd), numpy (np), scipy.stats (stats), sklearn, statsmodels (sm).
- DataFrames are already loaded — do NOT load them yourself.
- Variables from previous steps persist — you can reuse them.
- Always print() results. Keep output concise.
- Do NOT produce plots. Describe findings via print() instead."""

    def _format_history(self):
        """Format completed steps + execution results."""
        if not self.steps:
            return "(No steps yet — this is the first step.)"
        parts = []
        for s in self.steps:
            part = f"### Step {s.step_number} [{s.action_type}]\n"
            part += f"**Reasoning:** {s.reasoning}\n"
            part += f"**Action:** {s.action}\n"
            part += f"**Expected:** {s.expected_outcome}\n"
            er = s.execution_result
            if er.get("exit_code", 0) == 0 and er.get("stdout"):
                part += f"**Result (real data):**\n```\n{er['stdout']}\n```"
            elif er.get("stderr"):
                part += f"**Error:**\n```\n{er['stderr']}\n```"
            else:
                part += f"**Result:** {s.actual_outcome or '(no output)'}"
            parts.append(part)
        return "\n\n".join(parts)

    def save_snapshot(self):
        """Save current messages state for anchored mode (before LLM call)."""
        step_num = len(self.steps) + 1
        self.snapshots[step_num] = [dict(m) for m in self.messages]

    def get_snapshot(self, step_number):
        """Get messages snapshot from before a specific step."""
        return self.snapshots.get(step_number, [])

    async def run_step(self):
        """Run one step: prompt LLM -> parse -> execute code -> record."""
        step_num = len(self.steps) + 1
        step_prompt = self._build_step_prompt()
        self.messages.append({"role": "user", "content": step_prompt})

        # Save snapshot before LLM call (for anchored mode)
        self.save_snapshot()

        # Call LLM
        text = await call_messages(self.messages, max_tokens=4096)
        self.messages.append({"role": "assistant", "content": text})

        # Parse
        parsed = extract_json(text)
        if parsed is None:
            print(f"    WARNING: could not parse JSON for step {step_num}")
            return None

        # Execute code
        code = parsed.get("code", "")
        if code:
            er = python_exec(code, self.namespace)
        else:
            er = {"stdout": "", "stderr": "", "exit_code": 0, "truncated": False}

        actual_outcome = er["stdout"].strip() if er["exit_code"] == 0 else f"ERROR: {er['stderr'][:500]}"

        step = StepResult(
            step_number=step_num,
            action_type=parsed.get("action_type", "explore"),
            reasoning=parsed.get("reasoning", ""),
            action=parsed.get("action", ""),
            code=code,
            expected_outcome=parsed.get("expected_outcome", ""),
            execution_result=er,
            actual_outcome=actual_outcome,
        )
        self.steps.append(step)

        # Feed execution result back to conversation
        if code:
            if er["exit_code"] == 0:
                result_msg = f"## Execution Result for Step {step_num}\n```\n{er['stdout']}\n```"
            else:
                result_msg = f"## Execution Result for Step {step_num}\nError (exit code {er['exit_code']}):\n```\n{er['stderr']}\n```"
            self.messages.append({"role": "user", "content": result_msg})

        return step

    async def run_all_steps(self):
        """Run all steps sequentially."""
        model = get_model()
        consecutive_errors = 0

        for i in range(self.num_steps):
            print(f"  Step {i + 1}/{self.num_steps}...")
            step = await self.run_step()

            if step is None:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    print("    ABORT: 3 consecutive parse errors")
                    break
                continue

            if step.execution_result.get("exit_code", 0) == 0:
                consecutive_errors = 0
                print(f"    OK ({len(step.execution_result.get('stdout', ''))} chars)")
            else:
                consecutive_errors += 1
                print(f"    ERROR: {step.execution_result.get('stderr', '')[:200]}")

            if consecutive_errors >= 3:
                print("    ABORT: 3 consecutive errors")
                break

    async def run_final_hypothesis(self):
        """Ask for final hypothesis after all steps."""
        print("  Generating final hypothesis...")
        self.messages.append({
            "role": "user",
            "content": 'Based on all the analyses you\'ve run and their real results, state your final hypothesis answering the research question. Return ONLY a JSON object: {"final_hypothesis": "..."}'
        })
        text = await call_messages(self.messages, max_tokens=1024)
        result = extract_json(text)
        self.final_hypothesis = result.get("final_hypothesis", "") if result else ""
        self.finished = True

    def to_dict(self):
        """Export full trajectory as dict."""
        return {
            "task_id": self.task["task_id"],
            "model": get_model(),
            "paper_in_context": self.paper_text is not None,
            "steps": [s.to_dict() for s in self.steps],
            "final_hypothesis": self.final_hypothesis,
            "gold_hypothesis": self.task["queries"][0].get("true_hypothesis", ""),
        }
