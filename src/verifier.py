"""Single authority for scoring trajectory decisions (like SREG's VerifierTool).

All scoring logic lives here. Called by:
- generate_anchored.py (pair quality assessment)
- training/rubric.py (reward computation)
- format_eval.py (filtering low-quality pairs)
"""


class Score:
    """Result of scoring a trajectory or decision."""

    def __init__(self, hypothesis_score=0.0, methodology_score=0.0,
                 efficiency=0.0, per_step=None):
        self.hypothesis_score = hypothesis_score
        self.methodology_score = methodology_score
        self.efficiency = efficiency
        self.per_step = per_step or []

    def total(self):
        return (self.hypothesis_score + self.methodology_score) / 2

    def to_dict(self):
        return {
            "hypothesis_score": self.hypothesis_score,
            "methodology_score": self.methodology_score,
            "efficiency": self.efficiency,
            "per_step": self.per_step,
            "total": self.total(),
        }


class VerifierTool:
    """Single authority for scoring trajectory decisions."""

    @staticmethod
    def score_hypothesis_exact(agent_hypothesis, gold_hypothesis):
        """Simple keyword-based hypothesis match. Returns 0.0-1.0.

        Checks if key terms from gold hypothesis appear in agent's answer.
        For POC — replace with LLM judge for production.
        """
        if not gold_hypothesis or not agent_hypothesis:
            return 0.0

        gold_lower = gold_hypothesis.lower()
        agent_lower = agent_hypothesis.lower()

        # Extract key numeric values from gold
        import re
        gold_numbers = set(re.findall(r'\d+\.?\d*', gold_lower))
        agent_numbers = set(re.findall(r'\d+\.?\d*', agent_lower))
        number_overlap = len(gold_numbers & agent_numbers) / max(len(gold_numbers), 1)

        # Extract key terms (words > 4 chars, not stopwords)
        stopwords = {"that", "this", "with", "from", "have", "been", "were", "which", "their", "about", "would", "there", "between"}
        gold_terms = {w for w in gold_lower.split() if len(w) > 4 and w not in stopwords}
        agent_terms = {w for w in agent_lower.split() if len(w) > 4 and w not in stopwords}
        term_overlap = len(gold_terms & agent_terms) / max(len(gold_terms), 1)

        return 0.5 * number_overlap + 0.5 * term_overlap

    @staticmethod
    def score_step_has_code(step):
        """Does the step have executable code? Returns 0.0 or 1.0."""
        return 1.0 if step.get("code", "").strip() else 0.0

    @staticmethod
    def score_step_executed(step):
        """Did the step's code execute successfully? Returns 0.0 or 1.0."""
        er = step.get("execution_result", {})
        return 1.0 if er.get("exit_code", -1) == 0 else 0.0

    @staticmethod
    def score_pair_divergence(chosen, rejected):
        """How different are the chosen and rejected decisions? Returns 0.0-1.0.

        Simple heuristic: different action_type = high divergence,
        same action_type but different action = medium divergence.
        """
        if chosen.get("action_type") != rejected.get("action_type"):
            return 1.0

        # Same action type — check action text similarity
        chosen_words = set(chosen.get("action", "").lower().split())
        rejected_words = set(rejected.get("action", "").lower().split())
        if not chosen_words or not rejected_words:
            return 0.5
        overlap = len(chosen_words & rejected_words) / max(len(chosen_words | rejected_words), 1)
        return 1.0 - overlap  # higher divergence = less overlap

    @staticmethod
    def score_trajectory(trajectory, gold_hypothesis):
        """Score a full trajectory. Returns Score object."""
        steps = trajectory.get("steps", [])
        final = trajectory.get("final_hypothesis", "")

        hypothesis_score = VerifierTool.score_hypothesis_exact(final, gold_hypothesis)

        # Methodology: fraction of steps that executed successfully
        executed = sum(1 for s in steps if s.get("execution_result", {}).get("exit_code", -1) == 0)
        methodology_score = executed / max(len(steps), 1)

        # Efficiency: fewer steps to a good answer = better
        efficiency = hypothesis_score / max(len(steps), 1)

        per_step = []
        for s in steps:
            per_step.append({
                "step": s.get("step_number"),
                "has_code": VerifierTool.score_step_has_code(s),
                "executed": VerifierTool.score_step_executed(s),
            })

        return Score(
            hypothesis_score=hypothesis_score,
            methodology_score=methodology_score,
            efficiency=efficiency,
            per_step=per_step,
        )
