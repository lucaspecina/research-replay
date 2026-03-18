"""Stub for verifiers.StatefulToolEnv integration (like SREG's SregEnv).

This will wrap TrajectoryRunner for RL training when we're ready.
NOT IMPLEMENTED YET — just the structure.

Future usage:
    from training.env import ResearchEnv
    env = ResearchEnv(task, paper_text)
    # Use with verifiers library for DPO/GRPO training
"""

# TODO: Implement when ready for training
# from verifiers import StatefulToolEnv
# from trajectory_runner import TrajectoryRunner
# from training.rubric import score_submission


class ResearchEnv:
    """Stub — will implement verifiers.StatefulToolEnv when ready.

    The environment will:
    1. Initialize a TrajectoryRunner with task + data
    2. Provide tools: python_exec, propose_step, submit
    3. Compute terminal reward via rubric.score_submission
    4. Support both DPO and GRPO training loops
    """

    def __init__(self, task, paper_text=None):
        self.task = task
        self.paper_text = paper_text
        raise NotImplementedError(
            "ResearchEnv is a stub. Install verifiers library and "
            "implement StatefulToolEnv when ready for training."
        )
