"""Persistent Python namespace for code execution (like SREG's python_exec).

Instead of subprocess-per-step, code runs in a persistent namespace.
Variables defined in step 1 are available in step 3. Like a Jupyter notebook.
"""

import io
import os
import contextlib
import traceback

import numpy as np
import pandas as pd
import scipy.stats

MAX_OUTPUT = 3000

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Builtins to block (security)
_BLOCKED = {"exec", "eval", "compile", "open", "input", "breakpoint"}

# Allowed imports (safe libraries for data analysis)
_ALLOWED_IMPORTS = {
    "pandas", "numpy", "scipy", "sklearn", "statsmodels",
    "math", "statistics", "json", "collections", "itertools",
    "functools", "re", "warnings", "copy",
}


def _safe_import(name, *args, **kwargs):
    """Allow importing only whitelisted modules."""
    top_level = name.split(".")[0]
    if top_level not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed. Allowed: {sorted(_ALLOWED_IMPORTS)}")
    return __builtins_original_import__(name, *args, **kwargs)


# Keep original import for internal use
import builtins as _builtins_module
__builtins_original_import__ = _builtins_module.__import__


def _make_safe_builtins():
    safe = {k: v for k, v in vars(_builtins_module).items() if k not in _BLOCKED}
    safe["__import__"] = _safe_import
    return safe


def make_namespace(task):
    """Create a persistent namespace with datasets pre-loaded."""
    ns = {
        "__builtins__": _make_safe_builtins(),
        "pd": pd,
        "np": np,
        "scipy": scipy,
        "stats": scipy.stats,
    }

    # Try importing optional libraries
    try:
        import sklearn
        ns["sklearn"] = sklearn
    except ImportError:
        pass

    try:
        import statsmodels.api as sm
        ns["sm"] = sm
    except ImportError:
        pass

    # Load datasets
    datasets = task.get("datasets", [])
    for i, ds in enumerate(datasets):
        raw_path = ds.get("csv_path", "")
        abs_path = os.path.join(PROJECT_ROOT, raw_path)
        var_name = "df" if len(datasets) == 1 else f"df_{i + 1}"

        try:
            if raw_path.endswith(".dta"):
                ns[var_name] = pd.read_stata(abs_path)
            else:
                ns[var_name] = pd.read_csv(abs_path)
        except Exception as e:
            ns[var_name] = None
            print(f"Warning: could not load {ds['name']}: {e}")

    # Set pandas display options in namespace
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    return ns


def sanitize_code(code):
    """Fix common LLM code issues."""
    if "\\n" in code and "\n" not in code.replace("\\n", ""):
        code = code.replace("\\n", "\n")
    lines = code.split("\n")
    lines = [
        l for l in lines
        if not l.strip().startswith(("import pandas", "import numpy", "import warnings"))
    ]
    return "\n".join(lines)


def python_exec(code, namespace, timeout=30):
    """Execute code in persistent namespace. Returns dict with stdout/stderr/exit_code.

    The namespace persists between calls — variables defined in one call
    are available in the next. Like a Jupyter notebook.
    """
    code = sanitize_code(code)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, namespace)

        stdout = stdout_buf.getvalue()
        truncated = len(stdout) > MAX_OUTPUT
        if truncated:
            stdout = stdout[:MAX_OUTPUT] + "\n[TRUNCATED]"

        return {
            "stdout": stdout,
            "stderr": stderr_buf.getvalue(),
            "exit_code": 0,
            "truncated": truncated,
        }
    except Exception:
        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": traceback.format_exc(),
            "exit_code": 1,
            "truncated": False,
        }
