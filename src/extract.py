"""Extract task info from DiscoveryBench metadata into a structured JSON task file.

Usage:
    python src/extract.py --task biology_fish --output data/tasks/biology_fish.json
    python src/extract.py --task sociology_bmi --output data/tasks/sociology_bmi.json
    python src/extract.py --task economics_immigration --output data/tasks/economics_immigration.json
    python src/extract.py --list  # List all available metadata files
"""

import argparse
import json
import os
import glob

DISCOVERYBENCH_ROOT = os.path.join("data", "discoverybench", "discoverybench", "real", "train")

# Maps our POC aliases to DiscoveryBench folder names and preferred metadata index
TASK_ALIASES = {
    "biology_fish": {
        "folder": "evolution_freshwater_fish",
        "metadata_index": 0,
    },
    "sociology_bmi": {
        "folder": "nls_bmi",
        "metadata_index": 0,
    },
    "economics_immigration": {
        "folder": "immigration_offshoring_effect_on_employment",
        "metadata_index": 0,
    },
}


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_columns(raw_columns):
    """Convert DiscoveryBench column format to a flat dict."""
    result = {}
    for col in raw_columns:
        result[col["name"]] = col["description"]
    return result


def extract_task(metadata, folder_path, alias):
    """Extract a clean task object from DiscoveryBench metadata."""
    datasets_info = []
    for ds in metadata.get("datasets", []):
        ds_name = ds.get("name", "")
        columns = {}
        if ds.get("columns", {}).get("raw"):
            columns = extract_columns(ds["columns"]["raw"])

        ds_entry = {
            "name": ds_name,
            "description": ds.get("description", ""),
            "columns": columns,
            "csv_path": os.path.join(folder_path, ds_name),
        }
        datasets_info.append(ds_entry)

    # Queries are nested as [[{qid, question, true_hypothesis}, ...]]
    queries = []
    for query_group in metadata.get("queries", []):
        for q in query_group:
            queries.append({
                "qid": q.get("qid", 0),
                "question": q.get("question", ""),
                "question_type": q.get("question_type", ""),
                "true_hypothesis": q.get("true_hypothesis", ""),
            })

    return {
        "task_id": alias,
        "domain": metadata.get("domain", ""),
        "workflow_tags": metadata.get("workflow_tags", ""),
        "domain_knowledge": metadata.get("domain_knowledge", "") or None,
        "workflow": metadata.get("workflow", ""),
        "datasets": datasets_info,
        "queries": queries,
        "hypotheses": metadata.get("hypotheses", {}).get("main", []),
    }


def list_all_tasks(root):
    """List all available folders and their metadata files."""
    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        metas = sorted(glob.glob(os.path.join(folder_path, "metadata_*.json")))
        print(f"{folder}/")
        for m in metas:
            meta = load_metadata(m)
            domain = meta.get("domain", "?")
            n_queries = sum(len(qg) for qg in meta.get("queries", []))
            print(f"  {os.path.basename(m)}: domain={domain}, queries={n_queries}")


def main():
    parser = argparse.ArgumentParser(description="Extract task from DiscoveryBench")
    parser.add_argument("--task", choices=list(TASK_ALIASES.keys()),
                        help="Task alias (biology_fish, sociology_bmi, economics_immigration)")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--metadata-index", type=int, default=None,
                        help="Override which metadata_N.json to use (default: per task alias)")
    parser.add_argument("--list", action="store_true", help="List all available tasks")
    parser.add_argument("--discoverybench-root", default=DISCOVERYBENCH_ROOT)
    args = parser.parse_args()

    if args.list:
        list_all_tasks(args.discoverybench_root)
        return

    if not args.task or not args.output:
        parser.error("--task and --output are required (unless using --list)")

    alias_cfg = TASK_ALIASES[args.task]
    folder = alias_cfg["folder"]
    meta_idx = args.metadata_index if args.metadata_index is not None else alias_cfg["metadata_index"]

    folder_path = os.path.join(args.discoverybench_root, folder)
    if not os.path.isdir(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        print("Make sure you cloned DiscoveryBench:")
        print("  git clone https://github.com/allenai/discoverybench data/discoverybench")
        return

    meta_path = os.path.join(folder_path, f"metadata_{meta_idx}.json")
    if not os.path.isfile(meta_path):
        print(f"ERROR: Metadata file not found: {meta_path}")
        available = glob.glob(os.path.join(folder_path, "metadata_*.json"))
        print(f"Available: {[os.path.basename(p) for p in available]}")
        return

    print(f"Loading: {meta_path}")
    metadata = load_metadata(meta_path)
    task = extract_task(metadata, folder_path, args.task)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    print(f"Task extracted: {task['task_id']}")
    print(f"  Domain: {task['domain']}")
    print(f"  Datasets: {[d['name'] for d in task['datasets']]}")
    print(f"  Queries: {len(task['queries'])}")
    for q in task["queries"]:
        print(f"    Q{q['qid']}: {q['question'][:80]}...")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
