"""
Run evaluations on the development set.
Usage: python scripts/run_eval.py --split dev --versions v0_baseline v1_persona_rag
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--versions", nargs="+", required=True)
    args = parser.parse_args()
    
    logger.info(f"Running eval on {args.split} for {args.versions}")
    from policypilot.eval.harness import run_evaluation
    import json
    
    for version in args.versions:
        # For demo purposes, we will try to run on rag_dev.jsonl
        # If it doesn't exist, the harness will just log an error and return empty
        records, metrics = run_evaluation(f"rag_{args.split}.jsonl", version)
        print(f"--- Metrics for {version} ---")
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
