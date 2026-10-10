"""
Estimate costs for a full evaluation run.
Usage: python scripts/estimate_cost.py
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Estimating full evaluation run costs.")
    
    # Assume we run dev + test set for 6 prompt versions
    # Dev set size ~200, test set size ~200. Total ~400 questions per version.
    # Total versions = 6 (v0, v1, v2, v3, v4, v5, v6) -> 2400 queries
    
    avg_prompt_tokens = 1200
    avg_completion_tokens = 250
    
    total_prompt_tokens = 2400 * avg_prompt_tokens
    total_comp_tokens = 2400 * avg_completion_tokens
    
    # Example pricing (GPT-4o mini or similar)
    # prompt: $0.150 / 1M, comp: $0.600 / 1M
    cost_prompt = (total_prompt_tokens / 1_000_000) * 0.15
    cost_comp = (total_comp_tokens / 1_000_000) * 0.60
    
    logger.info(f"Total prompt tokens: {total_prompt_tokens:,}")
    logger.info(f"Total completion tokens: {total_comp_tokens:,}")
    logger.info(f"Estimated prompt cost: ${cost_prompt:.2f}")
    logger.info(f"Estimated completion cost: ${cost_comp:.2f}")
    logger.info(f"Total estimated cost for full benchmark suite: ${cost_prompt + cost_comp:.2f}")

if __name__ == "__main__":
    main()
