"""
Run parameter sweep.
Usage: python scripts/run_sweep.py
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Running parameter sweeps from configs/sweeps.yaml")
    logger.info("Sweep stub - requires loaded datasets to execute.")

if __name__ == "__main__":
    main()
