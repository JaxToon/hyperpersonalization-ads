import _bootstrap  # noqa: F401
from src.data_prep import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features from the Santander proxy dataset")
    parser.add_argument("--months", type=int, default=8)
    parser.add_argument("--sample-mod", type=int, default=5, help="keep 1 in N customers (1 = full dataset)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-month-offset", type=int, default=1)
    args = parser.parse_args()
    main(args.months, args.sample_mod, args.seed, args.eval_month_offset)
