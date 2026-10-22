"""Write the calibration traffic: prefill.jsonl and decode.jsonl, in gpumix traffic format.

python scripts/make_sweeps.py results/sweeps
"""

import argparse
from pathlib import Path

from gpumix.calibrate import decode_sweep, flatten, prefill_sweep
from gpumix.traffic import write_traffic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, probes in (("prefill", prefill_sweep()), ("decode", decode_sweep())):
        requests = flatten(probes)
        write_traffic(args.out_dir / f"{name}.jsonl", requests)
        print(f"wrote {len(requests)} requests to {args.out_dir / name}.jsonl")


if __name__ == "__main__":
    main()
