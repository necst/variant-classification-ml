import argparse
import yaml
import sys
from pathlib import Path
import os


def main():
    # 1. Set up Argument Parsing
    parser = argparse.ArgumentParser(
        description="Extract model parameters from optimization results for Hydra."
    )
    parser.add_argument(
        "yaml_path", type=Path, help="Path to the optimization_results.yaml file"
    )
    args = parser.parse_args()

    # 2. Check if file exists
    if not args.yaml_path.exists():
        print(f"Error: File not found at {args.yaml_path}", file=sys.stderr)
        sys.exit(1)

    # 3. Load the YAML
    try:
        with open(args.yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error parsing YAML: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Extract and Format Arguments
    ax_params = data.get("ax", {})
    hydra_args = [f"model={ax_params['model']}"]
    del ax_params["model"]

    for key, value in ax_params.items():
        hydra_args.append(f"{key}={value}")
    
    hydra_args.append("training.save_model=true")
    command = "python run_conf.py " + " ".join(hydra_args)
    # 5. Print result to stdout (joined by spaces)
    print(f"Executing train command '{command}'")
    os.system(command)


if __name__ == "__main__":
    main()
