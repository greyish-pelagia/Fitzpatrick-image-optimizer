import argparse

from fitzpatrick_optimizer.demo import create_demo_dataset
from fitzpatrick_optimizer.evaluate import main as evaluate_main
from fitzpatrick_optimizer.infer import main as infer_main
from fitzpatrick_optimizer.train import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fitzpatrick image optimizer portfolio demo"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train a model")
    subparsers.add_parser("evaluate", help="Evaluate a model")
    subparsers.add_parser("infer", help="Run inference")
    demo_parser = subparsers.add_parser(
        "create-demo-data",
        help="Create generated demo images",
    )
    demo_parser.add_argument("--output_dir", default="demo_assets")
    demo_parser.add_argument("--count", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command == "train":
        train_main(remaining)
    elif args.command == "evaluate":
        evaluate_main(remaining)
    elif args.command == "infer":
        infer_main(remaining)
    elif args.command == "create-demo-data":
        csv_path = create_demo_dataset(args.output_dir, args.count)
        print(f"Created demo dataset: {csv_path}")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
