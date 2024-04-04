import argparse
from process import process_data
from build import build_model
from optimize import optimize


def main(args):
    file_dir = args.file_dir
    name = file_dir[:file_dir.find(".")]

    # Process original data and save processed data
    process_data(name, args.missing_rate)

    # Train the model
    build_model(name, args.missing_rate, args.train_epochs, args.train_batch)

    # Customized optimization
    optimize(name, args.missing_rate, args.opt_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-dir",
        type=str,
        help="Path for the raw data (should be saved in 'data' folder)",
    )
    parser.add_argument(
        "--missing-rate",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--train-epochs",
        default=250,
        type=int,
        help="number of epochs for training the generator",
    )
    parser.add_argument(
        "--train-batch",
        default=250,
        help="batch size for training the generator"
    )
    parser.add_argument(
        "--opt-epochs",
        default=250,
        type=int,
        help="number of epochs for optimization",
    )

    args = parser.parse_args()

    main(args)
