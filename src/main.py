import os

from src.train import train
from src.web_service import launch
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run training or launch web service.")
    parser.add_argument("--train", action="store_true", help="Run in training mode")
    args = parser.parse_args()

    if args.train:
        print("Training the model...")
        train()
    else:
        print("Starting the web service...")
        launch()


if __name__ == "__main__":
    main()
