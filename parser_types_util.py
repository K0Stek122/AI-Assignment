from pathlib import Path
import argparse

def type_csv(value):
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError("Provided CSV does not exist.")
    if not path.is_file():
        raise argparse.ArgumentTypeError("Provided CSV is not a file.")
    if path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("Provided CSV is not a CSV file.")
    return path

def type_csv_nonreal(value):
    path = Path(value)
    if not path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("Provided CSV is not a CSV file.")
    return path
    
def type_keras(value):
    path = Path(value)
    if not path.parent.exists():
        raise argparse.ArgumentTypeError("Provided output path does not exist.")
    if path.suffix.lower() != ".keras":
        raise argparse.ArgumentTypeError("Output must end in .keras")
    return path

def type_keras_nonreal(value):
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError("Provided .keras must exist.")
    if path.suffix.lower() != ".keras":
        raise argparse.ArgumentTypeError("Argument must end in .keras")