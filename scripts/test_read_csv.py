from pathlib import Path
import numpy as np
import pandas as pd

def main() -> None:
    file_broken = Path.cwd()/"STC_Exp_DIC_211"/"Image_0699_0.tiff.csv"
    data_broken = pd.read_csv(file_broken)
    data_broken = data_broken.to_numpy()

    file_correct = Path.cwd()/"STC_Exp_DIC_211"/"Image_0600_0.tiff.csv"
    data_correct = pd.read_csv(file_correct)
    data_correct = data_correct.to_numpy()

    print()
    print(f"{data_broken.shape=}")
    print(f"{data_correct.shape=}")
    print()

if __name__ == "__main__":
    main()
