'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import valmetrics as vm

def main() -> None:
    print(80*"=")
    print("MAVM Calc for DIC vs Image Def Data")
    print(80*"=")

    sim_dir = Path.cwd()/ "Pulse38_ProbSim_ImageDef"
    exp_dir = Path.cwd() / "Pulse38_Exp_DIC"

    if not sim_dir.is_dir():
        raise FileNotFoundError(f"{sim_dir}: directory does not exist.")
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"{exp_dir}: directory does not exist.")



if __name__ == "__main__":
    main()