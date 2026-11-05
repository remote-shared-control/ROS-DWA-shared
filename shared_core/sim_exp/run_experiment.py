#!/usr/bin/env python3
"""Select a sim_exp controller without rosrun's duplicate-basename ambiguity."""

import argparse
import os
from pathlib import Path
import sys


def main():
    directory = Path(__file__).resolve().parent
    choices = sorted([path.name for path in directory.glob("dwa*.py")] + ["direct.py"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=choices,
                        default="dwa_simwaffi_new_midpoint_hybrid.py")
    args, ros_args = parser.parse_known_args()
    os.execv(sys.executable, [sys.executable, str(directory / args.controller), *ros_args])


if __name__ == "__main__":
    main()
