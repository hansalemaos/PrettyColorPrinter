from itertools import cycle, islice
import pandas as pd
from cprinter import TC  # pip install cprinter


def pdp(df, max_column_size=50, repeat_cols=0):
    def _print_cols():
        nonlocal columnsprint
        print("", end="\n")

        for indiprint, labels in enumerate(columnsprint):
            if indiprint != len(columnsprint):
                print(TC(f"{labels}  ").bg_red.fg_black, end="")
                print(TC("█").fg_yellow.bg_black, end="")
            if indiprint + 1 == len(columnsprint):
                print("", end="\n")

    def _get_colors(len_numpyarray):
        def repeatlist(it, count):
            return islice(cycle(it), count)

        allefarben = [
            "fg_yellow",
            "fg_red",
            "fg_purple",
            "fg_pink",
            "fg_orange",
            "fg_lightred",
            "fg_lightgrey",
            "fg_lightgreen",
            "fg_lightcyan",
            "fg_lightblue",
            "fg_green",
            "fg_cyan",
            "fg_blue",
        ]
        neuefarben = list(repeatlist(allefarben, len_numpyarray * 2))
        return neuefarben

    maxsize = [
        df[x].astype("string").array.astype("U").dtype.itemsize // 4 for x in df.columns
    ]
    maxsize_cols = [len(x) for x in df.columns]
    maxsize = [x if x >= y else y for x, y in zip(maxsize, maxsize_cols)]

    maxsize_ = [x + 2 if x < max_column_size else max_column_size for x in maxsize]
    maxindexsize = df.index.astype("string").array.astype("U").dtype.itemsize // 4
    if maxindexsize < 7:
        maxindexsize = 7

    columnsprint = [
        str(f"{col}").replace("\n", " ").replace("\r", " ")[:size].rjust(size + 1)
        for col, size in zip(df.columns, maxsize_)
    ]
    columnsprint.insert(0, "index".ljust(maxindexsize + 1))
    _print_cols()
    colprintcounter = 0
    npcolors = _get_colors(len(maxsize))

    for indi, row in zip(df.index, df.__array__()):

        emptyline = 0
        indi_ = (
            str(f" {indi}")
            .replace("\n", " ")
            .replace("\r", " ")[:maxindexsize]
            .ljust(maxindexsize + 1)
        )
        colprintcounter = colprintcounter + 1
        if repeat_cols != 0:
            if repeat_cols <= colprintcounter:
                _print_cols()
                colprintcounter = 0
        print(TC(f"{indi_}  ").fg_black.bg_cyan, end="")
        print(TC("█").fg_blue.bg_black, end="")

        for r, size, color in zip(row, maxsize_, npcolors):
            row_ = str(f"  {r}").replace("\n", " ").replace("\r", " ") + " " * size
            row_ = row_[:size].rjust(size + 1).ljust(size + 1)
            func = getattr(TC(f"{row_}  "), color)
            emptyline = emptyline + 1
            print(func.bg_black, end="")
            print(TC("█").fg_yellow.bg_black, end="")

        print("", end="\n")
