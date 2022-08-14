import numpy as np
from regex import regex
from varname import (
    argname,
)  # https://github.com/pwwang/python-varname pip install -U varname
from itertools import cycle, islice
import pandas as pd
from cprinter import TC  # pip install cprinter

len_array = np.frompyfunc(len, 1, 1)
regex_endvalue1 = regex.compile(r"""^\s*(['"])""")

regex_endvalue2 = regex.compile(r"""^\s*[\\'"]{0,3}""")
regex_endvalue3 = regex.compile(r"""[\\'"]{0,3}\s*$""")
regex_klammern = regex.compile(r"(\]\[|\[|\])")
nullend = regex.compile(r"\[0\]$")


def _get_colors(len_numpyarray):
    def repeatlist(it, count):
        return islice(cycle(it), count)

    allefarben = [
        "fg_yellow",
        "fg_red",
        # "fg_purple",
        "fg_pink",
        "fg_orange",
        "fg_lightred",
        # "fg_lightgrey",
        "fg_lightgreen",
        "fg_lightcyan",
        "fg_lightblue",
        # "fg_green",
        # "fg_cyan",
        # "fg_blue",
    ]
    neuefarben = list(repeatlist(allefarben, len_numpyarray * 2))
    return neuefarben


def getmaxlen(nparr):
    return str(nparr)


def allmax(nparr):
    return np.max(nparr)


getma = np.vectorize(getmaxlen)
getmaxone = np.vectorize(allmax)


def isiter(objectX):
    if isinstance(objectX, (np.ndarray, pd.core.frame.DataFrame, pd.core.frame.Series)):
        return True
    if isinstance(objectX, (str, bytes)):
        return False
    try:
        some_object_iterator = iter(objectX)
        return True
    except TypeError as te:
        return False


def getshape(arr):
    try:
        shapetouse = arr.shape[1]
    except:
        try:
            shapetouse = arr.shape[0]
        except:
            try:
                shapetouse = len(arr)
            except:
                shapetouse = len(str(arr))
    return shapetouse


def get_longest_item(item):
    if not isinstance(item, np.ndarray):
        try:
            item = np.array(item, dtype="object")
        except:
            item = transpose_list_of_lists(item)
            item = np.array(item, dtype="object")
            item = item.T

    try:
        item = np.array(
            np.array(
                [(x) if not isiter(x) else [xas for xas in x] for x in item],
            )
        )
    except:
        pass
    shapetouse = getshape(item)

    try:
        item = item.T.flatten().reshape((shapetouse, -1))
    except Exception as Fehler:
        # print(Fehler)
        pass
    strar = [[str(pp) for pp in x] for x in (item)]
    lenar = len_array(strar)
    malen = [np.max(x) for x in lenar]

    finalmax = []
    for zuzu in range(lenar.shape[1]):
        finalmax.extend(malen)
    return finalmax


def transpose_list_of_lists(listexxx):
    try:
        return [list(xaaa) for xaaa in zip(*listexxx)]
    except Exception as Fehler:
        print(Fehler)
        try:
            return np.array(listexxx).T.tolist()
        except Exception as Fehler:
            print(Fehler)
            return listexxx


def turnaround(item):
    if not isinstance(item, np.ndarray):
        item = np.asarray(item)

    item = item.T
    return item


def get_rightindexlist(arrayname, leveldeep, indent_, fixedcolsize=None, dimensions=1):
    indtoprint = (
        str(arrayname)
        + "["
        + "][".join([str(x) for x in leveldeep])
        + "]"
        + (len(str(arrayname) * 10) * " ")
    )
    if fixedcolsize is None:
        le = len(arrayname) + (dimensions * 15)

    else:
        le = fixedcolsize
    indtoprint_ajusted = indtoprint.rjust(le).ljust(le)
    return le, indtoprint_ajusted[:le]


def get_np_colors(subitem, intotal):
    try:
        npcolors = _get_colors(len(subitem) + (len(intotal) * 2))
    except Exception as fe:
        npcolors = _get_colors(len(intotal) * 2)
    return npcolors


def printdict(v, prefix="", repr_or_string="str"):
    results = {}
    maxlen_di = 0

    def aa_flatten_dict(v, prefix="", repr_or_string="str"):
        nonlocal results
        nonlocal maxlen_di
        if isinstance(v, dict):
            for k, v2 in v.items():
                if isinstance(k, (float, int)):
                    p2 = f"{prefix}[{k}]"
                else:
                    p2 = f'{prefix}["{k}"]'
                aa_flatten_dict(v2, p2, repr_or_string)
        elif isinstance(v, (list, tuple)):
            for i, v2 in enumerate(v):
                p2 = "{}[{}]".format(prefix, i)
                aa_flatten_dict(v2, p2, repr_or_string)
        else:
            if repr_or_string == "repr":
                endvalue = repr(v)
            else:
                endvalue = str(v)
            if isinstance(v, str):
                endvalue = regex_endvalue1.sub("\\" + r"\g<1>", endvalue)

                endvalue = regex_endvalue2.sub('"', endvalue)

                endvalue = regex_endvalue3.sub('"', endvalue)

            indtoprint = regex_klammern.split(prefix)
            npcolors = get_np_colors(indtoprint, indtoprint)
            lendict = len(prefix)
            if lendict > maxlen_di:
                maxlen_di = lendict
            if not prefix in results:
                results[prefix] = {}
                results[prefix]["prefo"] = prefix
                results[prefix]["len"] = len(prefix)
                results[prefix]["keys"] = []
                results[prefix]["value"] = TC(f"{endvalue}").fg_lightgrey.fg_black.bold

            for indi0, indii in enumerate(indtoprint):
                if any(regex_klammern.findall(indii)):
                    results[prefix]["keys"].append(TC(indii).bg_black.fg_yellow)
                    continue
                func = getattr(TC(f"{indii}"), npcolors[indi0])
                results[prefix]["keys"].append(func.bg_black)

    aa_flatten_dict(v, prefix=prefix, repr_or_string=repr_or_string)
    new_d = {}
    for k in sorted(results, key=len, reverse=False):
        new_d[k] = results[k]
    for k, v in new_d.items():
        for pref in v["keys"]:
            print(pref, end="")
        print(((maxlen_di + 4) - v["len"]) * " ", end="")
        print("= ", end="")
        print(v["value"], end="")
        print("", end="\n")


def get_blackblock():
    return TC("█").fg_black.bg_black


def get_lightgrey_block():
    return TC("█").fg_lightgrey.bg_black


def print_light_grey_block():
    print(get_lightgrey_block(), end="")


def get_purple_block():
    return TC("█").bg_lightgrey.fg_purple


def print_purple_block():
    print(get_purple_block(), end="")


def print_sep_block():
    print(get_sep_block(), end="")


def get_sep_block():
    return TC("»").fg_purple.bg_black


def numpyprinter(
    wholearray,
    printarr=None,
    indentlist=None,
    leveldeep=None,
    arrayname="",
    repr_or_string="str",
    max_col_width=0,
    withindent=True,
    isdf=False,
    fixedcolsize=None,
    dimensions=1,
):
    firstitem = False
    if leveldeep is None:
        firstitem = True
        leveldeep = [0]
    leveldeep_copy = leveldeep.copy()

    if indentlist is None:
        indentlist = [0]
    indent_ = sum(indentlist) * " "
    if not withindent:
        indent_ = ""

    indentlist.append(1)
    alreadyprinted = []
    intotal = []
    if isinstance(wholearray, (dict)):
        print("\n", end="")
        _, indtoprint = get_rightindexlist(
            arrayname, leveldeep[1:] + [0], indent_, fixedcolsize=None
        )
        indtoprint = indtoprint.strip()
        if firstitem:
            indtoprint = nullend.sub("", indtoprint)
        printdict(wholearray, prefix=indtoprint, repr_or_string=repr_or_string)
        return
    if isinstance(wholearray, (tuple, list)):
        wholearray = np.asarray(wholearray, dtype="object")
    if firstitem:
        dimensions = wholearray.ndim

    spacing = (
        "                                                                            "
    )

    if isiter(wholearray):
        npcolors = None
        try:
            longix = get_longest_item(wholearray)
        except:
            longix = get_longest_item(np.atleast_1d(wholearray))
        if printarr is not None:
            longix = printarr
        for indi0, subitem in enumerate(wholearray):

            longi = longix[indi0]
            if max_col_width != 0:
                if longi > max_col_width:
                    longi = max_col_width

            if not isiter(subitem):

                if npcolors is None:
                    npcolors = get_np_colors(wholearray, intotal)

                if repr_or_string == "repr":
                    itemtoprint = repr(subitem)
                else:
                    itemtoprint = str(subitem)
                if not withindent:
                    indent_ = ""
                itemtoprint = (
                    itemtoprint.replace("\n", " ").replace("\r", " ") + spacing
                )[:longi]
                if fixedcolsize is None:
                    fixedcolsize, indtoprint = get_rightindexlist(
                        arrayname,
                        leveldeep[1:],
                        indent_,
                        fixedcolsize=fixedcolsize,
                        dimensions=dimensions,
                    )
                else:
                    _, indtoprint = get_rightindexlist(
                        arrayname,
                        leveldeep[1:],
                        indent_,
                        fixedcolsize=fixedcolsize,
                        dimensions=dimensions,
                    )

                checkhash = hash(indtoprint)

                if not checkhash in alreadyprinted:
                    if fixedcolsize is None:
                        fixedcolsize, indtoprint = get_rightindexlist(
                            arrayname,
                            leveldeep[1:] + [indi0],
                            indent_,
                            fixedcolsize=fixedcolsize,
                            dimensions=dimensions,
                        )
                    else:
                        _, indtoprint = get_rightindexlist(
                            arrayname,
                            leveldeep[1:] + [indi0],
                            indent_,
                            fixedcolsize=fixedcolsize,
                            dimensions=dimensions,
                        )
                    row_ = f"{indent_}{indtoprint}{get_sep_block()}{get_blackblock()}{get_blackblock()}"
                    row2 = f"{itemtoprint.ljust(longi).rjust(longi)}  "
                    print(TC(row_).bg_black.fg_darkgrey, end="")
                    print(TC(row2).bg_black.fg_yellow, end="")

                    alreadyprinted.append(checkhash)
                    intotal = [1]
                    print_purple_block()
                else:
                    if fixedcolsize is None:

                        fixedcolsize, indtoprint = get_rightindexlist(
                            arrayname,
                            leveldeep[1:] + [len(intotal)],
                            indent_,
                            fixedcolsize=fixedcolsize,
                            dimensions=dimensions,
                        )
                    else:
                        _, indtoprint = get_rightindexlist(
                            arrayname,
                            leveldeep[1:] + [len(intotal)],
                            indent_,
                            fixedcolsize=fixedcolsize,
                            dimensions=dimensions,
                        )
                    row_ = f"{indent_}{indtoprint}{get_sep_block()}{get_blackblock()}{get_blackblock()}"
                    row2 = f"{itemtoprint.ljust(longi).rjust(longi)}  "

                    print(TC(row_).fg_darkgrey.bg_black, end="")

                    func = getattr(TC(f"{row2}"), npcolors[indi0])
                    print(func.bg_black, end="")

                    print_purple_block()
                    intotal.append(1)

            else:
                intotal.append(1)

                leveldeepold = leveldeep.copy()
                leveldeep = leveldeep + [indi0]

                if indi0 == 0:
                    indtoprint = "[0]"
                    if any(leveldeepold):
                        indtoprint = (
                            "[" + "][".join([str(x) for x in leveldeep[:]]) + "]"
                        )
                    if withindent:
                        indtoprint = f"{indent_}Start of: {arrayname}{indtoprint}"
                        print(TC(indtoprint).fg_black.bg_cyan, end="")
                        print("", end="\n")
                if not isinstance(subitem, dict):
                    if isinstance(subitem, tuple):
                        leerzeile = numpyprinter(
                            subitem,
                            printarr=longix[: len(subitem)],
                            indentlist=indentlist,
                            leveldeep=leveldeep,
                            arrayname=arrayname,
                            withindent=withindent,
                            repr_or_string=repr_or_string,
                            isdf=isdf,
                            max_col_width=max_col_width,
                            fixedcolsize=fixedcolsize,
                        )
                    else:
                        leerzeile = numpyprinter(
                            subitem.copy(),
                            printarr=longix[: len(subitem)],
                            indentlist=indentlist,
                            leveldeep=leveldeep,
                            arrayname=arrayname,
                            withindent=withindent,
                            repr_or_string=repr_or_string,
                            isdf=isdf,
                            max_col_width=max_col_width,
                            fixedcolsize=fixedcolsize,
                        )

                else:
                    leerzeile = numpyprinter(
                        subitem,
                        printarr=longix[: len(subitem)],
                        indentlist=indentlist,
                        leveldeep=leveldeep_copy,
                        arrayname=arrayname,
                        withindent=withindent,
                        repr_or_string=repr_or_string,
                        isdf=isdf,
                        max_col_width=max_col_width,
                        fixedcolsize=fixedcolsize,
                    )

                if leerzeile == " ":
                    indentlist = indentlist[:-1]
                    leveldeep = leveldeep[:-1]

        print("\n", end="")
        return " "
    else:
        print(f"{wholearray=}")
    print("\n", end="")
    return "  "


def pdp(dframe, max_column_size=50, repeat_cols=0, printasnp=False):
    try:
        arrayname = argname("dframe")
    except:
        arrayname = "a"
    if printasnp:
        if isinstance(dframe, (pd.core.frame.DataFrame, pd.core.frame.Series)):
            dframe = dframe.values
            arrayname = f"{arrayname}.iloc"
        numpyprinter(
            dframe, max_col_width=max_column_size, arrayname=arrayname, withindent=False
        )
        return

    if isinstance(dframe, pd.core.frame.Series):
        df = dframe.to_frame()
    else:
        df = dframe
    if not isinstance(df, (pd.core.frame.DataFrame, pd.core.frame.Series)):
        numpyprinter(df, max_col_width=max_column_size, arrayname=arrayname)
        return
    if len(list(dict.fromkeys(df.columns.to_list()))) != len(df.columns.to_list()):
        numpyprinter(
            df.values,
            max_col_width=max_column_size,
            arrayname=f"{arrayname}.iloc",
            withindent=False,
        )
        return

    def _print_cols():
        nonlocal columnsprint

        for indiprint, labels in enumerate(columnsprint):
            if indiprint != len(columnsprint):
                print(TC(f"{labels}  ").bg_red.fg_black, end="")
                print(TC("█").fg_yellow.bg_black, end="")
            if indiprint + 1 == len(columnsprint):
                pass
                print("", end="\n")

    maxsize = [
        df[x].astype("string").array.astype("U").dtype.itemsize // 4 for x in df.columns
    ]
    maxsize_cols = [len(str(x)) for x in df.columns]
    maxsize = [x if x >= y else y for x, y in zip(maxsize, maxsize_cols)]

    maxsize_ = [x + 2 if x < max_column_size else max_column_size for x in maxsize]
    maxindexsize = 4 + (df.index.astype("string").array.astype("U").dtype.itemsize // 4)
    if maxindexsize < 8:
        maxindexsize = 9

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


if __name__ == "__main__":
    print("Testing")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
    )
    df = df[:30]
    print("Regular Dataframe")
    pdp(df, max_column_size=75, repeat_cols=20)
    print("Dataframe as Numpy")
    pdp(df, max_column_size=75, repeat_cols=20, printasnp=True)
    print("Transposed DF as Numpy")
    dftr = df.T
    pdp(dftr, max_column_size=75, repeat_cols=20)
    print("values (pandas)")
    dfvals = df.values
    pdp(dfvals, max_column_size=75, repeat_cols=20)
    print("array np (pandas)")
    dfvarr = df.__array__()
    pdp(dfvarr, max_column_size=75, repeat_cols=20)
    print("dict")
    dfdict = df.to_dict()
    pdp(dfdict, max_column_size=75, repeat_cols=20)
    print("records from df (tuple/list)")
    dfrec = df.to_records()
    pdp(dfrec, max_column_size=75, repeat_cols=20)
    dfrecl = df.to_records().tolist()
    pdp(dfrecl, max_column_size=75, repeat_cols=20)
    dfrect = tuple(df.to_records().tolist())
    pdp(dfrect, max_column_size=25, repeat_cols=20)
    print("pd to numpy")
    dfnp = df.to_numpy()
    pdp(dfnp, max_column_size=25, repeat_cols=20)
