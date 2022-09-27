from typing import Union, Any
from pandas.core.base import PandasObject
from collections import defaultdict
import numpy as np
from regex import regex
from itertools import cycle, islice
import pandas as pd
from cprinter import TC  # pip install cprinter
from input_timeout import InputTimeout


len_array = np.frompyfunc(len, 1, 1)
regex_endvalue1 = regex.compile(r"""^\W*(['"])""")
regex_endvalue2 = regex.compile(r"""^\s*[\\'"]{0,3}""")
regex_endvalue3 = regex.compile(r"""[\\'"]+\W*$""")
regex_klammern = regex.compile(r"(\]\[|\[|\])")
nullend = regex.compile(r"\[0\]$")
nested_dict = lambda: defaultdict(nested_dict)


def _get_colors(len_numpyarray):
    def repeatlist(it, count):
        return islice(cycle(it), count)

    allefarben = [
        "fg_yellow",
        "fg_red",
        "fg_pink",
        "fg_orange",
        "fg_lightred",
        "fg_lightgreen",
        "fg_lightcyan",
        "fg_lightblue",
    ]
    neuefarben = list(repeatlist(allefarben, len_numpyarray * 2))
    return neuefarben


def isiter(objectX):
    if isinstance(objectX, (np.ndarray, pd.DataFrame, pd.Series)):
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
            np.array([(x) if not isiter(x) else [xas for xas in x] for x in item],)
        )
    except:
        pass
    shapetouse = getshape(item)

    try:
        item = item.T.flatten().reshape((shapetouse, -1))
    except Exception as Fehler:
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
        try:
            return np.array(listexxx).T.tolist()
        except Exception as Fehler:
            return listexxx


def get_rightindexlist(arrayname, leveldeep, fixedcolsize=None, dimensions=1):
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


def get_path(leveldeep):
    indtoprint = "[" + "][".join([str(x) for x in leveldeep]) + "]"
    return indtoprint


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
            if not isiter(v):
                if repr_or_string == "repr":
                    endvalue = repr(v)
                else:
                    endvalue = str(v)
                if isinstance(v, str):
                    endvalue = regex_endvalue1.sub(r"\g<1>", endvalue)

                    endvalue = regex_endvalue2.sub('"', endvalue)

                    endvalue = regex_endvalue3.sub('"', endvalue)
            else:
                endvalue = v

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
                results[prefix]["value"] = endvalue

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
    arrname = ""
    for k, v in new_d.items():
        for pref in v["keys"]:
            print(pref, end="")
            arrname = arrname + str(pref)
        print(((maxlen_di + 4) - v["len"]) * " ", end="")
        print("= ", end="")
        if not isiter(v["value"]):
            print(
                TC(
                    f'{v["value"]}'.replace("\n", "\\n").replace("\r", "\\r")
                ).bg_lightgrey.fg_black,
                end="",
            )
        else:
            print("\n")
            numpyprinter(v["value"], arrayname=arrname)
        arrname = ""
        print("", end="\n")


def get_blackblock():
    return TC("█").fg_black.bg_black


def get_purple_block():
    return TC("█").bg_lightgrey.fg_purple


def print_purple_block():
    print(get_purple_block(), end="")


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
    reshape_big_1_dim_arrays=0,  # 0 == no reshaping
    when_to_take_a_break=0,  # 0 == no break
    break_how_long=5,
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
            arrayname, leveldeep[1:] + [0], fixedcolsize=None
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
    if reshape_big_1_dim_arrays > 0 and wholearray.ndim <= 1 and firstitem:
        try:
            wholearray = wholearray.flatten()

            forreshape = 1
            reshape_big_1_dimension_arrays = 19
            for reshape_big_1_dimension_arrays_ in reversed(
                range(1, reshape_big_1_dimension_arrays)
            ):
                if wholearray.flatten().shape[0] % reshape_big_1_dimension_arrays_ == 0:
                    forreshape = reshape_big_1_dimension_arrays_
                    break

            wholearray = wholearray.reshape((-1, forreshape))
            print(
                TC(
                    f"\nBe careful!!Array was reshaped to {wholearray.shape}! \nThat means the index values that you see on the left side of each column (e.g. a[0][0]) are invalid for the original array!\n"
                ).fg_red.bg_black.bold.underline
            )
        except Exception:
            pass

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
                    itemtoprint.replace("\n", "\\n").replace("\r", "\\r") + spacing
                )[:longi]
                if fixedcolsize is None:
                    fixedcolsize, indtoprint = get_rightindexlist(
                        arrayname,
                        leveldeep[1:],
                        fixedcolsize=fixedcolsize,
                        dimensions=dimensions,
                    )
                else:
                    _, indtoprint = get_rightindexlist(
                        arrayname,
                        leveldeep[1:],
                        fixedcolsize=fixedcolsize,
                        dimensions=dimensions,
                    )

                checkhash = hash(indtoprint)

                if not checkhash in alreadyprinted:
                    if fixedcolsize is None:
                        fixedcolsize, indtoprint = get_rightindexlist(
                            arrayname,
                            leveldeep[1:] + [indi0],
                            fixedcolsize=fixedcolsize,
                            dimensions=dimensions,
                        )
                    else:
                        _, indtoprint = get_rightindexlist(
                            arrayname,
                            leveldeep[1:] + [indi0],
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
                            fixedcolsize=fixedcolsize,
                            dimensions=dimensions,
                        )
                    else:
                        _, indtoprint = get_rightindexlist(
                            arrayname,
                            leveldeep[1:] + [len(intotal)],
                            fixedcolsize=fixedcolsize,
                            dimensions=dimensions,
                        )

                    row_ = f"{indent_}{indtoprint}{get_sep_block()}{get_blackblock()}{get_blackblock()}"
                    print(TC(row_).fg_darkgrey.bg_black, end="")

                    try:
                        row2 = f"{itemtoprint.ljust(longi).rjust(longi)}  "
                        func = getattr(TC(f"{row2}"), npcolors[indi0])
                        print(func.bg_black, end="")
                    except Exception:
                        itemtoprint = f"Could not print"
                        row2 = f"{itemtoprint.ljust(longi).rjust(longi)}  "
                        func = getattr(TC(f"{row2}"), npcolors[indi0])
                        print(func.bg_black, end="")

                    print_purple_block()
                    intotal.append(1)

            else:
                intotal.append(1)

                leveldeepold = leveldeep.copy()
                leveldeep = leveldeep + [indi0]
                if when_to_take_a_break != 0:
                    try:
                        if leveldeep[-1] % when_to_take_a_break == 0:
                            ixxx = InputTimeout(
                                timeout=break_how_long,
                                input_message="Any key to continue",
                                timeout_message="",
                                defaultvalue="CONTINUE",
                                cancelbutton=None,
                                show_special_characters_warning=None,
                            ).finalvalue
                            print("\n")
                            if ixxx != "CONTINUE".strip() and ixxx != "".strip():
                                return
                    except Exception as Feh:
                        pass

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
                            reshape_big_1_dim_arrays=reshape_big_1_dim_arrays,
                            when_to_take_a_break=when_to_take_a_break,
                            break_how_long=break_how_long,
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
                            reshape_big_1_dim_arrays=reshape_big_1_dim_arrays,
                            when_to_take_a_break=when_to_take_a_break,
                            break_how_long=break_how_long,
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
                        reshape_big_1_dim_arrays=reshape_big_1_dim_arrays,
                        when_to_take_a_break=when_to_take_a_break,
                        break_how_long=break_how_long,
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


def pdp(
    dframe: Any,
    max_colwidth: int = 50,
    repeat_cols: int = 70,
    printasnp: bool = False,
    reshape_big_1_dim_arrays: int = 0,
    when_to_take_a_break: int = 0,
    break_how_long: int = 5,
) -> None:
    """
    Parameters
    ----------
    dframe : tuple, dict, list, np.ndarray, pd.Dataframe, pd.Series
        Array to print
    max_colwidth : int
        Width of each column (default is 50)
    repeat_cols : int (default is 70)
        Print columns again after n lines  (default is 70)
    printasnp: bool (default is False)
        Converts pandas DataFrame to np before printing.
        If there are duplicated columns in a Pandas DataFrame,
        it changes to printasnp = True  (default is False)
    reshape_big_1_dim_arrays: int
        if you have a huge one dimensional np array,
        you can use this option to reshape it before printing.
        0 means no reshaping  (default is 0)
    when_to_take_a_break: int
        You can pause after n lines to check your data.
        Press ENTER to continue or ANY KEY + ENTER to break (default is 0  [No break])
    break_how_long: int
        time to sleep, can be interrupted by pressing ENTER or ANY KEY + ENTER to break

        """
    max_column_size = max_colwidth
    arrayname = "a"
    if printasnp:
        if isinstance(dframe, (pd.DataFrame, pd.Series)):
            dframe = dframe.values
            arrayname = f"{arrayname}.iloc"
        numpyprinter(
            dframe,
            max_col_width=max_column_size,
            arrayname=arrayname,
            withindent=False,
            reshape_big_1_dim_arrays=reshape_big_1_dim_arrays,
            when_to_take_a_break=when_to_take_a_break,
            break_how_long=break_how_long,
        )
        return

    if isinstance(dframe, pd.Series):
        df = dframe.to_frame()
    else:
        df = dframe
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        numpyprinter(
            df,
            max_col_width=max_column_size,
            arrayname=arrayname,
            reshape_big_1_dim_arrays=reshape_big_1_dim_arrays,
            when_to_take_a_break=when_to_take_a_break,
            break_how_long=break_how_long,
        )
        return
    if len(list(dict.fromkeys(df.columns.to_list()))) != len(df.columns.to_list()):
        numpyprinter(
            df.values,
            max_col_width=max_column_size,
            arrayname=f"{arrayname}.iloc",
            withindent=False,
            reshape_big_1_dim_arrays=reshape_big_1_dim_arrays,
            when_to_take_a_break=when_to_take_a_break,
            break_how_long=break_how_long,
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
        str(f"{col}").replace("\n", "\\n").replace("\r", "\\r")[:size].rjust(size + 1)
        for col, size in zip(df.columns, maxsize_)
    ]
    columnsprint.insert(0, "index".ljust(maxindexsize + 1))
    _print_cols()
    colprintcounter = 0
    breakmaker = 0
    npcolors = _get_colors(len(maxsize))

    for indi, row in zip(df.index, df.__array__()):

        emptyline = 0
        indi_ = (
            str(f" {indi}")
            .replace("\n", "\\n")
            .replace("\r", "\\r")[:maxindexsize]
            .ljust(maxindexsize + 1)
        )
        breakmaker = breakmaker + 1
        if when_to_take_a_break != 0:
            if when_to_take_a_break == breakmaker:
                breakmaker = 0
                try:
                    ixxx = InputTimeout(
                        timeout=break_how_long,
                        input_message="Press ENTER to continue, or ANY LETTER + ENTER to break! ",
                        timeout_message="",
                        defaultvalue="CONTINUE",
                        cancelbutton="esc",
                        show_special_characters_warning=None,
                    ).finalvalue
                    print("\n")
                    if ixxx != "CONTINUE".strip() and ixxx != "".strip():
                        return
                except Exception as Feh:
                    pass

        colprintcounter = colprintcounter + 1
        if repeat_cols != 0:
            if repeat_cols <= colprintcounter:
                _print_cols()
                colprintcounter = 0
        print(TC(f"{indi_}  ").fg_black.bg_cyan, end="")
        print(TC("█").fg_blue.bg_black, end="")

        for r, size, color in zip(row, maxsize_, npcolors):
            row_ = str(f"  {r}").replace("\n", "\\n").replace("\r", "\\r") + " " * size
            row_ = row_[:size].rjust(size + 1).ljust(size + 1)
            try:
                func = getattr(TC(f"{row_}  "), color)
                print(func.bg_black, end="")
            except Exception:
                row_ = (
                    str(f"  Could not print  ")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    + " " * size
                )
                row_ = row_[:size].rjust(size + 1).ljust(size + 1)
                func = getattr(TC(f"{row_}  "), color)
                print(func.bg_black, end="")
            print(TC("█").fg_yellow.bg_black, end="")
            emptyline = emptyline + 1

        print("", end="\n")


def flattenlist_neu(iterable, types=(list, tuple)):
    def iter_flatten(iterable):
        it = iter(iterable)
        for e in it:
            if isinstance(e, types):
                for f in iter_flatten(e):
                    yield f
            else:
                yield e

    a = [i for i in iter_flatten(iterable)]
    return a


def _print_normal_pandas(
    df: Union[pd.DataFrame, pd.Series],
    maxrows: Union[None, int] = None,
    maxcols: Union[None, int] = None,
    max_colwidth: Union[None, int] = None,
) -> None:
    """
    Parameters
    ----------
    df : pd.DataFrame, pd.Series
        Df/Series to print
    maxrows : int, None
        display.max_rows (default is None)
    maxcols : int, None
        display.max_columns (default is None)
    max_colwidth : int, None
        max_colwidth (default is None)

        """
    with pd.option_context(
        "display.max_rows",
        maxrows,
        "display.max_columns",
        maxcols,
        "max_colwidth",
        max_colwidth,
    ):
        print(df)


def qq_ds_print(
    df: Union[pd.DataFrame, pd.Series],
    max_rows: int = 1000,
    max_colwidth: int = 300,
    repeat_cols: int = 70,
    asnumpy: bool = False,
    returndf: bool = False,
) -> Union[pd.DataFrame, pd.Series, None]:
    """
    Parameters
    ----------
    df : pd.DataFrame, pd.Series
        Array to print
    max_rows : int
        Stop printing after n lines (default is 1000)
    max_colwidth : int
        Width of each column (default is 300)
    repeat_cols : int (default is 70)
        Print columns again after n lines  (default is 70)
    asnumpy: bool (default is False)
        Converts pandas DataFrame to np before printing.
        If there are duplicated columns in a Pandas DataFrame,
        it changes to printasnp = True  (default is False)
    returndf:  bool (default is False)
        return the input DataFrame to allow chaining


        """
    dftouse = df
    try:
        if len(df.index[0]) > 1:
            df2 = df.copy().reset_index()
            dftouse = df2
    except Exception:
        pass
    try:
        pdp(
            dftouse[:max_rows],
            max_colwidth=max_colwidth,
            repeat_cols=repeat_cols,
            printasnp=asnumpy,
        )
    except Exception:
        _print_normal_pandas(
            df, maxrows=max_rows, maxcols=None, max_colwidth=max_colwidth
        )
    if returndf:
        return df


def qq_ds_print_nolimit(
    df: Union[pd.DataFrame, pd.Series],
    max_colwidth: int = 300,
    repeat_cols: int = 70,
    returndf: bool = False,
) -> Union[pd.DataFrame, pd.Series, None]:
    """
    Parameters
    ----------
    df : pd.DataFrame, pd.Series
        Array to print
    max_colwidth : int
        Width of each column (default is 300)
    repeat_cols : int (default is 70)
        Print columns again after n lines  (default is 70)
    returndf:  bool (default is False)
        return the input DataFrame to allow chaining


        """

    dftouse = df
    try:
        if len(df.index[0]) > 1:
            df2 = df.copy().reset_index()
            dftouse = df2
    except Exception:
        pass
    try:
        pdp(
            dftouse,
            max_colwidth=max_colwidth,
            repeat_cols=repeat_cols,
            printasnp=False,
        )
    except Exception:
        _print_normal_pandas(df, maxrows=None, maxcols=None, max_colwidth=None)
    if returndf:
        return df


def qq_ds_print_context(
    df: Union[pd.DataFrame, pd.Series],
    index: Any,
    top: int = 5,
    bottom: int = 5,
    max_colwidth: int = 300,
    repeat_cols: int = 70,
    returndf: bool = False,
    asnumpy: bool = False,
) -> Union[pd.DataFrame, pd.Series, None]:

    """
    Parameters
    ----------
    df : pd.DataFrame, pd.Series
        Array to print
    index: Any
        index you want to print
    top: int
        start printing n rows before index (default is 5)
    bottom: int
        stop printing n rows after index (default is 5)
    repeat_cols : int (default is 70)
        Print columns again after n lines  (default is 70)

    max_colwidth : int
        Width of each column (default is 300)

    asnumpy: bool (default is False)
        Converts pandas DataFrame to np before printing.
        If there are duplicated columns in a Pandas DataFrame,
        it changes to printasnp = True  (default is False)
    returndf:  bool (default is False)
        return the input DataFrame to allow chaining


        """
    iloc_index = df.index.get_indexer_for([index]).tolist()[0]
    beginning = iloc_index - top
    len_df = len(df)
    while beginning < 0:
        beginning = beginning + 1
    ending = iloc_index + bottom
    while ending > len_df - 1:
        ending = ending - 1
    finaldf = df[beginning:ending]
    qq_ds_print(
        finaldf,
        max_rows=len(finaldf),
        max_colwidth=max_colwidth,
        repeat_cols=repeat_cols,
        asnumpy=asnumpy,
        returndf=False,
    )
    if returndf:
        return finaldf.copy()


def qq_d_print_columns(
    df: Union[pd.DataFrame, pd.Series], max_colwidth: int = 300, asnumpy: bool = False,
) -> None:
    """
    Parameters
    ----------
    df : pd.DataFrame, pd.Series
        Df/Series to print

    max_colwidth : int, None
        max_colwidth (default is None)
    asnumpy: bool (default is False)
        Converts pandas DataFrame to np before printing.
        If there are duplicated columns in a Pandas DataFrame,
        it changes to printasnp = True  (default is False)
        """
    pdp(
        df.columns.__array__().reshape((-1, 1)),
        max_colwidth=max_colwidth,
        repeat_cols=5000,
        printasnp=asnumpy,
    )


def qq_ds_print_index(
    df: Union[pd.DataFrame, pd.Series], max_colwidth: int = 300, asnumpy: bool = False,
) -> None:
    """
    Parameters
    ----------
    df : pd.DataFrame, pd.Series
        Df/Series to print

    max_colwidth : int, None
        max_colwidth (default is None)
    asnumpy: bool (default is False)
        Converts pandas DataFrame to np before printing.
        If there are duplicated columns in a Pandas DataFrame,
        it changes to printasnp = True  (default is False)
        """
    pdp(
        df.index.__array__().reshape((-1, 1)),
        max_colwidth=max_colwidth,
        repeat_cols=5000,
        printasnp=asnumpy,
    )


def print_test_from_pandas_github():
    from random import choice

    add_printer()
    csvtests = [
        "https://github.com/pandas-dev/pandas/raw/main/doc/data/air_quality_long.csv",
        "https://github.com/pandas-dev/pandas/raw/main/doc/data/air_quality_no2.csv",
        "https://github.com/pandas-dev/pandas/raw/main/doc/data/air_quality_no2_long.csv",
        "https://github.com/pandas-dev/pandas/raw/main/doc/data/air_quality_parameters.csv",
        "https://github.com/pandas-dev/pandas/raw/main/doc/data/air_quality_pm25_long.csv",
        "https://github.com/pandas-dev/pandas/raw/main/doc/data/air_quality_stations.csv",
        "https://github.com/pandas-dev/pandas/raw/main/doc/data/baseball.csv",
        "https://github.com/pandas-dev/pandas/raw/main/doc/data/titanic.csv",
    ]
    csvfile = choice(csvtests)
    df = pd.read_csv(csvfile)
    print(f"Downloading: {csvfile}")

    print("""Executing: df.ds_color_print()""")
    df.ds_color_print()
    print("""Executing: df.ds_color_print(max_rows=25)""")
    df.ds_color_print(max_rows=25)
    print("""Executing: df.ds_color_print(max_rows=25, max_colwidth=15)""")
    df.ds_color_print(max_rows=25, max_colwidth=15)
    print(
        """Executing: df.ds_color_print(max_rows=25, max_colwidth=15, repeat_cols=10)"""
    )
    df.ds_color_print(max_rows=25, max_colwidth=15, repeat_cols=10)
    print(
        """Executing: df.ds_color_print(max_rows=25, max_colwidth=15, repeat_cols=10, asnumpy=True)"""
    )
    df.ds_color_print(max_rows=25, max_colwidth=15, repeat_cols=10, asnumpy=True)
    print(
        """Executing: df.ds_color_print(max_rows=25, max_colwidth=15, repeat_cols=10, asnumpy=True, returndf=True)"""
    )
    df.ds_color_print(
        max_rows=25, max_colwidth=15, repeat_cols=10, asnumpy=True, returndf=True
    )

    print("""Executing: df[choice(df.columns)].ds_color_print()""")
    df[choice(df.columns)].ds_color_print()
    print("""Executing: df[choice(df.columns)].ds_color_print(max_rows=50)""")
    df[choice(df.columns)].ds_color_print(max_rows=50)
    print(
        """Executing: df[choice(df.columns)].ds_color_print(max_rows=50, max_colwidth=10)"""
    )
    df[choice(df.columns)].ds_color_print(max_rows=50, max_colwidth=10)
    print(
        """Executing: df[choice(df.columns)].ds_color_print(max_rows=50, max_colwidth=10, repeat_cols=10)"""
    )
    df[choice(df.columns)].ds_color_print(max_rows=50, max_colwidth=10, repeat_cols=10)
    print(
        """Executing: df[choice(df.columns)].ds_color_print(max_rows=50, max_colwidth=10, repeat_cols=10, asnumpy=True)"""
    )
    df[choice(df.columns)].ds_color_print(
        max_rows=50, max_colwidth=10, repeat_cols=10, asnumpy=True
    )
    print(
        """Executing: df[choice(df.columns)].ds_color_print(max_rows=50, max_colwidth=10, repeat_cols=10, asnumpy=True, returndf=True)"""
    )
    df[choice(df.columns)].ds_color_print(
        max_rows=50, max_colwidth=10, repeat_cols=10, asnumpy=True, returndf=True
    )
    print("""Executing: df.ds_color_print_all()""")
    df.ds_color_print_all()
    print("""Executing: df[choice(df.columns)].ds_color_print_all()""")
    df[choice(df.columns)].ds_color_print_all()
    print("""Executing: df.ds_color_print_context(index=choice(df.index))""")
    df.ds_color_print_context(index=choice(df.index))
    print(
        """Executing: df[choice(df.columns)].ds_color_print_context(index=choice(df.index))"""
    )
    df[choice(df.columns)].ds_color_print_context(index=choice(df.index))
    print(
        """Executing: df.ds_color_print_all_with_break( max_colwidth=300, when_to_take_a_break=69, break_how_long=10, repeat_cols=70, returndf=False)"""
    )
    df.ds_color_print_all_with_break(
        max_colwidth=300,
        when_to_take_a_break=69,
        break_how_long=10,
        repeat_cols=70,
        returndf=False,
    )


def qq_ds_print_nolimit_with_break(
    df: Union[pd.DataFrame, pd.Series],
    max_colwidth: int = 300,
    when_to_take_a_break: int = 20,
    break_how_long: int = 10,
    repeat_cols: int = 70,
    returndf: bool = False,
) -> Union[None, pd.Series, pd.DataFrame]:
    """
    Parameters
    ----------
    df : tuple, dict, list, np.ndarray, pd.Dataframe, pd.Series
        Array to print
    max_colwidth : int
        Width of each column (default is 300)
    when_to_take_a_break: int
        You can pause after n lines to check your data.
        Press ENTER to continue or ANY KEY + ENTER to break (default is 0  [No break])
    break_how_long: int
        time to sleep, can be interrupted by pressing ENTER or ANY KEY + ENTER to break
    repeat_cols : int (default is 70)
        Print columns again after n lines  (default is 70)
    returndf:  bool (default is False)
        return the input DataFrame to allow chaining
        """
    dftouse = df
    try:
        if len(df.index[0]) > 1:
            df2 = df.copy().reset_index()
            dftouse = df2
    except Exception:
        pass
    try:
        pdp(
            dftouse,
            max_colwidth=max_colwidth,
            repeat_cols=repeat_cols,
            printasnp=False,
            when_to_take_a_break=when_to_take_a_break,
            break_how_long=break_how_long,
        )
    except Exception:
        _print_normal_pandas(df, maxrows=None, maxcols=None, max_colwidth=None)
    if returndf:
        return df


def add_printer():
    PandasObject.ds_color_print = qq_ds_print
    PandasObject.ds_color_print_all = qq_ds_print_nolimit
    PandasObject.d_color_print_columns = qq_d_print_columns
    PandasObject.d_color_print_index = qq_ds_print_index
    PandasObject.ds_color_print_all_with_break = qq_ds_print_nolimit_with_break
    PandasObject.ds_color_print_context = qq_ds_print_context
