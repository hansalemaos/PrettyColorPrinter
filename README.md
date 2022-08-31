**With PrettyColorPrinter, you can print numpy arrays / pandas dataframe / list / dicts / tuple! Shows the path to all items! It even works with nested objects.**

Very easy to use:

```python
        print("Testing")
        df = pd.read_csv(
            "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
        )
        df = df[:40]
        print(
            "Regular Dataframe, take a break of 1 sec every 20 lines, can be pulled by pressing enter, any other key + enter will stop the printing"
        )
        pdp(
            df,
            max_column_size=75,
            repeat_cols=20,
            when_to_take_a_break=20,
            break_how_long=10,
        )
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
        pdp(dfnp.flatten(), reshape_big_1_dim_arrays=10)
        user_dict = {}
        user_dict[12] = {
            "Category 1": {"att_1": 1, "att_2": df.__array__()},
            "Category 2": {"att_1": 23, "att_2": df.to_numpy()},
        }

        pdp(user_dict, repeat_cols=50)
```

@

<img title="" src="https://github.com/hansalemaos/PrettyColorPrinter/raw/main/a1.png" alt="">
<img title="" src="https://github.com/hansalemaos/PrettyColorPrinter/raw/main/a2.png" alt="">
<img title="" src="https://github.com/hansalemaos/PrettyColorPrinter/raw/main/a3.png" alt="">
<img title="" src="https://github.com/hansalemaos/PrettyColorPrinter/raw/main/a5.png" alt="">
