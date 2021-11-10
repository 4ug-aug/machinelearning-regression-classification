def import_dataset(year = None):
    """opens the data from /archive and returns the pandas dataframe

    Args:
        year (int, optional): year of data to get. Defaults to None.

    Raises:
        Exception: No year specified
    """

    import pandas as pd
    import numpy as np
    if year and year != 2019:
        fp = f"archive/{year}.csv"
    elif year == 2019:
        raise Exception("import_dataset() - Year 2019 specified which does not meet requirements for further usage")
    else:
        print("import_dataset() - No year specified")
        raise Exception("No year specified")

    df = pd.read_csv(fp)

    hap_score = np.median(df["Happiness Score"])

    print("import_dataset() - Adding new column: Happy (Binary)")

    df["Happy"] = np.where(df["Happiness Score"] > hap_score, 1,0)

    return df

if __name__ == "__main__":
    import_dataset(2016)