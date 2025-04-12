import pandas as pd
from datetime import datetime

# Create DataFrame resultRunModel_df
resultRunModel_df = pd.DataFrame(
    columns=[
        "ModelName",
        "IdRun",
        "DateAction",
        "options",
        "Params",
        "Remarks",
    ]
)


# Method updateRunModelDf
def updateRunModelDf(df, setx):

    # Search for matching row indexes based on "IdRun"
    index = df[(df["IdRun"] == setx["IdRun"])].index.tolist()

    if index:
        # Overwrite existing rows with new data in column order
        df.loc[index[0], df.columns] = [setx.get(col, None) for col in df.columns]
        df.at[index[0], "DateAction"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        # Add as a new line
        setx["DateAction"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.concat([df, pd.DataFrame([setx])], ignore_index=True)

    # Sort based on "IdRun" by ascending and resetting the index
    df = df.sort_values(by="IdRun").reset_index(drop=True)

    # Save to JSON and CSV files
    df.to_json("scripts/resultRunModel_df.json", orient="records", indent=4)
    df.to_csv("scripts/resultRunModel_df.csv", index=True)

    return df

def resetSetModelRun():
    return {"ModelName": "", "IdRun": "", "options": {}, "Params": {}}


from IPython.display import display, clear_output
def showResultDf(df):
    clear_output(wait=True)
    display(df)
