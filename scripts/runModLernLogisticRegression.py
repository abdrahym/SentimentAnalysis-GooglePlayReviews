# Model run - Logistic Regression

import sys

# Ambil argumen yang diberikan
run_selection = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else [
        "LR311",
        "LR312",
        "LR313",
        "LR314",
        "LR321",
        "LR322",
        "LR323",
        "LR324",
        "LR391",
        "LR392",
    ]
)


# scheme 80/20
# 'Run LR311 (scheme 80/20, tfidf)'
if "LR311" in run_selection:
    setModelRun = {
        "IdRun": "LR311",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"]
    )

# 'Run LR312 (scheme 80/20, tfidf, smote)'
if "LR312" in run_selection:
    setModelRun = {
        "IdRun": "LR312",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"]
    )

# 'Run LR313 (scheme 80/20, w2vec)'
if "LR313" in run_selection:
    setModelRun = {
        "IdRun": "LR313",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"]
    )


# 'Run LR314 (scheme 80/20, w2vec, smote)'
if "LR314" in run_selection:
    setModelRun = {
        "IdRun": "LR314",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"]
    )

# scheme 70/30
# 'Run LR321 (scheme 70/30, tfidf)'
if "LR321" in run_selection:
    setModelRun = {
        "IdRun": "LR321",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"]
    )

# 'Run LR322 (scheme 70/30, tfidf, smote)'
if "LR322" in run_selection:
    setModelRun = {
        "IdRun": "LR322",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"]
    )


# 'Run LR323 (scheme 70/30, w2vec)'
if "LR323" in run_selection:
    setModelRun = {
        "IdRun": "LR323",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"]
    )

# 'Run LR324 (scheme 70/30, w2vec, smote)'
if "LR324" in run_selection:
    setModelRun = {
        "IdRun": "LR324",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"]
    )


# 'Run LR391 (scheme 80/20, tfidf, randSCV)'
if "LR391" in run_selection:
    setModelRun = {
        "IdRun": "LR391",
        "options": {
            "randSCV": True,
            "extFitType": "tfidf",
            "scheme": "80/20",
            "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"])

# 'Run LR392 (scheme 70/30, tfidf, randSCV)'
if "LR392" in run_selection:
    setModelRun = {
        "IdRun": "LR392",
        "options": {
            "randSCV": True,
            "extFitType": "tfidf",
            "scheme": "70/30",
            "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runLogisticRegression(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Logistic Regression"])