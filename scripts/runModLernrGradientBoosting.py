# Model run - Gradient Boosting
import sys

# Ambil argumen yang diberikan
run_selection = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else [
        "GB511",
        "GB512",
        "GB513",
        "GB514",
        "GB521",
        "GB522",
        "GB523",
        "GB524",
    ]
)
# scheme 80/20
# 'Run GB511 (scheme 80/20, tfidf)'
if "GB511" in run_selection:
    setModelRun = {
        "IdRun": "GB511",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runGradientBoosting(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Gradient Boosting"]
    )

# 'Run GB512 (scheme 80/20, tfidf, smote)'
if "GB512" in run_selection:
    setModelRun = {
        "IdRun": "GB512",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runGradientBoosting(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Gradient Boosting"]
    )

# 'Run GB513 (scheme 80/20, w2vec)'
if "GB513" in run_selection:
    setModelRun = {
        "IdRun": "GB513",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runGradientBoosting(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Gradient Boosting"]
    )

# 'Run GB514 (scheme 80/20, w2vec, smote)'
if "GB514" in run_selection:
    setModelRun = {
        "IdRun": "GB514",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runGradientBoosting(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Gradient Boosting"]
    )


# scheme 70/30
# 'Run GB521 (scheme 70/30, tfidf)'
if "GB521" in run_selection:
    setModelRun = {
        "IdRun": "GB521",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runGradientBoosting(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Gradient Boosting"]
    )

# 'Run GB522 (scheme 70/30, tfidf, smote)'
if "GB522" in run_selection:
    setModelRun = {
        "IdRun": "GB522",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runGradientBoosting(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Gradient Boosting"]
    )


# 'Run GB523 (scheme 70/30, w2vec)'
if "GB523" in run_selection:
    setModelRun = {
        "IdRun": "GB523",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runGradientBoosting(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Gradient Boosting"]
    )

# 'Run GB524 (scheme 70/30, w2vec, smote)'
if "GB524" in run_selection:
    setModelRun = {
        "IdRun": "GB524",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runGradientBoosting(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Gradient Boosting"]
    )
