# Model run - Adaboost Classifier
import sys

# Ambil argumen yang diberikan
run_selection = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else [
        "AD611",
        "AD612",
        "AD613",
        "AD614",
        "AD621",
        "AD622",
        "AD623",
        "AD624",
    ]
)
# scheme 80/20
# 'Run AD611 (scheme 80/20, tfidf)'
if "AD611" in run_selection:
    setModelRun = {
        "IdRun": "AD611",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runAdaBoost(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Adaboost Classifier"]
    )

# 'Run AD612 (scheme 80/20, tfidf, smote)'
if "AD612" in run_selection:
    setModelRun = {
        "IdRun": "AD612",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runAdaBoost(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Adaboost Classifier"]
    )

# 'Run AD613 (scheme 80/20, w2vec)'
if "AD613" in run_selection:
    setModelRun = {
        "IdRun": "AD613",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runAdaBoost(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Adaboost Classifier"]
    )

# 'Run AD614 (scheme 80/20, w2vec, smote)'
if "AD614" in run_selection:
    setModelRun = {
        "IdRun": "AD614",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runAdaBoost(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Adaboost Classifier"]
    )


# scheme 70/30
# 'Run AD621 (scheme 70/30, tfidf)'
if "AD621" in run_selection:
    setModelRun = {
        "IdRun": "AD621",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runAdaBoost(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Adaboost Classifier"]
    )

# 'Run AD622 (scheme 70/30, tfidf, smote)'
if "AD622" in run_selection:
    setModelRun = {
        "IdRun": "AD622",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runAdaBoost(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Adaboost Classifier"]
    )


# 'Run AD623 (scheme 70/30, w2vec)'
if "AD623" in run_selection:
    setModelRun = {
        "IdRun": "AD623",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runAdaBoost(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Adaboost Classifier"]
    )

# 'Run AD624 (scheme 70/30, w2vec, smote)'
if "AD624" in run_selection:
    setModelRun = {
        "IdRun": "AD624",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runAdaBoost(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "Adaboost Classifier"]
    )
