# Model run - Decision Tree Classifier
import sys

# Ambil argumen yang diberikan
run_selection = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else [
        "DT411",
        "DT412",
        "DT413",
        "DT414",
        "DT421",
        "DT422",
        "DT423",
        "DT424",
        "DT431",
        "DT432",
        "DT433",
        "DT434",
        "DT441",
        "DT442",
        "DT443",
        "DT444",
        "DT451",
        "DT452",
        "DT453",
        "DT454",
        "DT461",
        "DT462",
        "DT463",
        "DT464",
        "DT491",
        "DT492",
    ]
)
  


tuning1 = {
    "criterion": "entropy",
    "max_depth": 15,
    "random_state": 42,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
}

tuning2 = {
    "criterion": "entropy",
    "max_depth": 10,
    "random_state": 42,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
}

# scheme 80/20
# 'Run DT411 (scheme 80/20, tfidf)'
if "DT411" in run_selection:
    setModelRun = {
        "IdRun": "DT411",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT412 (scheme 80/20, tfidf, smote)'
if "DT412" in run_selection:
    setModelRun = {
        "IdRun": "DT412",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT413 (scheme 80/20, w2vec)'
if "DT413" in run_selection:
    setModelRun = {
        "IdRun": "DT413",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT414 (scheme 80/20, w2vec, smote)'
if "DT414" in run_selection:
    setModelRun = {
        "IdRun": "DT414",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# scheme 80/20 tuning1
# 'Run DT421 (scheme 80/20, tfidf, tuning1)'
if "DT421" in run_selection:
    setModelRun = {
        "IdRun": "DT421",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT422 (scheme 80/20, tfidf, smote, tuning1)'
if "DT422" in run_selection:
    setModelRun = {
        "IdRun": "DT422",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT423 (scheme 80/20, w2vec, tuning1)'
if "DT423" in run_selection:
    setModelRun = {
        "IdRun": "DT423",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT424 (scheme 80/20, w2vec, smote, tuning1)'
if "DT424" in run_selection:
    setModelRun = {
        "IdRun": "DT424",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# scheme 80/20 tuning2
# 'Run DT431 (scheme 80/20, tfidf, tuning2)'
if "DT431" in run_selection:
    setModelRun = {
        "IdRun": "DT431",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT432 (scheme 80/20, tfidf, smote, tuning2)'
if "DT432" in run_selection:
    setModelRun = {
        "IdRun": "DT432",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT433 (scheme 80/20, w2vec, tuning2)'
if "DT433" in run_selection:
    setModelRun = {
        "IdRun": "DT433",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT434 (scheme 80/20, w2vec, smote, tuning2)'
if "DT434" in run_selection:
    setModelRun = {
        "IdRun": "DT434",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# scheme 70/30
# 'Run DT441 (scheme 70/30, tfidf)'
if "DT441" in run_selection:
    setModelRun = {
        "IdRun": "DT441",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT442 (scheme 70/30, tfidf, smote)'
if "DT442" in run_selection:
    setModelRun = {
        "IdRun": "DT442",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT443 (scheme 70/30, w2vec)'
if "DT443" in run_selection:
    setModelRun = {
        "IdRun": "DT443",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT444 (scheme 70/30, w2vec, smote)'
if "DT444" in run_selection:
    setModelRun = {
        "IdRun": "DT444",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# scheme 70/30 tuning1
# 'Run DT451 (scheme 70/30, tfidf, tuning1)'
if "DT451" in run_selection:
    setModelRun = {
        "IdRun": "DT451",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT452 (scheme 70/30, tfidf, smote, tuning1)'
if "DT452" in run_selection:
    setModelRun = {
        "IdRun": "DT452",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT453 (scheme 70/30, w2vec, tuning1)'
if "DT453" in run_selection:
    setModelRun = {
        "IdRun": "DT453",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT454 (scheme 70/30, w2vec, smote, tuning1)'
if "DT454" in run_selection:
    setModelRun = {
        "IdRun": "DT454",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )



# scheme 70/30 tuning2
# 'Run DT461 (scheme 70/30, tfidf, tuning2)'
if "DT461" in run_selection:
    setModelRun = {
        "IdRun": "DT461",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT462 (scheme 70/30, tfidf, smote, tuning2)'
if "DT462" in run_selection:
    setModelRun = {
        "IdRun": "DT462",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )


# 'Run DT463 (scheme 70/30, w2vec, tuning2)'
if "DT463" in run_selection:
    setModelRun = {
        "IdRun": "DT463",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT464 (scheme 70/30, w2vec, smote, tuning2)'
if "DT464" in run_selection:
    setModelRun = {
        "IdRun": "DT464",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(
        resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"]
    )

# 'Run DT491 (scheme 80/20, tfidf, randSCV)'
if "DT491" in run_selection:
    setModelRun = {
        "IdRun": "DT491",
        "options": {
            "randSCV": True,
            "extFitType": "tfidf",
            "scheme": "80/20",
            "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"])


# 'Run DT492 (scheme 70/30, tfidf, randSCV)'
if "DT492" in run_selection:
    setModelRun = {
        "IdRun": "DT492",
        "options": {
            "randSCV": True,
            "extFitType": "tfidf",
            "scheme": "70/30",
            "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runDecisionTree(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "DecisionTree Classifier"])