# Model run - Random Forest
import sys

# Ambil argumen yang diberikan
run_selection = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else [
        "RF211",
        "RF212",
        "RF213",
        "RF214",
        "RF221",
        "RF222",
        "RF223",
        "RF224",
        "RF231",
        "RF232",
        "RF241",
        "RF242",
        "RF251",
        "RF252",
        "RF261",
        "RF262",
        "RF271",
        "RF272",
        "RF281",
        "RF282",
        "RF291",
        "RF292",
    ]
)


tuning_tfidf1 = {
    "n_estimators": 400,
    "max_depth": 30,
    "random_state": 42,
    "min_samples_split": 15,
    "min_samples_leaf": 4,
    "max_features": 0.5,
}

tuning_tfidf2 = {
    "n_estimators": 500,
    "max_depth": 50,
    "random_state": 42,
    "min_samples_split": 20,
    "min_samples_leaf": 4,
    "max_features": 0.5,
}

tuning_w2vec1 = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
}

# scheme 80/20

# 'Run RF211 (scheme 80/20, tfidf)'
if "RF211" in run_selection:
    setModelRun = {
        "IdRun": "RF211",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])

# 'Run RF212 (scheme 80/20, tfidf, smote)'
if "RF212" in run_selection:
    setModelRun = {
        "IdRun": "RF212",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])

# 'Run RF213 (scheme 80/20, w2vec)'
if "RF213" in run_selection:
    setModelRun = {
        "IdRun": "RF213",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF214 (scheme 80/20, w2vec, smote)'
if "RF214" in run_selection:
    setModelRun = {
        "IdRun": "RF214",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# scheme 70/30
# 'Run RF221 (scheme 70/30, tfidf)'
if "RF221" in run_selection:
    setModelRun = {
        "IdRun": "RF221",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF222 (scheme 70/30, tfidf, smote)'
if "RF222" in run_selection:
    setModelRun = {
        "IdRun": "RF222",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF223 (scheme 70/30, w2vec)'
if "RF223" in run_selection:
    setModelRun = {
        "IdRun": "RF223",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF224 (scheme 70/30, w2vec, smote)'
if "RF224" in run_selection:
    setModelRun = {
        "IdRun": "RF224",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": True},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])

# tuning
# 'Run RF231 (scheme 80/20, tfidf, tuning1)'
if "RF231" in run_selection:
    setModelRun = {
        "IdRun": "RF231",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_tfidf1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF232 (scheme 80/20, tfidf, smote, tuning1)'
if "RF232" in run_selection:
    setModelRun = {
        "IdRun": "RF232",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_tfidf1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])

# 'Run RF241 (scheme 80/20, tfidf, tuning2)'
if "RF241" in run_selection:
    setModelRun = {
        "IdRun": "RF241",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_tfidf2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])

# 'Run RF242 (scheme 80/20, tfidf, smote, tuning2)'
if "RF242" in run_selection:
    setModelRun = {
        "IdRun": "RF242",
        "options": {"extFitType": "tfidf", "scheme": "80/20", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_tfidf2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF251 (scheme 70/30, tfidf, tuning1)'
if "RF251" in run_selection:
    setModelRun = {
        "IdRun": "RF251",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_tfidf1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF252 (scheme 70/30, tfidf, smote, tuning1)'
if "RF252" in run_selection:
    setModelRun = {
        "IdRun": "RF252",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_tfidf1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF261 (scheme 70/30, tfidf, tuning2)'
if "RF261" in run_selection:
    setModelRun = {
        "IdRun": "RF261",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_tfidf2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF262 (scheme 70/30, tfidf, smote, tuning2)'
if "RF262" in run_selection:
    setModelRun = {
        "IdRun": "RF262",
        "options": {"extFitType": "tfidf", "scheme": "70/30", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_tfidf2
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])

# 'Run RF271 (scheme 80/20, w2vec, tuning3)'
if "RF271" in run_selection:
    setModelRun = {
        "IdRun": "RF271",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_w2vec1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])

# 'Run RF272 (scheme 80/20, w2vec, smote, tuning3)'
if "RF272" in run_selection:
    setModelRun = {
        "IdRun": "RF272",
        "options": {"extFitType": "w2vec", "scheme": "80/20", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_w2vec1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF281 (scheme 70/30, w2vec, tuning3)'
if "RF281" in run_selection:
    setModelRun = {
        "IdRun": "RF281",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": False},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_w2vec1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF282 (scheme 70/30, w2vec, smote, tuning3)'
if "RF282" in run_selection:
    setModelRun = {
        "IdRun": "RF282",
        "options": {"extFitType": "w2vec", "scheme": "70/30", "smote": True},
        "Params": {},
    }
    setModelRun["Params"]["modelParams"] = tuning_w2vec1
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])


# 'Run RF291 (scheme 80/20, tfidf, randSCV)'
if "RF291" in run_selection:
    setModelRun = {
        "IdRun": "RF291",
        "options": {
            "randSCV": True,
            "extFitType": "tfidf",
            "scheme": "80/20",
            "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])

# 'Run RF292 (scheme 70/30, tfidf, randSCV)'
if "RF292" in run_selection:
    setModelRun = {
        "IdRun": "RF292",
        "options": {
            "randSCV": True,
            "extFitType": "tfidf",
            "scheme": "70/30",
            "smote": False},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runRandomForest(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Random Forest"])