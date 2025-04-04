# Model run - Naive Bayes
import sys

# Take the given argument
run_selection = (
    sys.argv[1:]
    if len(sys.argv) > 1
    else [
        "NB11",
        "NB12",
        "NB13",
        "NB14",
        "NB21",
        "NB23",
    ]
)


# Run 1-1 (scheme 80/20, multinomial)
if "NB11" in run_selection:
    setModelRun = {
        "IdRun": "NB11",
        "options": {
            "ModelType": "multinomial",
            "extFitType": "tfidf",
            "scheme": "80/20",
        },
        "Params": {"modelParams": {"alpha": 0.5}},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runNaiveBayes(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Naive Bayes"])

# Run 1-2 (scheme 80/20, bernoulli)
if "NB12" in run_selection:
    setModelRun = {
        "IdRun": "NB12",
        "options": {"ModelType": "bernoulli", "extFitType": "tfidf", "scheme": "80/20"},
        "Params": {"modelParams": {"alpha": 0.5}},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runNaiveBayes(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Naive Bayes"])

# Run 1-3 (scheme 70/30, multinomial)
if "NB13" in run_selection:
    setModelRun = {
        "IdRun": "NB13",
        "options": {
            "ModelType": "multinomial",
            "extFitType": "tfidf",
            "scheme": "70/30",
        },
        "Params": {"modelParams": {"alpha": 0.5}},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runNaiveBayes(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Naive Bayes"])

# Run 1-4 (scheme 70/30, bernoulli)
if "NB14" in run_selection:
    setModelRun = {
        "IdRun": "NB14",
        "options": {"ModelType": "bernoulli", "extFitType": "tfidf", "scheme": "70/30"},
        "Params": {"modelParams": {"alpha": 0.5}},
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runNaiveBayes(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Naive Bayes"])


# Run 21 (scheme 70/30, multinomial)
if "NB21" in run_selection:
    setModelRun = {
        "IdRun": "NB21",
        "options": {
            "gridSCV": True,
            "ModelType": "multinomial",
            "extFitType": "tfidf",
            "scheme": "70/30",
        },
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runNaiveBayes(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Naive Bayes"])


# Run 22 (scheme 70/30, bernoulli)
if "NB22" in run_selection:
    setModelRun = {
        "IdRun": "NB22",
        "options": {
            "gridSCV": True,
            "ModelType": "bernoulli",
            "extFitType": "tfidf",
            "scheme": "70/30",
        },
    }
    resultRunModel_df = updateRunModelDf(
        resultRunModel_df,
        runNaiveBayes(setModelRun, X, y)[0],
    )
    showResultDf(resultRunModel_df[resultRunModel_df["ModelName"] == "Naive Bayes"])