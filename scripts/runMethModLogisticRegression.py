# Logistic Regression (LR)
from sklearn.linear_model import LogisticRegression

def runLogisticRegression(setxi, X, y, resultPrint=False):
    # Load config set
    setx = setxi.copy()
    setx['ModelName'] = 'Logistic Regression'

    scheme = setx['options'].get('scheme')
    extFitType = setx['options'].get('extFitType', 'default')
    use_smote = setx['options'].get('smote', False)
    scoring_metric = setx['options'].get('scoring', 'accuracy')

    setx.setdefault('Params', {})

    # Feature extraction
    X_train, X_test, y_train, y_test = extrFeat(X, y, extFeatType=extFitType, scheme=scheme, getReturn=True)

    # Implement SMOTE if needed to deal with unbalanced data
    if use_smote:
        smote_obj = SMOTE(sampling_strategy="auto", random_state=42)
        X_train, y_train = smote_obj.fit_resample(X_train, y_train)

    # Conversion
    if extFitType == 'tfidf' and hasattr(X_train, "toarray"):
        X_train, X_test = X_train.toarray(), X_test.toarray()
    if extFitType == 'w2vec' and isinstance(X_train, list):
        X_train, X_test = np.array(X_train), np.array(X_test)

    # Model selection & parameters
    if setx['options'].get('gridSCV', False):
        params_grid = setx['Params'].get('gridParams', {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        })
        lr = LogisticRegression(max_iter=500, random_state=42)
        grid_search = GridSearchCV(lr, params_grid, cv=5, scoring=scoring_metric, verbose=1, n_jobs=1)
        grid_search.fit(X_train, y_train)
        logistic_regression = grid_search.best_estimator_
        setx['Params']['BestParamsGridSCV'] = grid_search.best_params_

    elif setx['options'].get('randSCV', False):
        params_randSCV = setx['Params'].get('randSCVParams', {
            'C': loguniform(0.01, 10),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        })
        lr = LogisticRegression(max_iter=500, random_state=42)
        random_search = RandomizedSearchCV(
            lr, params_randSCV, n_iter=10, cv=5, scoring=scoring_metric, random_state=42, verbose=1, n_jobs=1
        )
        random_search.fit(X_train, y_train)
        logistic_regression = random_search.best_estimator_
        setx['Params']['BestParamsRandSCV'] = random_search.best_params_
    else:
        # Parameter default
        default_params_model = {'random_state': 42}
        params_model = {**default_params_model, **setx['Params'].get('modelParams', {})}
        setx['Params']['modelParams'] = params_model
        logistic_regression = LogisticRegression(**params_model)

    # Training & Prediction
    logistic_regression.fit(X_train, y_train)
    y_pred_train_lr = logistic_regression.predict(X_train)
    y_pred_test_lr = logistic_regression.predict(X_test)

    # Model evaluation
    setx['AccuracyTrain'] = round(accuracy_score(y_train, y_pred_train_lr), 8)
    setx['AccuracyTest'] = round(accuracy_score(y_test, y_pred_test_lr), 8)
    setx['PrecisionTest'] = round(precision_score(y_test, y_pred_test_lr, average='weighted'), 8)
    setx['RecallTest'] = round(recall_score(y_test, y_pred_test_lr, average='weighted'), 8)
    setx['F1ScoreTest'] = round(f1_score(y_test, y_pred_test_lr, average='weighted'), 8)

    # Show results if needed
    if resultPrint:
        print(f"Logistic Regression ({extFitType}{', SMOTE' if use_smote else ''})")
        print(f'\taccuracy_train: {setx["AccuracyTrain"]}')
        print(f'\taccuracy_test: {setx["AccuracyTest"]}')
        print(f'\tprecision_test: {setx["PrecisionTest"]}')
        print(f'\trecall_test: {setx["RecallTest"]}')
        print(f'\tf1_score_test: {setx["F1ScoreTest"]}\n')

    return setx, logistic_regression