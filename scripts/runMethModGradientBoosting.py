# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

def runGradientBoosting(setxi, X, y, resultPrint=True):

    # load config set
    setx = setxi.copy()
    setx['ModelName'] = 'Gradient Boosting'

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
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7]
        })
        gbc = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gbc, params_grid, cv=5, scoring=scoring_metric, verbose=1, n_jobs=1
        )
        grid_search.fit(X_train, y_train)
        gradient_boosting = grid_search.best_estimator_
        setx['Params']['BestParamsGridSCV'] = grid_search.best_params_

    elif setx['options'].get('randSCV', False):
        params_randSCV = setx['Params'].get('randSCVParams', {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.5),
            'max_depth': randint(3, 7)
        })
        gbc = GradientBoostingClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            gbc, params_randSCV, n_iter=10, cv=3, scoring=scoring_metric, random_state=42, verbose=1, n_jobs=1
        )
        random_search.fit(X_train, y_train)
        gradient_boosting = random_search.best_estimator_
        setx['Params']['BestParamsRandSCV'] = random_search.best_params_

    else:
        # Parameter default
        default_params_model = {'random_state': 42}
        params_model = {**default_params_model, **setx['Params'].get('modelParams', {})}
        setx['Params']['modelParams'] = params_model
        gradient_boosting = GradientBoostingClassifier(**params_model)


    # Training & Prediction
    gradient_boosting.fit(X_train, y_train)
    y_pred_train_gbc = gradient_boosting.predict(X_train)
    y_pred_test_gbc = gradient_boosting.predict(X_test)

    # Model evaluation
    setx['AccuracyTrain'] = round(accuracy_score(y_train, y_pred_train_gbc), 8)
    setx['AccuracyTest'] = round(accuracy_score(y_test, y_pred_test_gbc), 8)
    setx['PrecisionTest'] = round(precision_score(y_test, y_pred_test_gbc, average='weighted'), 8)
    setx['RecallTest'] = round(recall_score(y_test, y_pred_test_gbc, average='weighted'), 8)
    setx['F1ScoreTest'] = round(f1_score(y_test, y_pred_test_gbc, average='weighted'), 8)

    # Show results if prompted
    if resultPrint:
        print(f"Gradient Boosting ({extFitType}{', SMOTE' if use_smote else ''})")
        print(f'\tAccuracy Train: {setx["AccuracyTrain"]}')
        print(f'\tAccuracy Test: {setx["AccuracyTest"]}')
        print(f'\tPrecision Test: {setx["PrecisionTest"]}')
        print(f'\tRecall Test: {setx["RecallTest"]}')
        print(f'\tF1 Score Test: {setx["F1ScoreTest"]}\n')

    return setx, gradient_boosting