# Naive Bayes (NB)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

def runNaiveBayes(setxi, X, y, resultPrint=False):
    
    # load config set
    setx = setxi.copy()
    setx['ModelName'] = 'Naive Bayes'

    scheme = setx['options'].get('scheme')
    extFitType = setx['options'].get('extFitType', 'default')
    use_smote = setx['options'].get('smote', False)
    setx.setdefault('Params', {})

    # Feature extraction
    X_train, X_test, y_train, y_test = extrFeat(X,y, extFeatType=extFitType, scheme=scheme, getReturn=True)

    # Implement SMOTE if needed to deal with unbalanced data
    if use_smote:
        smote_obj = SMOTE(sampling_strategy="auto", random_state=42)
        X_train, y_train = smote_obj.fit_resample(X_train, y_train)

    # Conversion
    if extFitType == 'tfidf' and hasattr(X_train, "toarray"):
        X_train, X_test = X_train.toarray(), X_test.toarray()
    if extFitType == 'w2vec' and isinstance(X_train, list):
        X_train, X_test = np.array(X_train), np.array(X_test)

    # Model selection
    model_type = setx['options'].get('ModelType')
    if model_type == "bernoulli":
        model_class = BernoulliNB
    elif model_type == "multinomial":
        model_class = MultinomialNB
    else:
        raise ValueError("Model Type harus 'bernoulli' atau 'multinomial'")

    # parameter & model
    if setx['options'].get('gridSCV', False):
        params_grid = setx['Params'].get('gridParams', {'alpha': [0.01, 0.05, 0.1, 0.5, 1.0]})
        grid_search = GridSearchCV(model_class(), params_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=2)
        grid_search.fit(X_train, y_train)
        naive_bayes = grid_search.best_estimator_
        setx['Params']['BestParamsGridSCV'] = grid_search.best_params_
    else:
        default_params_model = {'alpha': 0.1}
        params_model = {**default_params_model, **setx['Params'].get('modelParams', {})}
        setx['Params']['modelParams'] = params_model

        naive_bayes = model_class(**params_model)

    # Training model
    naive_bayes.fit(X_train, y_train)

    # Predictions
    y_pred_train = naive_bayes.predict(X_train)
    y_pred_test = naive_bayes.predict(X_test)

    # Evaluation
    setx['AccuracyTrain'] = round(accuracy_score(y_train, y_pred_train), 8)
    setx['AccuracyTest'] = round(accuracy_score(y_test, y_pred_test), 8)
    setx['PrecisionTest'] = round(precision_score(y_test, y_pred_test, average='weighted'), 8)
    setx['RecallTest'] = round(recall_score(y_test, y_pred_test, average='weighted'), 8)
    setx['F1ScoreTest'] = round(f1_score(y_test, y_pred_test, average='weighted'), 8)

    # Print results if needed
    if resultPrint:
        print(f"Naive Bayes ({model_type}{', SMOTE' if use_smote else ''})")
        print(f"\taccuracy_train: {setx['AccuracyTrain']}")
        print(f"\taccuracy_test: {setx['AccuracyTest']}")
        print(f"\tprecision_test: {setx['PrecisionTest']}")
        print(f"\trecall_test: {setx['RecallTest']}")
        print(f"\tf1_score_test: {setx['F1ScoreTest']}\n")

    return setx, naive_bayes
