# Random Forest (RF)

from sklearn.ensemble import RandomForestClassifier

def runRandomForest(setxi, X, y, resultPrint=True):

    # Load config set
    setx = setxi.copy()
    setx['ModelName'] = 'Random Forest'

    scheme = setx['options'].get('scheme')
    extFitType = setx['options'].get('extFitType', 'default')
    use_smote = setx['options'].get('smote', False)
    scoring_metric = setx['options'].get('scoring', 'accuracy')

    setx.setdefault('Params', {})

    # Ekstraksi fitur
    X_train, X_test, y_train, y_test = extrFeat(X, y, extFeatType=extFitType, scheme=scheme, getReturn=True)

    # Terapkan SMOTE jika diperlukan untuk menangani data tidak seimbang
    if use_smote:
        smote_obj = SMOTE(sampling_strategy="auto", random_state=42)
        X_train, y_train = smote_obj.fit_resample(X_train, y_train)

    # Konversi
    if extFitType == 'tfidf' and hasattr(X_train, "toarray"):
        X_train, X_test = X_train.toarray(), X_test.toarray()
    if extFitType == 'w2vec' and isinstance(X_train, list):
        X_train, X_test = np.array(X_train), np.array(X_test)


    # Pemilihan model & parameter
    if setx['options'].get('gridSCV', False):
        params_grid = setx['Params'].get('gridParams', {
            'n_estimators': [100, 300, 500],
            'max_depth': [20, 30, None],
            'min_samples_split': [10, 15],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2', 0.5]
        })
        rf = RandomForestClassifier(random_state=42, n_jobs= 2)
        grid_search = GridSearchCV(
            rf, params_grid, cv=3, scoring=scoring_metric, verbose=1, n_jobs=1
        )
        grid_search.fit(X_train, y_train)
        random_forest = grid_search.best_estimator_
        setx['Params']['BestParamsGridSCV'] = grid_search.best_params_

    elif setx['options'].get('randSCV', False):
        params_randSCV = setx['Params'].get('randSCVParams', {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        })
        rf = RandomForestClassifier(random_state=42, n_jobs= 2)
        random_search = RandomizedSearchCV(
            rf, params_randSCV, n_iter=5, cv=3, scoring=scoring_metric, random_state=42, verbose=1, n_jobs=1
        )
        random_search.fit(X_train, y_train)
        random_forest = random_search.best_estimator_
        setx['Params']['BestParamsRandSCV'] = random_search.best_params_

    else:
        # Parameter default
        default_params_model = {'n_jobs': 2}
        params_model = {**default_params_model, **setx['Params'].get('modelParams', {})}
        setx['Params']['modelParams'] = params_model
        random_forest = RandomForestClassifier(**params_model)



    # Training & prediksi
    random_forest.fit(X_train, y_train)
    y_pred_train_rf = random_forest.predict(X_train)
    y_pred_test_rf = random_forest.predict(X_test)

    # Evaluasi model
    setx['AccuracyTrain'] = round(accuracy_score(y_train, y_pred_train_rf), 8)
    setx['AccuracyTest'] = round(accuracy_score(y_test, y_pred_test_rf), 8)
    setx['PrecisionTest'] = round(precision_score(y_test, y_pred_test_rf, average='weighted'), 8)
    setx['RecallTest'] = round(recall_score(y_test, y_pred_test_rf, average='weighted'), 8)
    setx['F1ScoreTest'] = round(f1_score(y_test, y_pred_test_rf, average='weighted'), 8)

    # Tampilkan hasil jika diperlukan
    if resultPrint:
        print(f"Random Forest ({extFitType}{', SMOTE' if use_smote else ''})")
        print(f'\taccuracy_train: {setx["AccuracyTrain"]}')
        print(f'\taccuracy_test: {setx["AccuracyTest"]}')
        print(f'\tprecision_test: {setx["PrecisionTest"]}')
        print(f'\trecall_test: {setx["RecallTest"]}')
        print(f'\tf1_score_test: {setx["F1ScoreTest"]}\n')

    return setx, random_forest