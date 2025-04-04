
def train_word2vec(X):
    # Tokenization of each text in the dataset
    X_tokenized = [word_tokenize(str(text).lower()) for text in X]
    return Word2Vec(sentences=X_tokenized, vector_size=100, window=5, min_count=5, workers=4)

def extrFeat(X, y, extFeatType='default', scheme='80/20', dispFeat=False, getReturn=False, word2vec_model=None):

    if extFeatType == 'default':
        if getReturn:
            if scheme == '80/20':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                return X_train, X_test, y_train, y_test
            if scheme == '70/30':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                return X_train, X_test, y_train, y_test

    # Feature extraction with TF-IDF
    if extFeatType == 'tfidf':

        tfidf = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.8 )
        X_tfidf = tfidf.fit_transform(X)

        # Convert feature extraction results to dataframes
        features_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

        # Display feature extraction results
        if dispFeat:
            print('features_tfidf_df:')
            display(features_tfidf_df)

        if getReturn:
            if scheme == '80/20':
                X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
                return X_train, X_test, y_train, y_test

            if scheme == '70/30':
                X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
                return X_train, X_test, y_train, y_test

    # Feature extraction with Word2Vec
    if extFeatType == 'w2vec':

        if word2vec_model is None:
            word2vec_model = train_word2vec(X)

        def vectorize_text(text):
            words = word_tokenize(str(text).lower())
            word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
            return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(word2vec_model.vector_size)

        X_w2vec = np.array([vectorize_text(text) for text in X])
        features_w2v_df = pd.DataFrame(X_w2vec)

        # Display feature extraction results
        if dispFeat:
            print('features_w2v_df:')
            display(features_w2v_df)

        # Divide data into training and test data
        if getReturn:
            if scheme == '80/20':
                X_train, X_test, y_train, y_test = train_test_split(X_w2vec, y, test_size=0.2, random_state=42)
                return X_train, X_test, y_train, y_test

            if scheme == '70/30':
                X_train, X_test, y_train, y_test = train_test_split(X_w2vec, y, test_size=0.3, random_state=42)
                return X_train, X_test, y_train, y_test

    # Feature Extraction with BoW
    if extFeatType == 'bow':
        # Initialize the CountVectorizer with parameters similar to TF-IDF
        bow_vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.8)

        # Transform text into BoW representation
        X_bow = bow_vectorizer.fit_transform(X)

        # Convert feature extraction results to DataFrame
        features_bow_df = pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out())

        # Display feature extraction results
        if dispFeat:
            print('features_bow_df:')
            display(features_bow_df)

        if getReturn:
            if scheme == '80/20':
                X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)
                return X_train, X_test, y_train, y_test

            if scheme == '70/30':
                X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.3, random_state=42)
                return X_train, X_test, y_train, y_test
    return None