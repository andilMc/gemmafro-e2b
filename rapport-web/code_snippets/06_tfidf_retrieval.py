# TF-IDF/multilingual_health_qa_starter_notebook.ipynb — baseline retrieval
#
# Pour chaque question de test, nous cherchons la question d'entrainement la plus
# proche par similarite cosinus sur des n-grammes de CARACTERES (et non de mots) :
# ce choix fonctionne independamment du systeme d'ecriture (latin ou guèze pour
# l'amharique), sans tokenisation specifique a chaque langue. Un modele est
# entraine separement pour chacune des 8 langues, avec un modele global de repli.

class TfidfRetrievalAnswerer:
    def __init__(self, question_col, answer_col, group_col=None,
                 ngram_range=(3, 5), max_features=200_000):
        self.question_col, self.answer_col = question_col, answer_col
        self.group_col = group_col
        self.ngram_range, self.max_features = ngram_range, max_features
        self.models, self.global_model = {}, None

    def _fit_single(self, df):
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=self.ngram_range,
            min_df=1, max_features=self.max_features,
            lowercase=False,   # preserve la casse, utile pour les ecritures non latines
        )
        X = vectorizer.fit_transform(df[self.question_col])
        nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(X)
        return {'vectorizer': vectorizer, 'nn': nn, 'answers': df[self.answer_col].values}

    def fit(self, df):
        self.global_model = self._fit_single(df)
        if self.group_col:
            for group, sub in df.groupby(self.group_col):
                if len(sub) >= 2:
                    self.models[group] = self._fit_single(sub)
        return self
