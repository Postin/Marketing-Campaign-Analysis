from mlxtend.frequent_patterns import apriori, association_rules


class Apriori:
    """
    #max_len = Max lenght of apriori n-grams
    """
    def __init__(self, df_assoc, min_support=0.8, max_len=10 ):
        self.frequent_items = apriori(df_assoc, use_colnames=True, min_support=min_support, max_len=max_len + 1)
        self.rules = association_rules(self.frequent_items, metric='lift', min_threshold=1)

    def analyze(self, product, segment):
        target = product + "_" + segment
        results = self.rules[self.rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(
            by='confidence', ascending=False)
        return results

    