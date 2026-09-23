# 02_build_dataset.ipynb — sur-echantillonnage plafonne des langues minoritaires
#
# La langue majoritaire (Eng_Uga, ~7 600 lignes) pese plus de 4 fois la langue la
# plus rare (Amh_Eth, ~1 800 lignes). Un reequilibrage total dupliquerait
# l'amharique plus de 4 fois, ce qui risque de faire memoriser des reponses plutot
# que generaliser. Nous plafonnons donc le sur-echantillonnage a 3 fois la taille
# d'origine de chaque langue : un compromis assume entre reduire le desequilibre
# et limiter ce risque de memorisation.

MAX_OVERSAMPLE_FACTOR = 3
majority_count = train_split['subset'].value_counts().max()

def rebalance(df, max_factor, majority_count, seed):
    parts = []
    for lang, group in df.groupby('subset'):
        target = min(majority_count, len(group) * max_factor)
        if target > len(group):
            extra = group.sample(target - len(group), replace=True, random_state=seed)
            group = pd.concat([group, extra], ignore_index=True)
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

train_balanced = rebalance(train_split, MAX_OVERSAMPLE_FACTOR, majority_count, SEED)
# Les 3 langues les plus minoritaires atteignent le plafond de x3 et restent donc,
# malgre le reequilibrage, en dessous du niveau de la langue majoritaire.
