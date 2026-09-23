# mt-5/notebook/Notebook_02_preprocessing.ipynb — controle visuel du tokenizer
#
# Avant de tokeniser tout le dataset, nous avons voulu voir concretement ce que
# le tokenizer SentencePiece de mT5 fait d'un exemple reel : texte d'origine,
# tokens obtenus, puis identifiants numeriques. Ce controle nous a permis de
# reperer que les caracteres speciaux de l'Akan (Ɔ, ɛ) sont souvent decoupes en
# sous-tokens separes du reste du mot — signe que ces caracteres sont
# sous-representes dans le vocabulaire pre-entraine, un phenomene que nous avons
# retrouve plus tard, encore plus marque, sur l'amharique (alphabet Ge'ez).

MODEL_NAME = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

example = train.iloc[0]["input"]

print("Texte original :")
print(example)

tokens = tokenizer.tokenize(example)
print("\nTokens :")
print(tokens)

token_ids = tokenizer(example)["input_ids"]
print("\nIDs :")
print(token_ids)
