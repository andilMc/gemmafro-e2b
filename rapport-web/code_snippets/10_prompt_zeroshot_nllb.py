# TF-IDF/multilingual_health_qa_starter_notebook.ipynb — baseline LLM zero-shot
#
# C'est ce build_prompt() qui explique le score quasi nul du zero-shot : la
# version avec instruction de tache et nom de langue est commentee, et la
# fonction ne fait au final que renvoyer la question brute. Le modele ne recoit
# donc aucune indication qu'il doit repondre a une question de sante, ni dans
# quelle langue — NLLB, un modele de traduction, traduit alors la question au
# lieu d'y repondre.

def build_prompt(question: str, language: str = None) -> str:
    # if language:
    #     lang_name = subset_to_language_name(language)
    #     return f'Answer this health question in {lang_name}: {question}'
    # return f'Answer this health question: {question}'
    return str(question).strip()
