# 03_finetune.ipynb — masquage de la question dans le calcul de la perte
#
# full_text = prompt + reponse ; prompt_text = prompt seul. Nous tokenisons les
# deux et masquons (label = -100) les tokens du prompt dans full_text, pour que la
# perte ne porte que sur la reponse generee — sans dependre du texte exact des
# tokens speciaux du template (qui s'est avere etre <|turn>...<turn|> pour ce
# checkpoint, different du <start_of_turn> d'autres versions de Gemma).

class PromptMaskingCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        full_texts = [ex['full_text'] for ex in batch]
        prompt_texts = [ex['prompt_text'] for ex in batch]

        enc = self.tokenizer(
            full_texts, add_special_tokens=False, truncation=True,
            max_length=self.max_length, padding=True, return_tensors='pt',
        )
        labels = enc['input_ids'].clone()

        for i, prompt in enumerate(prompt_texts):
            prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
            prompt_len = min(prompt_len, labels.shape[1])
            labels[i, :prompt_len] = -100
        labels[enc['attention_mask'] == 0] = -100

        return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': labels}
