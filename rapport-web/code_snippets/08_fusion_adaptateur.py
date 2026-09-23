# 06_merge_adapter.ipynb — fusion de l'adaptateur LoRA en modele autonome
#
# Jusqu'ici, chaque notebook devait retelecharger le modele de base depuis
# Hugging Face, le recharger en 4 bits, deballer les 232 couches
# Gemma4ClippableLinear, puis rattacher l'adaptateur LoRA — une sequence lourde a
# repeter a chaque session. Nous fusionnons desormais l'adaptateur directement
# dans les poids du modele de base, en pleine precision (bf16, non quantifie,
# necessaire pour une addition mathematiquement propre des poids).

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0},
)
base_model = unwrap_clippable_linears(base_model)

peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
merged_model = peft_model.merge_and_unload()   # addition des poids LoRA, une fois pour toutes

merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
# Operation gourmande en memoire : necessite un GPU A100, le L4 (24 Go) etant
# insuffisant pour charger le modele complet en bf16 non quantifie.
