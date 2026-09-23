# 03_finetune.ipynb — chargement du modele de base en 4 bits (QLoRA)
#
# Un fine-tuning complet de Gemma E2B (5,13 milliards de parametres) est hors de
# portee du GPU gratuit/Colab dont nous disposons. Nous quantifions donc le modele
# de base en 4 bits (NF4) avant d'y greffer des adaptateurs LoRA entraines, eux, en
# pleine precision.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # format recommande pour des poids de reseau de neurones
    bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 : natif sur L4/A100 (Ada Lovelace/Ampere)
    bnb_4bit_use_double_quant=True,         # quantifie aussi les constantes de quantification
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},   # tout sur le GPU unique, sans offload CPU
)

# Resultat mesure : ~6,7 Go de VRAM occupee pour un modele de 5,1 milliards de
# parametres — sans cette quantification, le chargement seul aurait echoue.
