# 03_finetune.ipynb — configuration des adaptateurs LoRA
#
# Nous gelons entierement le modele de base et n'entrainons que de petites matrices
# ajoutees sur les couches d'attention et du MLP. Nous avons retenu r=16 : une valeur
# usuelle dans la litterature LoRA, qui offre un compromis raisonnable entre capacite
# d'adaptation et nombre de parametres reellement entraines.

for param in model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Resultat obtenu :
#   parametres entrainables : 29 859 840
#   parametres totaux       : 5 134 157 344
#   proportion entrainable  : 0,58 %
