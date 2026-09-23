# 03_finetune.ipynb — contournement d'une particularite du checkpoint Gemma E2B
#
# Les projections d'attention de ce checkpoint sont enveloppees dans une classe
# Gemma4ClippableLinear (clipping des activations) que PEFT ne reconnait pas pour y
# attacher un adaptateur LoRA. Nous la deballons vers son Linear4bit interne avant
# de configurer LoRA — ce qui desactive ce clipping pendant l'entrainement.
# A surveiller : si la loss divergeait ou produisait des NaN, ce clipping serait
# probablement necessaire et demanderait une solution plus fine.

def unwrap_clippable_linears(model):
    count = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child.__class__.__name__ == "Gemma4ClippableLinear":
                setattr(module, child_name, child.linear)
                count += 1
    print(f'{count} couches Gemma4ClippableLinear deballees vers Linear4bit')
    return model

model = unwrap_clippable_linears(model)
# 232 couches deballees — la loss d'entrainement a decru normalement,
# le risque signale ne s'est pas materialise sur ce run.
