# TF-IDF/multilingual_health_qa_starter_notebook.ipynb — fine-tuning bonus NLLB
#
# Configuration fournie par le notebook de demarrage pour le fine-tuning
# bonus : Seq2SeqTrainer standard, 3 epochs, batch 8, sur l'ensemble du jeu
# d'entrainement (toutes langues confondues, sans reequilibrage).

FINETUNE_OUTPUT_DIR     = './mt5-finetuned-health-qa'
FINETUNE_EPOCHS         = 3
FINETUNE_BATCH_SIZE     = 8      # Reduce to 4 if you hit OOM errors
FINETUNE_LEARNING_RATE  = 5e-5
FINETUNE_MAX_INPUT_LEN  = 256    # Must match MAX_INPUT_LENGTH used at inference
FINETUNE_MAX_TARGET_LEN = 512    # Must match MAX_OUTPUT_LENGTH used at inference
FINETUNE_VAL_SIZE       = 0.05   # 5% of training data used for validation
