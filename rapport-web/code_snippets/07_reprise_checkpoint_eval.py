# 04_evaluate.ipynb — reprise automatique apres deconnexion Colab
#
# Un premier run de plusieurs heures s'est arrete apres 4 000+ generations, toutes
# perdues : rien n'etait ecrit sur disque avant la toute fin du notebook. Nous avons
# corrige cela en sauvegardant chaque batch au fur et a mesure sur Drive, avec
# reprise automatique au redemarrage — plus jamais de run perdu en cours de route.

@torch.no_grad()
def generate_answers(df_eval, col_name, checkpoint_path, batch_size=32, max_new_tokens=400):
    checkpoint_data = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                checkpoint_data[item['ID']] = item['pred']
        print(f"{len(checkpoint_data)} predictions deja chargees depuis le checkpoint.")

    ids = df_eval['ID'].tolist()
    to_generate = [i for i, doc_id in enumerate(ids) if doc_id not in checkpoint_data]

    with open(checkpoint_path, 'a', encoding='utf-8') as f:
        for i in range(0, len(to_generate), batch_size):
            batch_idx = to_generate[i:i + batch_size]
            # ... generation du batch ...
            for doc_id, pred_text in zip(batch_ids, decoded):
                checkpoint_data[doc_id] = pred_text
                f.write(json.dumps({'ID': doc_id, 'pred': pred_text}, ensure_ascii=False) + '\n')
            f.flush()  # ecriture physique immediate sur Drive, pas seulement en memoire

    return [checkpoint_data[doc_id] for doc_id in ids]
