"""Name vectors: a pretrained text model reads each name in
the game cache with its examine text, so a name never trained on ("Yew", "Maple tree")
lands near related ones instead of on an untrained, empty embedding.

    bun scripts/cache_names.ts logs/names/cache-names.json
    python scripts/name_vectors.py logs/names/cache-names.json logs/names/name-vectors.npz

Needs `transformers` and the sentence-transformers/all-MiniLM-L6-v2 weights (about 90 MB),
deliberately outside requirements.txt: it runs once, offline. Mean pooling over tokens,
then unit length, as sentence-transformers computes that model. The output is
Jagex-derived data: keep it under logs/, never in git.
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def texts(rows):
    """One text per kind and name: the name, then its most common examine text."""
    descs = defaultdict(Counter)
    for row in rows:
        descs[(row['kind'], row['name'].strip().lower())][(row.get('desc') or '').strip()] += 1
    keys = sorted(descs)
    return keys, [f"{name}. {descs[(kind, name)].most_common(1)[0][0]}".strip() for kind, name in keys]


def embed(sentences, batch=256):
    from transformers import AutoModel, AutoTokenizer
    tokenizer, model = AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            tokens = tokenizer(sentences[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors='pt')
            hidden = model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    return np.concatenate(out)


def main():
    source, target = sys.argv[1], sys.argv[2]
    keys, sentences = texts(json.load(open(source)))
    vectors = embed(sentences)
    np.savez_compressed(target, kind=np.array([k for k, _ in keys]), name=np.array([n for _, n in keys]),
                        vector=vectors.astype(np.float16), model=MODEL)
    print(f'{len(keys)} names, {vectors.shape[1]} dimensions -> {target}')
    index = {key: i for i, key in enumerate(keys)}
    for kind, name in (('loc', 'yew'), ('loc', 'tree'), ('loc', 'rocks'), ('npc', 'jail guard'), ('obj', 'logs')):
        if (kind, name) not in index: continue
        near = np.argsort(-(vectors @ vectors[index[(kind, name)]]))[1:6]
        print(f'  {kind}:{name} ->', ', '.join(f'{keys[i][0]}:{keys[i][1]}' for i in near))


if __name__ == '__main__':
    main()
