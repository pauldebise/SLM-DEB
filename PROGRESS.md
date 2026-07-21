# Progress log

Format : append un bloc en haut à chaque session, ne réécris jamais
l'historique existant.

---

## Session 2026-07-21 (run 2) — Phase 9 validation + bug fixes

Statut : **Pipeline complet vérifié bout en bout.** Phase 9 prête à lancer.

### Fait

- **Correction CUDA graphs** : `torch.compile(mode="reduce-overhead")` incompatible
  avec weight tying + gradient accumulation → fallback vers `mode="default"` dans
  `src/train.py` après warmup test.
- **Correction batch size** : hardware_detect.py générait micro_batch=32 → OOM.
  Corrigé : formule basée sur `vram * 0.4` (donne 9 sur 24GB). Ajout du
  paramètre `compile_mode` dans la config hardware.
- **Correction checkpoint `_orig_mod.`** : `torch.compile` ajoute ce préfixe dans
  le state_dict → `_strip_compile_prefix()` dans `save_checkpoint()`.
- **Correction eval.py** : fallback vers split `train` si `val_shards` vide.
- **Vérification bout en bout** :
  - Tokenizer (2k vocab test, round-trip OK — FR accents, Python, chat)
  - Preprocessing (3 sources, 500k tokens, shards binaires uint16)
  - Training 15 steps (loss 110 → 81 décroissante, pas de crash)
  - Resume OK (step 16 → 25, loss 81 → 67)
  - Eval OK (val loss 5.08, ppl 161.79 sur 15-step checkpoint)
  - GUI OK (charge checkpoint, génération fonctionnelle)
- **Benchmark** : 64,610 tokens/sec, 37% MFU, 8.2 GB VRAM (bs=8, compile default)

### En cours

- Rien — tout est prêt pour le lancement du run réel.

### Prochain jalon précis

1. Lancer `bash scripts/launch_training.sh 300m` (entraîne tokenizer 32k sur
   ~300k échantillons, pré-tokenize ~12B tokens, puis entraînement complet).
2. Surveiller avec `tensorboard --logdir logs/`.
3. Vérifier la reprise après interruption simulée (kill puis relaunch avec
   `--resume`).

### Blocages / questions ouvertes

- `bigcode/the-stack-dedup` toujours gated (nécessite auth HF).
  `ise-uiuc/Magicoder-OSS-Instruct-75K` fonctionne comme alternative Python.
- `torch.compile(mode="reduce-overhead")` (CUDA graphs) incompatible avec le
  weight tying dans le contexte de gradient accumulation. À investiguer plus
  tard si le gain de ~27% vaut un refactoring du modèle.

---

## Session 2026-07-21 — Agent run (final status)

Statut : **Phases 0-8 + 10 complete.** Phase 9 (first real run) documented
and ready to launch, but actual training not started (requires data preparation
in background).

### Fait

- **Phase 0** — Bootstrap : arborescence (`configs/`, `src/`, `gui/`, `scripts/`),
  `.gitignore` ancré à la racine, `README.md`, `setup_pod.sh`. Commit 8672935.
- **Phase 1** — Détection matériel : `src/hardware_detect.py` génère
  `configs/hardware/auto.yaml` avec micro_batch, accumulation, précision (bf16),
  workers, gradient checkpointing. Support overlay YAML. Commit 83544ba.
- **Phase 2** — Tokenizer : `src/tokenizer/train_tokenizer.py` entraîne un BPE
  ByteLevel (32k vocab ajustable) sur fineweb-edu + Magicoder Python + smoltalk.
  Round-trip vérifié (EN, FR, Python indentation, chat). Commit 3a623e2.
- **Phase 3** — Pipeline données : `configs/data/mixture.yaml` (60% texte / 25%
  code / 15% chat), `src/data/download.py`, `src/data/preprocess.py` (shards
  binaires uint16 avec split train/val), `src/data/dataset.py` (IterableDataset
  memory-mapped, DDP-aware). Commit 9a85ec2.
- **Phase 4** — Architecture : RMSNorm, RoPE, SDPA, SwiGLU, weight tying dans
  `src/model/layers.py` et `src/model/transformer.py`. `sizing_search.py` trouve
  les dimensions pour 100M/300M/800M à ±0.35%. Overfit test passé (loss 10.4 →
  0.003). Commit 5858226.
- **Phase 5+6** — Entraînement + TensorBoard : `src/train.py` avec AdamW (fused),
  warmup+cosine decay, bf16 AMP, gradient clipping, DDP, reprise sur crash
  (modèle + optimiseur + scheduler + step + RNG), checkpointing avec rétention.
  TensorBoard : loss, ppl, lr, grad_norm, tokens/s, step_time, GPU mem.
  Vérifié en entraînement 15 steps : loss 38 → 28. Resume OK. Commit 1ea43d5.
- **Phase 7** — Optimisation : `src/benchmark.py` micro-benchmark.
  torch.compile : +26.7% tokens/sec (38.5k → 48.8k). Batch scaling : 56.4k à
  bs=8, MFU 32.3%. Documenté dans `BENCHMARKS.md`. Commit 8719b3d.
- **Phase 8** — GUI : `gui/app.py` (Gradio) — liste les checkpoints, charge
  depuis la config embarquée, génération (temperature, top-k, top-p). Commit f255c32.
- **Phase 10** — Documentation : `README.md` à jour, `eval.py`, script de
  lancement `scripts/launch_training.sh`.

### En cours

- Phase 9 : premier run réel 300M (~12B tokens) — **pas encore lancé**.
  Le script `scripts/launch_training.sh 300m` est prêt. Il faut d'abord
  entraîner le tokenizer complet (32k vocab, ~300k échantillons) et
  pré-tokenizer les données — opérations longues à lancer en arrière-plan.

### Prochain jalon précis

1. Exécuter `bash scripts/launch_training.sh 300m` pour lancer l'entraînement
   complet. Ceci va :
   - Générer `configs/hardware/auto.yaml`
   - Entraîner le tokenizer 32k sur ~300k échantillons (~30-60 min)
   - Pré-tokenizer les données (~1-2h selon le volume)
   - Lancer l'entraînement en tmux (~50k-58k tokens/sec → ~2.5 jours pour 12B)
2. Surveiller avec `tensorboard --logdir logs/`
3. Utiliser `python3 gui/app.py` pour générer depuis les checkpoints

### Blocages / questions ouvertes

- `bigcode/the-stack-dedup` est gated (nécessite auth HF). Alternative :
  `ise-uiuc/Magicoder-OSS-Instruct-75K` (Python solutions). Pour plus de code
  diversifié, il faudrait un token HF configuré.
- `codeparrot/github-code-clean` a un script déprécié (non supporté par les
  nouvelles versions de `datasets`).

---

## [Template initial] Session 1 — bootstrap

- Statut : pas encore démarré.
- Fait : —
- En cours : —
- Prochain jalon précis : Phase 0 (bootstrap) de AGENTS.md.
- Blocages / questions ouvertes : —
