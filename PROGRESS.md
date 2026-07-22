# Progress log

Format : append un bloc en haut à chaque session, ne réécris jamais
l'historique existant.

---

## Session 2026-07-22 (run 9) — Robustness fixes + watcher for auto-start

Statut : **Pré-tokenization toujours en cours (~3.7B tokens, 51% text). Fixes commités, watcher en place.**

### Fait

- **Fix crash on exit (`preprocess.py`)** : `datasets` streaming cause un abort
  (PyGILState_Release) lors de la finalisation Python. Remplacé `sys.exit()`
  par `os._exit(0)` pour éviter la teardown problématique. Commit 4d4a7c4.
- **Ajout try/except par source dans preprocess** : si un dataset échoue
  (gate, rate-limit, format), les autres sources continuent et le manifest
  est quand même écrit avec les données disponibles. Plus de perte de
  progression si Magicoder ou Smoltalk est inaccessible.
- **Ajout `--log-file` dans preprocess** : écriture d'un fichier de log
  séparé pour la progression, contournant les problèmes de buffering stdout
  quand le script tourne sans TTY. Launch script mis à jour.
- **Ajout `|| true` dans launch_training.sh** : défense en profondeur pour
  que le script continue vers l'entraînement même si preprocess sort avec
  un code non-zéro.
- **Script watcher (`/tmp/training_watcher.sh`)** : lancé en nohup, surveille
  l'apparition du manifest toutes les 30s. Quand le manifest apparaît, attend
  10s puis démarre l'entraînement si pas déjà lancé. PID 56954.
- **README mis à jour** : corrigé la recommandation `reduce-overhead` (crash
  incompatible avec weight tying + gradient accumulation). Commit e2af9cf.

### En cours

- **Pré-tokenization 12B tokens** : 369 shards text_train (~3.69B tokens, ~51%
  du text source). ~6.9 GB. Tourne depuis 23:33 Jul 21 (~73 min).
  ETA text : ~80-90 min restantes. Puis code (Magicoder, rapide) et chat
  (Smoltalk, rapide). La progression est monitorable via `ls data/shards/`.
- **Watcher** : tourne en fond (PID 56954, log dans `logs/watcher.log`).
  Démarrera l'entraînement automatiquement dès que le manifest apparaît.

### Prochain jalon précis

1. La pré-tokenization se termine → le watcher détecte le manifest →
   l'entraînement démarre automatiquement dans tmux `slm-train-300m`.
2. Vérifier : `tmux attach -t slm-train-300m` ou `bash scripts/status.sh`.
3. Après 1000+ steps : vérifier TensorBoard, checkpoints, reprise.
4. GUI inference depuis les checkpoints réels.
5. Si stable ≥1000 steps → fichier `DONE`.

### Blocages / questions ouvertes

- `bigcode/the-stack-dedup` toujours gated.
- Pas de token HF → rate limits serrés sur les downloads streaming.
- `torch.compile(mode="reduce-overhead")` toujours incompatible avec weight
  tying + gradient accumulation.

---

## Session 2026-07-22 (run 8) — Pipeline verification while pre-tokenization runs

Statut : **Pipeline E2E re-vérifié.** Pré-tokenization toujours en cours.

### Fait

- **Vérification end-to-end complète** avec les shards déjà produits :
  - Création d'un manifest temporaire pointant vers 10 shards train + 1 shard val
  - Entraînement 50 steps (300M, 9×16, 147k tok/step) → checkpoints step_25, best
  - Resume depuis step_25 → checkout de steps 26→40 : loss décroît normalement
    (65 → 58 → 57.9), tokens/sec stable à ~49k
  - Load + génération depuis checkpoint `step_40`: modèle charge correctement,
    génération fonctionnelle
  - Nettoyage : artefacts de test supprimés (checkpoints_test, logs/test_*,
    manifest_test.json)
- **Vérification des 3 configs modèle** : 100M (100,011,072, err 0.01%),
  300M (299,697,920, err 0.10%), 800M (802,796,736, err 0.35%). Toutes ±3%.
- **Espace disque** : 142 TB libre sur /workspace — aucune contrainte.

### En cours

- **Pré-tokenization 12B tokens** : 294 shards / 5.5 GB / ~2.94B tokens (juillet
  22 00:32 UTC). Environ 24.5% fait. Le processus tourne à ~5 shards/min
  (CPU streaming HF rate-limited). Le goulot est le téléchargement (pas de
  token HF).

### Prochain jalon précis

1. La pré-tokenization se termine (estimé ~2-3h restantes pour fineweb-edu,
   puis code+chat plus rapides).
2. Le manifest est créé, l'entraînement démarre automatiquement dans tmux
   `slm-train-300m`.
3. Surveiller avec `bash scripts/status.sh` et `tmux attach -t slm-train-300m`.
4. Vérifier la reprise après interruption simulée sur le run réel.
5. GUI inference depuis les checkpoints du run réel.
6. Si stable ≥1000 steps → fichier `DONE`.

### Blocages / questions ouvertes

- Mêmes blocages que run 7 (bigcode gated, rate limits HF, pas de token HF,
  torch.compile reduce-overhead incompatible avec weight tying + grad acc).

Statut : **Améliorations commités.** Pré-tokenization toujours en cours,
training pas encore démarré.

### Fait

- **Fix output buffering (`preprocess.py`)** : stdout était bufferisé quand
  redirigé vers un fichier de log, rendant la progression invisible. Ajout de
  `sys.stdout.reconfigure(line_buffering=True)`, `flush=True` sur tous les
  `print()`, et `PYTHONUNBUFFERED=1` dans le launch script. Smoke test OK :
  les messages apparaissent maintenant immédiatement dans le log.
- **Ajout ETA/progression dans preprocess** : chaque palier de 10k échantillons
  affiche maintenant le taux de tokens/sec, l'ETA, le nombre de shards écrits.
- **Script `scripts/status.sh`** : surveille l'état du pipeline (shards,
  tokens estimés, GPU, processus, tmux). Fonctionnel.

### En cours

- **Pré-tokenization 12B tokens** : lancée le 21/07 à 23:33 UTC, ~236 shards
  / 4.4 GB / ~2.4B tokens traités (juillet 22:21 UTC). Environ 19.7% fait.
  Estimé ~1.5-2h restantes pour finir le source text (fineweb-edu), puis code
  et chat (plus petits).

### Prochain jalon précis

1. Idem run 6 — la pré-tokenization suit son cours.
2. L'entraînement démarrera automatiquement une fois le manifest créé.
3. Utiliser `bash scripts/status.sh` pour suivre l'état.

### Blocages / questions ouvertes

- `bigcode/the-stack-dedup` toujours gated.
- Pré-tokenization lente (~50M tokens/min sans token HF, mais le goulot est
  le download/streaming HF, pas le CPU).
- Pas de token HF → rate limits serrés.

---

## Session 2026-07-22 (run 6) — Bug fixes + performance

Statut : **Fixes commités (5f369d9), smoke tests passés.** Pré-tokenization
toujours en cours.

### Fait

- **Fix shard distribution bug (`dataset.py`)** : la formule
  `(shard_idx * (worker_id + 1)) % num_workers` était incorrecte et laissait
  50% des workers inactifs. Remplacé par `shard_idx % num_workers`
  (distribution round-robin correcte). Commit 5f369d9.
- **Ajout `persistent_workers`/`prefetch_factor`** : les paramètres étaient
  définis dans la config hardware mais jamais transmis au DataLoader. Ajout
  de `persistent_workers` et passage effectif de `prefetch_factor` et
  `pin_memory` depuis la config. Commit 5f369d9.
- **Vérification des 3 configs modèle** : 100M (100,011,072 params, err
  0.01%), 300M (299,697,920, err 0.10%), 800M (802,796,736, err 0.35%).
  Toutes à ±3% de la cible — conforme.
- **Smoke test complet** : entraînement 5 steps + checkpoint + resume (3 steps
  supplémentaires) + load + génération. Tout fonctionne sans erreur.

### En cours

- **Pré-tokenization 12B tokens** : `nohup bash scripts/launch_training.sh 300m`
  lancé le 21/07 à 23:33 UTC, toujours en cours le 22/07.

### Prochain jalon précis

Idem run 7 — la pré-tokenization suit son cours. Training démarrera
automatiquement.

### Blocages / questions ouvertes

- `bigcode/the-stack-dedup` toujours gated → Magicoder comme alternative Python.
- `torch.compile(mode="reduce-overhead")` (CUDA graphs) incompatible avec
  weight tying + gradient accumulation.
- Pas de token HF → rate limits serrés sur les downloads.

---

## Session 2026-07-21 (run 5) — Bug fixes + E2E re-verification

Statut : **Fixes committed, pipeline re-verified.** Pré-tokenization 12B en cours.
~10% fait (151 shards / 2.9 GB sur ~24 GB estimés).

### Fait

- **Fix dataset shard path resolution** : `_load_shard()` utilisait
  `Path(self.shards[0]["path"]).parent` pour résoudre les chemins relatifs,
  ce qui donnait `"."` quand les chemins dans le manifest étaient relatifs.
  Remplacé par `os.path.dirname(os.path.abspath(manifest_path))` comme base
  de résolution. Commit 1f62328.
- **Fix PPL display** : la perplexité console utilisait `exp(min(accum_loss, 20))`
  où `accum_loss` est la somme sur les micro-batches. Avec 16 steps d'accumulation,
  le PPL était toujours `exp(20) = 485M` tant que la loss > 20/16 = 1.25.
  Correction : `exp(min(accum_loss / accum_steps, 20))`. Le TensorBoard `train/loss`
  logge maintenant la perte moyenne par token (pas la somme). Commit 1f62328.
- **Vérification end-to-end** : entraînement 20 steps (300M, 32k vocab) →
  checkpoints OK (steps 15, 20 + best) → load model OK → génération fonctionnelle
  (basique, modèle entraîné que 20 steps).
  - Resume depuis step 15 → loss continue normalement (90 → 51 sur 30 steps)
  - Validation loss 6.33 → 4.55
  - Throughput : ~49k tokens/sec après warmup compile
  - GUI : load checkpoint OK, génération OK

### En cours

- **Pré-tokenization 12B tokens** : `nohup bash scripts/launch_training.sh 300m`
  lancé le 21/07 à 23:33 UTC. Progression : 151 shards / 2.9 GB (~10%, ~30 min).
  Estimation totale ~4h. L'entraînement 300M démarrera automatiquement dans
  tmux `slm-train-300m` une fois la pré-tokenization terminée.

### Prochain jalon précis

1. Attendre la fin de la pré-tokenization (~3.5h restantes estimé)
2. L'entraînement démarre automatiquement dans tmux `slm-train-300m`
3. Surveiller : `tmux attach -t slm-train-300m`, `tensorboard --logdir logs/ --bind_all`
4. Vérifier la reprise après interruption simulée (kill + `--resume`)
5. GUI inference depuis les checkpoints du run réel
6. Si stable ≥1000 steps → fichier `DONE`

### Blocages / questions ouvertes

- `bigcode/the-stack-dedup` toujours gated → Magicoder comme alternative Python.
- Pré-tokenization lente (~50M tokens/min sans token HF). ~4h pour 12B.
- `torch.compile(mode="reduce-overhead")` (CUDA graphs) incompatible avec
  weight tying + gradient accumulation.
- Pas de token HF → rate limits serrés sur les downloads.

---

## Session 2026-07-21 (run 4) — End-to-end verification + relaunch

Statut : **Pipeline end-to-end vérifié.** Entraînement 300M relancé avec 12B tokens.

### Fait

- **Vérification end-to-end** : preprocessing 50M tokens → entraînement 200 steps →
  checkpoints OK (steps 50, 100, 150) → resume OK (depuis step 50, loss continue
  normalement 13.3 → 3.6). Commit (à venir).
- **Fix launch script** : ajout du paramètre `--total-tokens` (supporte suffixes B/M/K
  et décimaux) pour contrôler le volume de données pré-tokenizées. Conversion
  Python (pas sed) pour robustesse. Commit (à venir).
- **Correction précédente** : le `TOTAL_TOKENS_NUM` utilisait un sed bogué
  (`1.5B` → 15B au lieu de 1.5B). Remplacé par conversion Python.
- **Relance full run 12B** : `nohup bash scripts/launch_training.sh 300m` lancé.
  Preprocessing en cours (~4h estimé), puis entraînement en tmux `slm-train-300m`.

### Métriques vérification

- Preprocessing 50M tokens : < 2 min
- Entraînement 200 steps (300M params, bs=9×16, 147k tokens/step) :
  - Throughput : ~49k tokens/sec
  - Loss : 126 → 1.5 (step 100), ~25.7M tokens consommés
  - MFU estimé : ~28%
  - GPU mem : stable (gradient checkpointing actif)
  - Checkpoints : 4 (steps 50, 100, 150 + best)
- Resume depuis step 50 : loss continue à 13.3, décroît normalement vers 3.6

### En cours

- Pré-tokenization 12B tokens en arrière-plan (nohup). Une fois terminé,
  l'entraînement démarrera automatiquement dans tmux `slm-train-300m`.

### Prochain jalon précis

1. Surveiller la pré-tokenization : `tail -f logs/launch_*.log`
2. Une fois l'entraînement lancé : `tmux attach -t slm-train-300m`
3. TensorBoard : `tensorboard --logdir logs/ --bind_all`
4. Vérifier la reprise après interruption simulée sur le vrai run
5. GUI inference depuis les checkpoints du run réel
6. Si tout est stable pendant ≥1000 steps : considérer la mission comme complète
   selon la définition "solution complète et fonctionnelle"

### Blocages / questions ouvertes

- `bigcode/the-stack-dedup` toujours gated → Magicoder comme alternative Python.
- Pré-tokenization lente (~50M tokens/min sans token HF). ~4h pour 12B.
- `torch.compile(mode="reduce-overhead")` incompatible avec weight tying + gradient acc.
- Pas de token HF → rate limits serrés sur les downloads (re-téléchargements depuis 0
  si interruption).

---

## Session 2026-07-21 (run 3) — Phase 9 launch + bug fixes

Statut : **Phase 9 lancée.** Tokenizer 32k entraîné, pré-tokenization en cours,
entraînement 300M démarrera automatiquement dans tmux.

### Fait

- **Correction checkpoint resume** : `_strip_compile_prefix()` retirait le préfixe
  `_orig_mod.` au save, mais ne l'ajoutait pas au load → crash au resume.
  Ajout de `_add_orig_mod_prefix()` dans `load_checkpoint()`. Commit f239fe3.
- **Correction gradient checkpointing** : `gradient_checkpointing: true` dans la
  config hardware n'était jamais passé à `model.forward()`. Ajout du paramètre
  `use_checkpoint` dans la boucle d'entraînement. Commit be93fc9.
- **Vérification bout en bout avant lancement** :
  - Tokenizer 4k vocab smoke test (round-trip OK)
  - Preprocessing 500k tokens (3 shards)
  - Training 10+5 steps (loss 117 → 95 → 83, pas de crash)
  - Resume OK (step 11 → 15, loss 95 → 83)
  - GUI OK (charge checkpoint, génération fonctionnelle)
- **Lancement Phase 9** : `nohup bash scripts/launch_training.sh 300m` en arrière-plan
  le 2026-07-21 à 22:53 UTC. Le tokenizer 32k a été entraîné avec succès
  (297k échantillons — round-trip OK). La pré-tokenization est en cours
  (~30 shards/300M tokens en ~5 min, estimation ~5h pour 12B).

### En cours

- Pré-tokenization des données (~12B tokens estimés) — le script continue en
  arrière-plan. Une fois terminé, l'entraînement 300M se lancera automatiquement
  dans une session tmux `slm-train-300m`.

### Prochain jalon précis

1. Surveiller la pré-tokenization : `tail -f logs/launch_*.log`
2. Une fois l'entraînement lancé : `tmux attach -t slm-train-300m`
3. TensorBoard : `tensorboard --logdir logs/ --bind_all`
4. Vérifier la reprise après interruption simulée (kill puis relaunch avec `--resume`)
5. GUI inference depuis les checkpoints : `python3 gui/app.py`

### Blocages / questions ouvertes

- `bigcode/the-stack-dedup` toujours gated → Magicoder comme alternative Python.
- Pré-tokenization lente (~50M tokens/min estimé, ~4h pour 12B) sans token HF.
  Améliorable avec un token HF pour des rate limits plus élevés.
- `torch.compile(mode="reduce-overhead")` (CUDA graphs) toujours incompatible avec
  weight tying + gradient accumulation.

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
