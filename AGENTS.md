# Mission : pipeline d'entraînement SLM from-scratch sur RunPod (RTX 4090)

Tu es un agent d'ingénierie ML autonome. Tu travailles sans supervision humaine
directe entre deux relances. À CHAQUE session tu dois : lire ce fichier, lire
`PROGRESS.md` et `BENCHMARKS.md`, reprendre exactement là où tu t'es arrêté,
avancer d'un jalon concret, tester ce que tu as fait, committer, pousser sur
`origin`, puis mettre à jour `PROGRESS.md`/`BENCHMARKS.md` avant de terminer.

## Contraintes non négociables

- **Aucun poids pré-entraîné.** Le tokenizer peut être entraîné par toi sur le
  corpus, mais l'architecture du modèle et tous ses poids doivent être
  initialisés et entraînés depuis zéro. N'importe pas de checkpoint HuggingFace,
  ne charge pas de `state_dict` externe.
- **Scalabilité en taille de modèle.** Toute la config (nombre de couches,
  d_model, têtes, FFN, contexte) doit être paramétrique et pilotée par un
  fichier de config, pas codée en dur. Cible actuelle : ~300M paramètres.
  Les cibles 100M et 800M doivent fonctionner avec le même code, juste une
  config différente.
- **Scalabilité au matériel détecté.** Rien (batch size, accumulation,
  précision, workers, DDP ou non) ne doit être codé en dur pour "1x RTX 4090".
  Tout doit découler d'une détection matérielle au lancement.
- **Commits réguliers.** Un commit par jalon fonctionnel testé, jamais de gros
  commit fourre-tout. Voir section Git.
- **Boucle d'optimisation continue.** Tu ne "finis" jamais d'optimiser tant
  que tu n'as pas explicitement vérifié, à chaque itération, quel est le
  goulot d'étranglement actuel (voir section dédiée). C'est une exigence de
  méthode, pas une option.

## Contexte matériel de départ

- 1x RTX 4090 (24 Go VRAM, Ada Lovelace, tensor cores gen4, bf16 natif,
  pas de NVLink). D'autres GPU (plus nombreux / plus gros) viendront plus
  tard sur d'autres pods : le code doit rester compatible `torchrun`/DDP dès
  maintenant même s'il ne tourne que sur 1 GPU aujourd'hui.
- Volume persistant `/workspace` : 100 Go, prévu pour ~12 milliards de tokens
  d'entraînement pré-tokenizés + checkpoints + logs. Ce volume pourra être
  agrandi plus tard pour des modèles plus gros : ne suppose jamais une taille
  fixe, lis-la dynamiquement (`shutil.disk_usage`) et adapte les rétentions de
  checkpoints en conséquence.

## Arborescence attendue du dépôt

```
slm-trainer/
  AGENTS.md, opencode.json, agent_loop.sh, setup_pod.sh   (déjà fournis)
  README.md                     (à toi de l'écrire et de le tenir à jour)
  PROGRESS.md, BENCHMARKS.md    (à tenir à jour à chaque session)
  configs/
    model/{100m,300m,800m}.yaml
    hardware/auto.yaml          (généré au runtime, pas commité)
    data/mixture.yaml
  src/
    hardware_detect.py
    tokenizer/train_tokenizer.py
    data/{download.py, preprocess.py, dataset.py}
    model/{layers.py, transformer.py, sizing_search.py}
    train.py
    eval.py
    benchmark.py
  gui/app.py
  scripts/ (utilitaires ponctuels : dry-run, smoke tests, sanity checks)
  data/          (gitignored — shards tokenisés binaires)
  checkpoints/   (gitignored)
  logs/          (tensorboard — gitignored ou échantillon seulement)
```

## Phases de travail

Respecte cet ordre. Ne passe à la phase suivante que si la précédente a un
test qui passe. Chaque phase = au moins un commit.

### Phase 0 — Bootstrap
Structure de dossiers, environnement Python, `.gitignore` (exclut `data/`,
`checkpoints/`, gros logs, venv), README initial, premier commit.

### Phase 1 — Détection & scaling matériel
`hardware_detect.py` qui interroge `torch.cuda.get_device_properties`, le
nombre de GPU, la VRAM par GPU, le nombre de CPU, la RAM système, et écrit
`configs/hardware/auto.yaml` avec : batch size micro, gradient accumulation
steps (pour viser un batch effectif raisonnable, ex. ~0.5M tokens/step),
précision (bf16 si supporté sinon fp16), nombre de workers dataloader,
activation ou non de la gradient checkpointing, DDP oui/non. Ce fichier doit
être régénéré (pas édité à la main) à chaque lancement, avec possibilité de
forcer des valeurs manuellement en overlay.

### Phase 2 — Tokenizer
Entraîne un tokenizer BPE (32k vocab, ajustable) sur un échantillon
représentatif du mélange de données (texte + code + chat), avec la lib
`tokenizers`. Vérifie le round-trip encode/decode sur des exemples de chaque
domaine (accents français si pertinent, indentation Python, tours de dialogue).

### Phase 3 — Pipeline de données
- Vérifie toi-même la disponibilité actuelle des datasets ci-dessous (les
  identifiants/versions HuggingFace évoluent) avant de t'y fier aveuglément :
  - Texte généraliste type "manuel" : `HuggingFaceFW/fineweb-edu` (config
    échantillon, ex. sample-10BT) et/ou `HuggingFaceTB/cosmopedia`.
  - Code Python : un sous-ensemble Python filtré de `bigcode/the-stack-dedup`
    ou équivalent (`codeparrot/github-code-clean` filtré en langage=Python).
  - Chat / instructions : `HuggingFaceH4/ultrachat_200k`, `OpenAssistant/oasst2`
    ou `HuggingFaceTB/smoltalk`.
  - Si un identifiant est mort/gated/renommé, trouve l'équivalent actuel du
    même type de contenu — l'important est la *composition* (texte éducatif /
    code / chat), pas l'identifiant exact.
- Mélange indicatif à exposer dans `configs/data/mixture.yaml` (ajustable) :
  ~60% texte éducatif, ~25% code Python, ~15% chat.
- Pré-tokenise en shards binaires memory-mappés (uint16 si vocab < 65536) —
  ne tokenize jamais à la volée pendant l'entraînement, c'est un goulot
  d'étranglement classique.
- Réserve ~0.5-1% des tokens en validation, séparée par shard (pas de fuite).
- Calcule et logue le volume disque réel attendu (tokens visés × 2 octets)
  et vérifie via `shutil.disk_usage('/workspace')` qu'il reste de la marge
  pour checkpoints + logs avant de lancer un run complet.

### Phase 4 — Architecture du modèle
Transformer décodeur-only écrit à la main (pas de `transformers.AutoModel`) :
RMSNorm, RoPE, attention via `torch.nn.functional.scaled_dot_product_attention`
(active nativement le backend flash/mem-efficient sans dépendance externe),
MLP SwiGLU, poids d'embedding et de sortie liés (weight tying) pour économiser
des paramètres sur un petit modèle. Écris `sizing_search.py` : étant donné un
nombre de paramètres cible, recherche (n_layers, d_model, n_heads, d_ff) qui
l'atteint à ±3% en respectant `d_model % n_heads == 0`. Formule approximative
à utiliser comme point de départ (embeddings liés) :

```
params ≈ 12 * n_layers * d_model^2 + vocab_size * d_model
```

Points de départ indicatifs à affiner par la recherche (pas des valeurs à
coder en dur) :

| Cible | n_layers | d_model  | n_heads |
|-------|----------|----------|---------|
| 100M  | ~12      | ~768     | 12      |
| 300M  | ~18-20   | ~1024-1152 | 16    |
| 800M  | ~24      | ~1536    | 16-24   |

Fournis les 3 configs (`configs/model/100m.yaml`, `300m.yaml`, `800m.yaml`)
générées par la recherche, avec le nombre de paramètres réel loggé. Longueur
de contexte de départ : 1024 tokens (augmentable plus tard, RoPE s'y prête).

Test de non-régression obligatoire avant de continuer : forward+backward sur
un batch jouet, vérifie que la loss baisse sur quelques dizaines de steps sur
un tout petit sous-ensemble de données (overfit test).

### Phase 5 — Boucle d'entraînement
AdamW (fused si dispo), warmup linéaire + cosine decay, gradient clipping,
bf16 autocast (pas de GradScaler nécessaire en bf16), reprise complète sur
crash (modèle + optimiseur + scheduler + step + RNG state), sauvegarde
périodique + rétention limitée de checkpoints (garde les N derniers + le
meilleur sur val, purge le reste — base la fréquence/rétention sur l'espace
disque réellement disponible, cf. Phase 3). Écris le point d'entrée pour
tourner aussi bien en single-GPU qu'en `torchrun` multi-GPU (DDP), même si
seul le cas 1 GPU est testable aujourd'hui.

### Phase 6 — Observabilité TensorBoard
Log au minimum, à intervalle régulier :
- `train/loss`, `train/perplexity`, `train/lr`, `train/grad_norm`
- `train/tokens_per_sec`, `train/step_time_ms`, `train/mfu` (Model FLOPs
  Utilization estimée par rapport au pic théorique bf16 de la RTX 4090)
- `system/gpu_mem_allocated`, `system/gpu_util`, `system/dataloader_wait_ms`
  (temps passé à attendre un batch — signal direct de goulot data)
- `val/loss`, `val/perplexity` à intervalle plus espacé (implémente-le, ce
  n'est pas fondamentalement plus compliqué qu'un forward sans backward sur
  des batches fixes de validation — ne le saute que si tu rencontres un vrai
  blocage technique, et documente-le dans ce cas dans `PROGRESS.md`)

### Phase 7 — Boucle d'optimisation des performances (répétée en continu)
À CHAQUE itération à partir de cette phase, avant/après toute optimisation :

1. Lance un micro-benchmark reproductible (200-500 steps sur données réelles,
   pas jouet) et mesure : tokens/sec, MFU, % temps GPU compute vs % temps
   d'attente dataloader, mémoire GPU utilisée vs disponible.
2. Diagnostique le goulot dominant avec cette grille de lecture :
   - Attente dataloader élevée → augmente `num_workers`/prefetch, vérifie que
     les shards sont bien memory-mappés et pas re-décompressés à la volée,
     envisage le pinning mémoire (`pin_memory=True`) et `persistent_workers`.
   - GPU compute-bound mais MFU faible → essaie `torch.compile` (mode
     `reduce-overhead` ou `max-autotune`), vérifie `allow_tf32`, vérifie que
     `scaled_dot_product_attention` utilise bien un backend fused (pas le
     fallback "math").
   - Mémoire proche de la limite → active/renforce le gradient checkpointing,
     augmente le gradient accumulation en réduisant le micro-batch, envisage
     un optimiseur 8-bit si pertinent.
   - Beaucoup de petits kernels / overhead de lancement → CUDA graphs via
     `torch.compile(mode="reduce-overhead")`.
3. Applique UN changement à la fois, re-benchmark, garde seulement si ça
   améliore réellement les tokens/sec sans casser la convergence (vérifie la
   loss sur quelques centaines de steps après changement).
4. Consigne dans `BENCHMARKS.md` : date, changement testé, tokens/sec avant/
   après, décision (gardé/rejeté) et pourquoi.
5. Arrête cette boucle d'optimisation micro quand les gains deviennent
   marginaux (< quelques % par tentative) — documente-le et passe au run
   d'entraînement réel, tout en gardant l'œil ouvert : si un futur run change
   de régime (plus gros modèle, plus de GPU), relance ce diagnostic.

### Phase 8 — GUI d'inférence
App Gradio (`gui/app.py`), lancée sur `0.0.0.0:7860` :
- Liste les checkpoints disponibles dans `checkpoints/` (y compris ceux d'un
  entraînement en cours, pas seulement les modèles "finis") avec leur step
  et leur config associée (sauvegardée dans le checkpoint lui-même).
- Charge le modèle depuis la config embarquée dans le checkpoint (pas besoin
  que l'architecture soit redécrite ailleurs).
- Interface de génération : prompt, température, top-p, top-k, longueur max,
  bouton de rafraîchissement de la liste de checkpoints (pour voir les
  nouveaux au fur et à mesure de l'entraînement).

### Phase 9 — Premier run réel (300M, ~12B tokens)
Lance l'entraînement complet du modèle 300M sur le mélange de données complet,
avec tout le monitoring actif. Documente dans `PROGRESS.md` le lancement
(commande exacte, date, step visé) et laisse-le tourner en tâche de fond
(cf. `agent_loop.sh` / tmux) pendant que tu continues, si besoin, à travailler
sur la GUI ou la doc en parallèle — ne bloque pas une session entière à
regarder une barre de progression.

### Phase 10 — Documentation
`README.md` à jour : comment lancer un entraînement pour une taille de
modèle donnée, comment relancer la GUI, comment lire TensorBoard, comment
adapter la config pour un nouveau matériel/volume de stockage.

## Définition de "solution complète et fonctionnelle"

- Les 3 configs (100M/300M/800M) génèrent des modèles dont le nombre réel de
  paramètres est vérifié à ±3% de la cible.
- Le run du 300M sur ~12B tokens est lancé, tourne sans crash, checkpoint et
  reprend correctement après une interruption simulée.
- TensorBoard affiche toutes les métriques de la Phase 6, train et val.
- La GUI charge et fait générer aussi bien un checkpoint en cours qu'un
  modèle "terminé".
- `BENCHMARKS.md` montre une trace claire d'au moins 3-4 itérations
  d'optimisation avec décisions justifiées par des chiffres.
- Tout est commité, poussé, et un `README.md` permet de relancer le tout
  sur un pod neuf sans connaître l'historique du projet.

## Git

- Un commit = un jalon testé. Messages type conventional commits
  (`feat:`, `perf:`, `fix:`, `docs:`, `chore:`).
- Push sur `origin main` après chaque commit (le pod est éphémère, ne laisse
  jamais de travail non poussé plus d'une itération).
- `.gitignore` doit exclure `data/`, `checkpoints/`, `logs/` (sauf peut-être
  un petit exemple), et tout environnement virtuel.

## Continuité inter-sessions

Tu n'as pas de mémoire d'une session à l'autre au-delà de ce dépôt. Avant de
terminer une session, mets impérativement à jour :
- `PROGRESS.md` : où tu en es, ce qui est fait, ce qui est en cours, le
  prochain jalon précis à attaquer.
- `BENCHMARKS.md` : historique des mesures de performance et décisions
  d'optimisation (append-only, ne réécris pas l'historique).

Si tu estimes que la mission est entièrement remplie selon la section
"Définition de solution complète et fonctionnelle" ci-dessus, crée un fichier
vide `DONE` à la racine du dépôt et explique pourquoi dans `PROGRESS.md`.
