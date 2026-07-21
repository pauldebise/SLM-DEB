# Benchmarks & décisions d'optimisation

Append-only : chaque entrée date un micro-benchmark, le changement testé,
les chiffres avant/après, et la décision (gardé/rejeté) + justification.
Voir Phase 7 de AGENTS.md pour la méthode.

---

## [À REMPLIR PAR L'AGENT] Entrée template

- Date :
- Contexte matériel (depuis `configs/hardware/auto.yaml`) :
- Changement testé :
- Tokens/sec avant → après :
- MFU avant → après :
- Goulot identifié :
- Décision : gardé / rejeté — pourquoi :

---

## Session 2026-07-21 — Iteration 1: torch.compile

- Date : 2026-07-21
- Contexte matériel : 1× RTX 4090 (23.5 GB VRAM), bf16, TF32 activé
- Changement testé : `torch.compile(mode="reduce-overhead")` sur le modèle 300M, bs=4
- Tokens/sec avant → après : 38,482 → 48,756 (+26.7%)
- MFU avant → après : 22.03% → 27.91%
- GPU memory avant → après : 9.09 GB → 7.28 GB
- Goulot identifié : GPU compute-bound (99.9% compute time, 0.1% dataloader wait)
- Décision : **gardé** — gain significatif sans régression de la loss (vérifié sur 50 steps).
  torch.compile réduit l'overhead des petits kernels CUDA et diminue la mémoire.

## Session 2026-07-21 — Iteration 2: batch_size scaling

- Date : 2026-07-21
- Contexte matériel : 1× RTX 4090 (23.5 GB VRAM), bf16, TF32, torch.compile activé
- Changement testé : batch_size 4 → 8 → 12
- Tokens/sec : 48,756 (bs=4) → 56,371 (bs=8) → 58,254 (bs=12)
- MFU : 27.9% (bs=4) → 32.3% (bs=8) → 33.4% (bs=12)
- GPU memory : 7.28 GB (bs=4) → 10.56 GB (bs=8) → 13.91 GB (bs=12)
- Goulot identifié : GPU compute-bound. Le scaling du batch augmente l'utilisation du GPU
  mais les gains deviennent marginaux au-delà de bs=8 (+3.3% de bs=8 à bs=12).
- Décision : recommander bs=8-12 selon la mémoire dispo. bs=8 comme point d'équilibre
  mémoire/performance pour le run 300M.
