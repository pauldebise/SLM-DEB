# Benchmarks & décisions d'optimisation

Append-only : chaque entrée date un micro-benchmark, le changement testé,
les chiffres avant/après, et la décision (gardé/rejeté) + justification.
Voir Phase 7 de AGENTS.md pour la méthode.

---

## Session 2026-07-21 (run 2) — Iteration 3: compile mode re-evaluation

- Date : 2026-07-21
- Contexte matériel : 1x RTX 4090 (23.5 GB VRAM), bf16, TF32 activé
- Changement testé : `torch.compile(mode="reduce-overhead")` vs `mode="default"` avec gradient accumulation
- Tokens/sec avec default mode (bs=8) : 64,610
- MFU avec default mode (bs=8) : 37.0%
- GPU memory (bs=8) : 8.20 GB
- Goulot identifié : `reduce-overhead` (CUDA graphs) est incompatible avec
  weight tying + gradient accumulation → crash. `default` mode fonctionne et
  donne des performances acceptables.
- Décision : **gardé le fallback automatique** `reduce-overhead` → `default`.
  Gain de 26.7% de `reduce-overhead` perdu par rapport à l'itération 1,
  compensé partiellement par le scaling batch plus agressif (64.6k vs 48.8k
  tokens/sec). MFU de 37% avec `default` est honorable. Un refactoring du
  weight tying pour compatibilité CUDA graphs est laissé pour plus tard.

## Session 2026-07-21 (run 2) — Iteration 4: batch size hardware detection fix

- Date : 2026-07-21
- Contexte matériel : 1x RTX 4090 (23.5 GB VRAM), bf16, TF32, torch.compile default
- Changement testé : formule micro_batch dans hardware_detect.py (vram*2 →
  vram*0.4) + cible effective tokens proportionnelle
- Avant : micro_batch=32 → OOM au premier forward
- Après : micro_batch=9, effective_target=147,456, grad_accum=16
- GPU memory (avéré) : ~8-10 GB stable en entraînement avec bs=9
- Décision : **gardé** — le scaling conservateur évite l'OOM et le gradient
  accumulation maintient un throughput élevé. La formule s'adapte
  automatiquement à d'autres GPUs.

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
