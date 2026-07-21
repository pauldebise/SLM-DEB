#!/usr/bin/env bash
# Boucle d'itération autonome pour OpenCode + DeepSeek V4 Pro.
# A lancer DANS une session tmux/screen pour survivre aux déconnexions SSH :
#   tmux new -s agent
#   ./agent_loop.sh
# (Ctrl+B puis D pour détacher, `tmux attach -t agent` pour revenir voir.)

set -uo pipefail

REPO_DIR="/workspace/slm-trainer"
LOG_DIR="$REPO_DIR/logs/agent_runs"
MODEL="deepseek/deepseek-v4-pro"
MAX_ITERS="${1:-500}"
SLEEP_BETWEEN=30

mkdir -p "$LOG_DIR"
cd "$REPO_DIR" || { echo "Repo introuvable : $REPO_DIR"; exit 1; }

PROMPT='Continue le projet decrit dans AGENTS.md. Lis PROGRESS.md et BENCHMARKS.md pour savoir ou tu en es exactement. Termine le prochain jalon non fait, teste-le reellement (pas seulement "ca compile"), commit ton travail avec un message clair de type conventional commits, push sur origin, puis mets a jour PROGRESS.md et BENCHMARKS.md avant de terminer ta session. Si le projet est entierement termine selon la section "Definition de solution complete et fonctionnelle" de AGENTS.md, cree un fichier vide nomme DONE a la racine du depot et explique pourquoi dans PROGRESS.md.'

for i in $(seq 1 "$MAX_ITERS"); do
  if [ -f "$REPO_DIR/DONE" ]; then
    echo "[loop] Fichier DONE detecte, arret de la boucle."
    break
  fi

  TS=$(date +%Y%m%d_%H%M%S)
  echo "[loop] Iteration $i - $TS"

  opencode run -m "$MODEL" "$PROMPT" \
    > "$LOG_DIR/iter_${i}_${TS}.log" 2>&1

  # Filet de securite : si l'agent a oublie de committer/pousser, on le fait
  # nous-memes pour ne jamais perdre de travail sur un pod ephemere.
  cd "$REPO_DIR"
  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "chore(auto): sauvegarde de securite fin d'iteration $i" || true
  fi
  git push origin main || echo "[loop] ATTENTION: push echoue a l'iteration $i (verifie le remote/les credentials)"

  sleep "$SLEEP_BETWEEN"
done

echo "[loop] Boucle terminee (DONE atteint ou MAX_ITERS=$MAX_ITERS epuise)."
