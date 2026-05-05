#!/bin/bash
# ==============================================================================
# orchestrate_dinov3.sh — Pipeline complet DINOv3 pré-calculé
#
# Enchaîne automatiquement:
#   0. Vérifie que DSMPrompter v2 est terminé
#   1. Libère l espace disque (supprime SAM + SAM2 embeddings)
#   2. Pré-calcule les features DINOv3 ViT-L
#   3. Lance l entraînement DINOv3 avec features pré-calculées
#
# Usage:
#   bash src/misc/orchestrate_dinov3.sh
#   bash src/misc/orchestrate_dinov3.sh --skip-check    # Skip DSMPrompter check
#   bash src/misc/orchestrate_dinov3.sh --skip-cleanup   # Skip SAM/SAM2 deletion
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SAM_EMBED_DIR="src/misc/sam_embeddings_vith"
SAM2_EMBED_DIR="src/misc/sam2_embeddings"
DINO_EMBED_DIR="src/misc/dinov3_embeddings"
DSMV2_LOG="/workspace/training_logs/dsm_prompter_v2.log"
DSMV2_OUTPUT="src/training/output_dsm_prompter_v2"

SKIP_CHECK=false
SKIP_CLEANUP=false

for arg in "$@"; do
    case $arg in
        --skip-check) SKIP_CHECK=true ;;
        --skip-cleanup) SKIP_CLEANUP=true ;;
    esac
done

echo "================================================================"
echo "  DINOv3 Pre-Compute + Training Pipeline"
echo "================================================================"
echo "  Project root: $PROJECT_ROOT"
echo "  Date:         $(date)"
echo "================================================================"

# ==============================================================================
# Étape 0: Vérifier que DSMPrompter v2 est terminé
# ==============================================================================
echo ""
echo "--- Étape 0: Vérification DSMPrompter v2 ---"

if [ "$SKIP_CHECK" = true ]; then
    echo "  [SKIP] Vérification DSMPrompter v2 ignorée (--skip-check)"
else
    DSMV2_DONE=false

    # Check for best model checkpoint
    if [ -f "$DSMV2_OUTPUT/model_best.pth" ]; then
        echo "  model_best.pth trouvé dans $DSMV2_OUTPUT"
        DSMV2_DONE=true
    fi

    # Check log for completion markers
    if [ -f "$DSMV2_LOG" ]; then
        if grep -q "Entraînement terminé\|EARLY STOP\|Training complete" "$DSMV2_LOG" 2>/dev/null; then
            echo "  Marqueur de fin trouvé dans le log"
            DSMV2_DONE=true
        fi
    fi

    # Check if tmux session is still running
    if tmux has-session -t dsmv2 2>/dev/null; then
        echo "  ⚠️  Session tmux dsmv2 encore active!"
        if [ "$DSMV2_DONE" = false ]; then
            echo "  ❌ DSMPrompter v2 semble encore en cours."
            echo "     Attendez la fin ou utilisez --skip-check pour forcer."
            echo "     Dernières lignes du log:"
            tail -5 "$DSMV2_LOG" 2>/dev/null || echo "     (log introuvable)"
            exit 1
        else
            echo "  Le modèle est terminé mais la session tmux est encore ouverte."
            echo "  Fermeture de la session tmux..."
            tmux kill-session -t dsmv2 2>/dev/null || true
        fi
    fi

    if [ "$DSMV2_DONE" = true ]; then
        echo "  ✅ DSMPrompter v2 terminé!"
    else
        echo "  ❌ DSMPrompter v2 ne semble pas terminé."
        echo "     Pas de model_best.pth ni de marqueur de fin dans le log."
        echo "     Utilisez --skip-check pour forcer."
        exit 1
    fi
fi

# ==============================================================================
# Étape 1: Libérer l espace disque
# ==============================================================================
echo ""
echo "--- Étape 1: Libération espace disque ---"

if [ "$SKIP_CLEANUP" = true ]; then
    echo "  [SKIP] Nettoyage ignoré (--skip-cleanup)"
else
    echo "  Espace avant nettoyage:"
    df -h /workspace | tail -1

    if [ -d "$SAM_EMBED_DIR" ]; then
        SIZE=$(du -sh "$SAM_EMBED_DIR" 2>/dev/null | cut -f1)
        echo "  Suppression $SAM_EMBED_DIR ($SIZE)..."
        rm -rf "$SAM_EMBED_DIR"
        echo "  ✅ SAM embeddings supprimés"
    else
        echo "  SAM embeddings déjà supprimés"
    fi

    if [ -d "$SAM2_EMBED_DIR" ]; then
        SIZE=$(du -sh "$SAM2_EMBED_DIR" 2>/dev/null | cut -f1)
        echo "  Suppression $SAM2_EMBED_DIR ($SIZE)..."
        rm -rf "$SAM2_EMBED_DIR"
        echo "  ✅ SAM2 embeddings supprimés"
    else
        echo "  SAM2 embeddings déjà supprimés"
    fi

    echo "  Espace après nettoyage:"
    df -h /workspace | tail -1
fi

# ==============================================================================
# Étape 2: Pré-calculer les features DINOv3
# ==============================================================================
echo ""
echo "--- Étape 2: Pré-calcul features DINOv3 ---"

# Check if already done
TRAIN_COUNT=0
VAL_COUNT=0
if [ -d "$DINO_EMBED_DIR/train" ]; then
    TRAIN_COUNT=$(find "$DINO_EMBED_DIR/train" -name "*.pt" | wc -l)
fi
if [ -d "$DINO_EMBED_DIR/val" ]; then
    VAL_COUNT=$(find "$DINO_EMBED_DIR/val" -name "*.pt" | wc -l)
fi

echo "  Features existantes: train=$TRAIN_COUNT, val=$VAL_COUNT"

if [ "$TRAIN_COUNT" -ge 18700 ] && [ "$VAL_COUNT" -ge 4300 ]; then
    echo "  ✅ Features DINOv3 déjà pré-calculées (train=$TRAIN_COUNT, val=$VAL_COUNT)"
else
    echo "  Lancement Pre_Compute_DINOv3.py..."
    echo "  (Estimation: ~4-6h sur RTX 3090)"
    
    source /venv/main/bin/activate 2>/dev/null || true
    PYTHONPATH="$PROJECT_ROOT" python3 src/misc/Pre_Compute_DINOv3.py 2>&1 | tee /workspace/training_logs/pre_compute_dinov3.log

    # Verify
    TRAIN_COUNT=$(find "$DINO_EMBED_DIR/train" -name "*.pt" | wc -l)
    VAL_COUNT=$(find "$DINO_EMBED_DIR/val" -name "*.pt" | wc -l)
    
    if [ "$TRAIN_COUNT" -lt 18700 ] || [ "$VAL_COUNT" -lt 4300 ]; then
        echo "  ❌ ERREUR: Pré-calcul incomplet!"
        echo "     train: $TRAIN_COUNT (attendu: ~18746)"
        echo "     val:   $VAL_COUNT (attendu: ~4392)"
        echo "     Vérifiez le log: /workspace/training_logs/pre_compute_dinov3.log"
        exit 1
    fi
    
    echo "  ✅ Pré-calcul terminé: train=$TRAIN_COUNT, val=$VAL_COUNT"
fi

echo "  Espace disque après pré-calcul:"
df -h /workspace | tail -1

# ==============================================================================
# Étape 3: Entraînement DINOv3 avec features pré-calculées
# ==============================================================================
echo ""
echo "--- Étape 3: Entraînement DINOv3 (mode pré-calculé) ---"

source /venv/main/bin/activate 2>/dev/null || true

echo "  Lancement de l entraînement..."
echo "  (Estimation: ~6-8h sur RTX 3090)"

PYTHONPATH="$PROJECT_ROOT" python3 src/training/train_dino.py \
    --use-precomputed \
    --embed-dir "$DINO_EMBED_DIR" \
    2>&1 | tee /workspace/training_logs/dino_maskrcnn_precomputed.log

echo ""
echo "================================================================"
echo "  ✅ Pipeline DINOv3 terminé!"
echo "  Log: /workspace/training_logs/dino_maskrcnn_precomputed.log"
echo "  Output: src/training/DINOv3_MaskRCNN_Trees/"
echo "  Date: $(date)"
echo "================================================================"
