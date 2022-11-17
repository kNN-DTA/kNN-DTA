DATASET="BindingDB_Ki"

EXP_NAME=${DATASET}_lr${LR}_bsz${BATCH_SIZE}_total_updates${TOTAL_NUM_UPDATES}_warmup_updates${WARMUP_UPDATES}_${TRAIN_SUBSET}_k${K}_k_mol${K_MOL}_k_pro${K_PRO}_mh${META_HIDDEN}_mhmol${META_HIDDEN_MOL}_mhpro${META_HIDDEN_PRO}_v3_relu_seed${SEED}_1109

DTI_BIN=./data-bin/${DATASET}

SAVE_PATH=./checkpoints/adaptive_knn_training/${EXP_NAME}

python evaluate.py \
    --task dti_separate_add_mask_token_no_register_class \
    --batch-size 32 \
    --valid-subset test \
    --criterion dti_separate_eval \
    --path ${SAVE_PATH}/checkpoint_best.pt \
    --output-fn ./tmp.tsv \
    $DTI_BIN