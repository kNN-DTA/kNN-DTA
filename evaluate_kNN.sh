python evaluate_kNN.py \
    --task dti_separate_add_mask_token \
    --datastore-path $dstore_path \
    --result-file-path $result_path \
    --prediction-mode combine \
    --dataset $dataset \
    --T $T --k $k --l $l \
    --T-0 $T_0 --k-0 $k_0 --knn-embedding-weight-0 $knn_embedding_weight_0 \
    --T-1 $T_1 --k-1 $k_1 --knn-embedding-weight-1 $knn_embedding_weight_1 \
    --batch-size 32 \
    --valid-subset test \
    --criterion dti_separate_knn_cls_eval_no_cross_attn \
    --path $ckpt_path \
    $data_bin