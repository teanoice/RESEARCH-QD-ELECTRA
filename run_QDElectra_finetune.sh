#!/usr/bin/env bash
python classify.py \
        --task 'mrpc' \
        --mode 'train' \
        --base_train_cfg 'config/QDElectra_pretrain.json' \
        --train_cfg 'config/train_mrpc.json' \
        --model_cfg 'config/QDElectra_base.json' \
        --data_file 'data/msr_paraphrase_train.txt' \
        --vocab 'data/vocab.txt' \
        --save_dir 'test_save' \
        --pred_distill True
