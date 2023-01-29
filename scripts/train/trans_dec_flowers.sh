cd ../..
python ./train/train_trans_dec.py \
--dataset flowers \
--evaluate iou \
--gpus 1 \
--batch_size 128 \
--num_slots 2 \
--log_name trans_dec \
--check_val_every_n_epoch 50 \
--is_logger_enabled \
--seed 42 \
--sigma_steps 0 \
--init_method 'embedding' \
--truncate 'bi-level' \