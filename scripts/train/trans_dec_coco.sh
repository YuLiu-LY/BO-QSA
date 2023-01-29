cd ../..
python ./train/train_trans_dec.py \
--dataset coco \
--evaluate ap \
--gpus 1 \
--batch_size 128 \
--num_slots 6 \
--log_name trans_dec \
--check_val_every_n_epoch 50 \
--is_logger_enabled \
--seed 42 \
--sigma_steps 30000 \
--init_method 'embedding' \
--truncate 'bi-level' \