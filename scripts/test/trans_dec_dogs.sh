cd ../..
python ./test/test_trans_dec.py \
--dataset dogs \
--evaluate iou \
--monitor avg_IoU \
--gpus 1 \
--batch_size 128 \
--num_slots 2 \
--log_name trans_dec \
--check_val_every_n_epoch 30 \
--seed 42 \
# --is_logger_enabled \
