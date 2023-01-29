cd ../..
python ./test/test_trans_dec.py \
--dataset cars \
--evaluate iou \
--monitor avg_IoU \
--gpus 1 \
--batch_size 128 \
--num_slots 2 \
--log_name trans_dec \
--seed 42 \
--sigma_steps 0 \
# --is_logger_enabled \
