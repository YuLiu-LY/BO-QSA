cd ../..
python ./test/test_trans_dec.py \
--dataset coco \
--evaluate ap \
--gpus 1 \
--batch_size 128 \
--num_slots 6 \
--log_name trans_dec \
--seed 42 \
# --is_logger_enabled \
