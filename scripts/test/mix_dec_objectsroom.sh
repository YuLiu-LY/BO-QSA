cd ../..
python ./test/test_mixture_dec.py \
--dataset objectsroom \
--evaluate ari \
--monitor avg_ARI_FG \
--gpus 1 \
--batch_size 128 \
--num_slots 5 \
--check_val_every_n_epoch 1 \
--use_rescale \
--seed 42 \
--resolution 64 64 \
--init_resolution 8 8 \
--encoder_strides 1 1 1 1 \
--decoder_strides 2 2 2 1 \
--log_name mixture_dec \
# --is_logger_enabled \
