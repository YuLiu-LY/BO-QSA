cd ../..
python ./test/test_mixture_dec.py \
--dataset shapestacks \
--evaluate ari \
--monitor avg_ARI_FG \
--gpus 1 \
--batch_size 128 \
--num_slots 8 \
--check_val_every_n_epoch 1 \
--use_rescale \
--seed 42 \
--resolution 128 128 \
--init_resolution 8 8 \
--encoder_strides 2 1 1 1 \
--decoder_strides 2 2 2 2 \
--log_name mixture_dec \
# --is_logger_enabled \
