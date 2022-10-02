#!/bin/bash

# dataset_lst=("fb100" "arxiv-year" "snap-patents" "pokec" "genius" "twitch-gamer") 
# sub_dataset="Penn94" # Only fb100 uses sub_dataset

cuda=$1
dataset=$2

hidden_channels_lst=(64)
n_layers_lst=(3 4 5)

lr_lst=(0.001 0.005)

weight_decay_lst=(0. 0.0005 0.001 0.005)

dropout_lst=(0.2 0.5 0.7)

R_list=(1. 10. 100.)

for n_layer in "${n_layers_lst[@]}"; do
	for hidden in "${hidden_channels_lst[@]}"; do
		for lr in "${lr_lst[@]}"; do
			for wd in "${weight_decay_lst[@]}"; do
				for dp in "${dropout_lst[@]}"; do
					for R in "${R_list[@]}"; do
							python run_regnn.py --dataset $dataset --model regcn --save_postfix $dataset-regcn --repeat 10 \
							--device $cuda --num_layers $n_layer --hidden_dim $hidden --dropout $dp \
							--lr $lr --weight_decay $wd --R $R --feats_type 2
done
done
done
done
done
done