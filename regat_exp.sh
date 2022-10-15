cuda=$1
dataset=$2
ft=$3
gnn=$4

heads_lst=(4 8)
hidden_channels_lst=(16 64 128)
n_layers_lst=(3)

lr_lst=(0.001 0.005)

weight_decay_lst=(0. 0.0005 0.001 0.005)

dropout_lst=(0.2 0.5 0.7)

R_list=(100.)

for n_layer in "${n_layers_lst[@]}"; do
	for head in "${heads_lst[@]}"; do
		for hidden in "${hidden_channels_lst[@]}"; do
			for lr in "${lr_lst[@]}"; do
				for wd in "${weight_decay_lst[@]}"; do
					for dp in "${dropout_lst[@]}"; do
						for R in "${R_list[@]}"; do
							python run_regnn.py --dataset $dataset --model $gnn --save_postfix $dataset-$gnn --repeat 10 \
							--device $cuda --num_layers $n_layer --num_heads $head --hidden_dim $hidden --dropout $dp \
							--lr $lr --weight_decay $wd --R $R --feats_type $ft
done
done
done
done
done
done
done