cuda=$1
dataset=$2
ft=$3

hidden_channels_lst=(16 32)
n_layers_lst=(2 3 4)

lr_lst=(0.001 0.005)

weight_decay_lst=(0. 0.0005 0.001)

dropout_lst=(0.5 0.7)

R_list=(10. 100.)

for n_layer in "${n_layers_lst[@]}"; do
	for hidden in "${hidden_channels_lst[@]}"; do
		for lr in "${lr_lst[@]}"; do
			for wd in "${weight_decay_lst[@]}"; do
				for dp in "${dropout_lst[@]}"; do
					for R in "${R_list[@]}"; do
							python run_regnn.py --dataset $dataset --model regin --save_postfix $dataset-regin --repeat 10 \
							--device $cuda --num_layers $n_layer --hidden_dim $hidden --dropout $dp \
							--lr $lr --weight_decay $wd --R $R --feats_type $ft
done
done
done
done
done
done