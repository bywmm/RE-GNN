# GATv2
python -u run_regnn.py --dataset ACM --model regatv2 --save_postfix ACM-regatv2 --feats_type 2 --hidden 16 --weight_decay 0.005 --dropout 0.2 --repeat 10 --device $1

# MixHop
python -u run_regnn.py --dataset ACM --model remixhop --save_postfix ACM-remixhop --feats_type 2 --weight_decay 0.005 --dropout 0.7 --repeat 10 --device $1

# GraphSAGE
python -u run_regnn.py --dataset ACM --model resage --hidden_dim 32 --feats_type 2 --lr 0.005 --weight_decay 0.001 --dropout 0.5 --R 10 --device $1

# GIN
python -u run_regnn.py --dataset ACM --model regin --save_postfix ACM-regin --feats_type 2 --weight_decay 0.005 --repeat 10 --device $1