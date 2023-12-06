# GATv2
python -u run_regnn.py --dataset DBLP --model regatv2 --save_postfix DBLP-regatv2 --lr 0.005 --dropout 0.7 --repeat 10 --device $1

# MixHop
python -u run_regnn.py --dataset DBLP --model remixhop --save_postfix DBLP-remixhop --lr 0.001 --weight_decay 0.0005 --repeat 10 --device $1
python -u run_regnn.py --dataset DBLP --model remixhop --no_re --save_postfix DBLP-mixhop --lr 0.005 --weight_decay 0. --dropout 0.7 --repeat 10 --device $1

# GraphSAGE
python -u run_regnn.py --dataset DBLP --model resage --save_postfix DBLP-resage --repeat 10 --dropout 0.5 --R 10 --device $1

# GIN
python -u run_regnn.py --dataset DBLP --model regin --save_postfix DBLP_regin --repeat 10 --device $1