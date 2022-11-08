# GATv2
python run_regnn.py --dataset DBLP --model regatv2 --save_postfix DBLP-regatv2 --lr 0.005 --dropout 0.7 --repeat 10 --device $1

# MixHop
python run_regnn.py --dataset DBLP --model remixhop --save_postfix DBLP-remixhop --lr 0.001 --weight_decay 0.0005 --repeat 10 --device $1
python run_regnn.py --dataset DBLP --model remixhop --no_re --save_postfix DBLP-mixhop --lr 0.005 --weight_decay 0. --dropout 0.7 --repeat 10 --device $1