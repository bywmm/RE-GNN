# python run_regnn.py --dataset DBLP --model regat --save-postfix DBLP-regat --weight_decay 0.001 --repeat 5 --device $1

# regat,ft:0,n_layer:4,he:8,hd:16,lr:0.001,wd:0.0005,dp:0.7,R:100.0,94.06 ± 0.32,94.51 ± 0.29,79.11 ± 1.14,84.12 ± 1.20
python run_regnn.py --dataset DBLP --model regat --save_postfix DBLP-regat --hidden 16 --weight_decay 0.0005 --dropout 0.7 --repeat 10 --device $1
