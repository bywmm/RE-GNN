# python run_regnn.py --dataset ACM --model regat --save-postfix ACM-regat --feats-type 2 --repeat 5 --device $1


# regat,ft:2,n_layer:4,he:8,hd:16,lr:0.001,wd:0.005,dp:0.2,R:100.0,93.79 ± 0.17,93.77 ± 0.19,77.02 ± 0.92,80.92 ± 1.22
python run_regnn.py --dataset ACM --model regat --save_postfix ACM-regat --feats_type 2 --hidden 16 --weight_decay 0.005 --dropout 0.2 --repeat 10 --device $1