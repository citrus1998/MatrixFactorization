today=$(date "+%b_%d_%Y_%H:%M:%S")
scripts/count.pl
val=$(cat scripts/log.txt)

#https://qiita.com/Kobecow/items/bd2cee4b5f2ab51f656b
nohup python3 main.py \
--num_epochs 100 \
--plot "movielens/$val_$today" \
> log/movielens/$val_$today.text & \
