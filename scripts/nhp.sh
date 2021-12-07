today=$(date "+%b_%d_%Y_%H:%M:%S")
scripts/count.pl
val=$(cat scripts/log.txt)

#https://qiita.com/Kobecow/items/bd2cee4b5f2ab51f656b
#python3 ./main.py \
nohup python3 ./jester.py \
--num_epochs 100 \
--plot "$val_$today" \
> log/$val_$today.text & \
