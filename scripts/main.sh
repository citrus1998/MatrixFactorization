today=$(date "+%b_%d_%Y_%H:%M:%S")

#python3 ./main.py \
python3 ./jester.py \
--num_epochs 100 \
--plot "$today" \