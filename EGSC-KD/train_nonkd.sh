date=032122

mode=both
epochs=5000
bs=128
lr=0.001
dataset='AIDS700nef'
python src/main_nonkd.py --dataset $dataset --mode $mode --epochs $epochs --batch-size $bs --learning-rate $lr > 'logs/log_nonkd_'$dataset'_'$date'_'$epochs'_'$bs'_'$lr'_'$mode'.txt'

# epochs=6000
# bs=128
# lr=0.001
# dataset='LINUX'
# python src/main_nonkd.py  --dataset $dataset --mode $mode --epochs $epochs --batch-size $bs --learning-rate $lr  > 'logs/log_nonkd_'$dataset'_'$date'_'$epochs'_'$bs'_'$lr'_'$i'.txt'

# epochs=6000
# bs=128
# lr=0.001
# dataset='IMDBMulti'
# python src/main_nonkd.py  --dataset $dataset --mode $mode --epochs $epochs --batch-size $bs --learning-rate $lr  > 'logs/log_nonkd_'$dataset'_'$date'_'$epochs'_'$bs'_'$lr'_'$i'.txt'
