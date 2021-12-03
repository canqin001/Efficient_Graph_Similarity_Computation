net=gin
epochs=6000
bs=128
lr=0.001

dataset='AIDS700nef'
python src/main.py --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot > 'logs/log_'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$i'.txt'


dataset='LINUX'
python src/main.py  --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot > 'logs/log_'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$i'.txt'


dataset='IMDBMulti'
python src/main.py  --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot > 'logs/log_'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$i'.txt'

dataset='ALKANE'
python src/main.py  --dataset $dataset --gnn-operator $net --epochs $epochs --batch-size $bs --learning-rate $lr --plot > 'logs/log_'$dataset'_'$net'_'$epochs'_'$bs'_'$lr'_'$i'.txt'
