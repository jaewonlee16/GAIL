export PYTHONPATH="/home/core/forces_pro_client":$PYTHONPATH

for test in -2 -3 -4
do
python3 train.py --epochs 900000 --lr 5e-5 --test $test
python3 train.py --epochs 900000 --lr 5e-5 --is_scale --test $test
done