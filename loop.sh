export PYTHONPATH="/home/core/forces_pro_client":$PYTHONPATH

#rm gail/dynamic_obs_states.npy gail/static_obs_states.npy gail/dynamic_obs_controls.npy

python3 train.py --generate_data --num_tasks 30000

for test in -4 -5 -7
do
for epochs in 850000 900000 950000 1000000
do
for lr in 1e-4 5e-5 1e-5
do
python3 train.py --epochs $epochs --lr $lr --test $test
done
done
done