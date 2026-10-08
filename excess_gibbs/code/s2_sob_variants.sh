cd "$(dirname "$0")"; mkdir -p ../results/s2v2_logs
export S2_TAG=s2v2 S2_SEEDS=0 TORCH_THREADS=1
run() { SOB_LAMBDA=$1 SOB_LOSS=$2 S2_NAME=$3 python s2_train.py gex_sob > ../results/s2v2_logs/train_$3.log 2>&1; }
( run 0.03 mse gex_sob_l003 ; run 0.1 huber gex_sob_l01h ) &
( run 0.01 mse gex_sob_l001 ; run 0.03 huber gex_sob_l003h ) &
wait
echo SOBDONE > ../results/s2v2_logs/sob_variants.done
