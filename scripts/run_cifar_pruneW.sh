export CUDA_VISIBLE_DEVICES=0,1,2,3

density=(875 750 625 500 375 250 125)

declare -i TARGET
TARGET=${density[0]}
python3 main_prune.py --task cifar10 --model resnet20 --exp_name "PruneWuniform_cifar10resnet20_FP32_M8_$TARGET" \
--prune_M 8 --target_density $TARGET --resume_path ./pretrainModels/vanilla/resnet20_AdderNet_cifar10.pth --do_train 1 \
--prune_warmup_epoch 0 --batch_size 256 --prune_wgt_method UNIFORM > log0.txt 2>&1
wait
sleep 1

for (( i = 1; i <= 6; i++ ))
do
  echo "Current pruning target density is: ${density[$i]}"
  python3 main_prune.py --task cifar10 --model resnet20 --exp_name "PruneWuniform_cifar10resnet20_FP32_M8_${density[$i]}" \
--prune_M 8 --target_density ${density[$i]} \
--resume_path "./outputs/PruneWuniform_cifar10resnet20_FP32_M8_${density[$((i-1))]}/best_model.pth" \
--do_train 1 --prune_warmup_epoch 0 --batch_size 256 --prune_wgt_method UNIFORM > log0.txt 2>&1
  wait
  sleep 1
done
