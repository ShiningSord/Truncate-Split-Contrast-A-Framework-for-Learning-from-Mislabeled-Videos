curtime=$(date "+%Y-%m-%d-%H:%M:%S")

dirname='./history/k200_pairflip_04/'${curtime}'_'$2

mkdir "${dirname}"

 	
pip3 install tensorboardX scikit-learn matplotlib

python3 train_test.py k200 RGB --arch resnet50 \
 --num_segments 8 --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50  --batch-size 64 -j 32  --dropout 0.5 \
 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --vec_len 2048 \
 --shift_place=blockres --npb --warmup_length 1 --dense_sample \
 --noise-type 'pairflip' --noise-rate 0.4 \
 --eval_batch_size=160 --eval_test_crops=1 \
 --eval_start_epoch=0 --eval_feature_pool=True \
 --dist-url 'tcp://'$3':12857' --multiprocessing-distributed --world-size $1 --rank $2


    

cp *.py $dirname
cp *.sh $dirname
cp simple_config.json $dirname
cp -r ops $dirname
cp -r tools $dirname
mv checkpoint $dirname
mv log $dirname
echo ''backup and checkpoint is saved to ''$dirname''
