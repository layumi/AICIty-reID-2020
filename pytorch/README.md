## Vehicle Re-ID

### Prepare data 
```bash
python prepare_2020.py
python prepare_cam2020.py
```

### Training Model
```bash
python train_2020.py --name SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug   --warm_epoch 5 --droprate 0 --stride 1 --erasing_p 0.5 --autoaug --inputsize 384 --lr 0.02 --use_SE  --gpu_ids 0,1,2  --train_virtual --batchsize 24; 
```

### Finetuning Model
```bash
python train_ft_2020.py --name ft_SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug  --init_name SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug  --droprate 0 --stride 1 --erasing_p 0.5 --inputsize 384 --lr 0.02 --use_SE  --gpu_ids 0,1  --train_all --batchsize 24
```
