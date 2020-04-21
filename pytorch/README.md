## Vehicle Re-ID

### Prepare data 
```bash
python prepare_2020.py            #used to train the re-id model
python prepare_cam2020.py         #used to train the camera-aware model
```

### Train Model
```bash
python train_2020.py --name SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug   --warm_epoch 5 --droprate 0 --stride 1 --erasing_p 0.5 --autoaug --inputsize 384 --lr 0.02 --use_SE  --gpu_ids 0,1,2  --train_virtual --batchsize 24; 
```

### Validate Model
```bash
python test_2020.py --name SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug
```

### Fine-tune Model
```bash
python train_ft_2020.py --name ft_SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug  --init_name SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug  --droprate 0 --stride 1 --erasing_p 0.5 --inputsize 384 --lr 0.02 --use_SE  --gpu_ids 0,1  --train_all --batchsize 24
```

### Extract Feature for Post-processing
```bash
python submit_result_multimodel.py --name ft_SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug
```

### Combine the feature extracted from Paddlepaddle & Pytorch Models to output the Submission.
```bash
python fast_submit.py
```
