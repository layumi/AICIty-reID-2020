## Vehicle Re-ID
The code is modified from our baseline code (https://github.com/layumi/Person_reID_baseline_pytorch)

### Prerequisite
EfficientNet-Pytorch https://github.com/lukemelas/EfficientNet-PyTorch

### Prepare data
Make a dir and put the AICity2020 data into this folder.
```bash
mkdir data
```
Extract XML information https://github.com/PaddlePaddle/Research/tree/master/CV/PaddleReid/process_aicity_data 
and rename the file. 
```
|- data
    |- 2020AICITY
        |- ...
        |- 000345_c020_9.jpg
        |- ...
        |- 002028_c036_4_9_95_2.jpg
```

Then you could run the following code to prepare the data for pytorch to load data. You may modify the data path.
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

If you want to directly test the result, the extracted features & camera prediction & direction prediction could be dowanloaded from [GoogleDrive](https://drive.google.com/file/d/1RAQFT9umi6kTehFRiISu0g9xKI3PScbc/view?usp=sharing) or [OneDrive](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/ES6hLEPxZpBNhniTczS6R9sBURNdPqG-l2krgO4joUH4UA?e=lJEhTr).

### Combine the feature extracted from Paddlepaddle & Pytorch Models to output the Submission.
```bash
python fast_submit.py
```
