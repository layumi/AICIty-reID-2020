## AICity-reID 2020 (track2)

![](https://github.com/layumi/AICIty-reID-2020/blob/master/heatmap2020.jpg)

In this repo, we include the 1st Place submission to [AICity Challenge](https://www.aicitychallenge.org/) 2020 re-id track (Baidu-UTS submission) 

[[Paper]](https://github.com/layumi/AICIty-reID-2020/blob/master/paper.pdf) [[Video]](https://www.bilibili.com/video/BV1hK411A78n/)

We fuse the models trained on Paddlepaddle and Pytorch. To illustrate them, we provide the two training parts seperatively as following. 

- We include the [Paddlepaddle](https://github.com/PaddlePaddle/Paddle) training code at [Here](https://github.com/PaddlePaddle/Research/tree/master/CV/PaddleReid).
- We include the [Pytorch](https://pytorch.org/) training code at [Here](https://github.com/layumi/AICIty-reID-2020/tree/master/pytorch).

### Performance：
 AICITY2020 Challange Track2 Leaderboard
 
 |TeamName|mAP|Link|
 |--------|----|-------|
 |**Baidu-UTS(Ours)**|84.1%|[code](https://github.com/layumi/AICIty-reID-2020)|
 |RuiYanAI|78.1%|[code](https://github.com/Xiangyu-CAS/AICity2020-VOC-ReID)|
 |DMT|73.1%|[code](https://github.com/heshuting555/AICITY2020_DMT_VehicleReID)|
 
 
### Trained Models 
How to extract features? Please refer to [[Here]](https://github.com/layumi/AICIty-reID-2020/tree/master/pytorch#extract-feature-for-post-processing) 
and there is one simplified version at [[Here]](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial#part-21-extracting-feature-python-testpy).
Here we provide one model of the final models.

- SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug (AICity 2020) can be downloaded at [[GoogleDrive]](https://drive.google.com/file/d/1AZ4hHbRbz2T8OHJ6QTG9bR7CP2zUOyQh/view?usp=sharing). 

The state-of-the-art model achieving 83.41% mAP on [VeRi-776](https://github.com/JDAI-CV/VeRidataset), which is based on our TMM paper. 
- Training on VehicelNet only (80.91): Res50_imbalance_s1_256_p0.5_lr2_mt_d0_b48 (TMM) can be downloaded at [[GoogleDrive]](https://drive.google.com/file/d/1wUbYm5-EJs0W-LAGS69yvb33D6NkFWpH/view?usp=sharing).
- Finetuning on VeRi (83.41): ft_Res50_imbalance_s1_256_p0.5_lr1_mt_d0.2_b48_w5 (TMM) can be downloaded at [[GoogleDrive]](https://drive.google.com/file/d/1Sor7Grh_1Kot6CBLaw2alDT4Nr3JuH3C/view?usp=sharing).

### Extracted Features & Camera Prediction & Direction Prediction:
I have updated the feature. You may download from [GoogleDrive](https://drive.google.com/file/d/1q0ap5smXoRIQ-oEUMbSMMSl_lEOT0Fk6/view?usp=sharing) or [OneDrive](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/EdxlWLP9bB9Bga0jfDyoIO8Berahz8plAeRY6M4t8g_6iA?e=mSttQx) (expired by July 1 2022)
```
├── final_features/
│   ├── features/                  /* extracted pytorch feature
│   ├── pkl_feas/                   /* extracted paddle feature (include direction similarity)
│       ├── real_query_fea_ResNeXt101_32x8d_wsl_416_416_final.pkl 
|           ...
│       ├── query_fea_Res2Net101_vd_final2.pkl                 
│   ├── gallery_cam_preds_baidu.txt      /*  gallery camera prediction
│   ├── query_cam_preds_baidu.txt      /*  query camera prediction
|   ├── submit_cam.mat             /*  camera feature for camera similarity calculation
```

### Related Repos：

- :helicopter:  Drone-based building re-id [[code]](https://github.com/layumi/University1652-Baseline)  [[paper]](https://arxiv.org/abs/2002.12186)
 
- [Vehicle re-ID Paper Collection] https://github.com/layumi/Vehicle_reID-Collection

- [Person re-ID Baseline] https://github.com/layumi/Person_reID_baseline_pytorch

- [Person/Vehicle Generation] https://github.com/NVlabs/DG-Net

### Citation
Please cite this paper if it helps your research:
```bibtex
@inproceedings{zheng2020going,
  title={Going beyond real data: A robust visual representation for vehicle re-identification},
  author={Zheng, Zhedong and Jiang, Minyue and Wang, Zhigang and Wang, Jian and Bai, Zechen and Zhang, Xuanmeng and Yu, Xin and Tan, Xiao and Yang, Yi and Wen, Shilei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={598--599},
  year={2020}
}

@article{zheng2020beyond,
  title={VehicleNet: Learning Robust Visual Representation for Vehicle Re-identification},
  author={Zheng, Zhedong and Ruan, Tao and Wei, Yunchao and Yang, Yi and Mei, Tao},
  journal={IEEE Transactions on Multimedia (TMM)},
  doi={10.1109/TMM.2020.3014488},
  note={\mbox{doi}:\url{10.1109/TMM.2020.3014488}},
  year={2020}
}
```

The heatmap visualization is based on 
```bibtex
@article{zheng2017discriminatively,
  title={A discriminatively learned cnn embedding for person reidentification},
  author={Zheng, Zhedong and Zheng, Liang and Yang, Yi},
  journal={ACM transactions on multimedia computing, communications, and applications (TOMM)},
  volume={14},
  number={1},
  pages={1--20},
  year={2017},
  publisher={ACM New York, NY, USA}
}
```
