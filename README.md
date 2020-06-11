## AICity-reID 2020 (track2)

![](https://github.com/layumi/AICIty-reID-2020/blob/master/heatmap2020.jpg)

In this repo, we include the 1st Place submission to [AICity Challenge](https://www.aicitychallenge.org/) 2020 re-id track (Baidu-UTS submission) [[Paper]](https://github.com/layumi/AICIty-reID-2020/blob/master/paper.pdf)

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

### Extracted Features & Camera Prediction & Direction Prediction:
You may dowanload from [GoogleDrive](https://drive.google.com/file/d/1_S1RnnP2fK3wQ68DsSAiFJYvyqzh55_d/view?usp=sharing) or [OneDrive](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/EebBIatfjE9Nksrl9UQGfQ4B5DIFem7DbJu9i2erpSZ1IQ?e=ukKcuZ)
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
@article{zheng2020beyond,
  title={Going beyond real data: a robust visual representation for vehicle re-identification},
  author={Zhedong Zheng, Minyue Jiang, Zhigang Wang, Jian Wang, Zechen Bai, Xuanmeng Zhang, Xin Yu, Xiao Tan, Yi Yang, Shilei Wen, and Errui Ding},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2020}
}
```
