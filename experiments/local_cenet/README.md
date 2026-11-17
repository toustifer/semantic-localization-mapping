# CENet 本地验证脚本

已测环境和性能边界见 [实验记录](../../docs/本地部署与性能验证_2026-11-17.md)。evidence 为已完成运行的原始统计和结果图。

## 目录约定

脚本通过自身目录定位source、data和results。在有足够空间的目录中复制本目录脚本，并在其下准备上游source。不要在接近满载的D盘运行下载。

```bash
git clone https://github.com/toustifer/semantic-slam.git source
git -C source checkout dc6b23a4e6568ad4ef0b7652e458c1e6aeb6c508
python download_kitti_subset.py
python validate_kitti_subset.py
python test_projection.py
python smoke_cenet.py
python real_cenet.py
python plot_real_cenet.py
```

所需Python包：numpy、requests、PyYAML、CUDA版torch；出图另需matplotlib。已测torch版本2.5.1+cu121，其余包版本未锁定，暂不宣称跨环境完全可复现。下载器要求服务器支持HTTP 206 Range并检查返回区间，每个ZIP条目通过CRC并记录SHA256。数据遵循KITTI原许可，见 https://www.cvlibs.net/datasets/kitti/eval_odometry.php 。

real_cenet.py 生成 results/real06/predictions，输出为原始SemanticKITTI类别ID的uint32标签，非实例ID、非运动标签。200帧为部署冒烟测试而非完整序列评价。summary.json记录逐帧信息，preview用于可视化，不作为正确性标注。

当前实现对同像素多点直接使用同一预测，未启用KNN。投影单元测试覆盖近点遮挡、索引0、点序映射及非法输入。ROS和LeGO-LOAM尚未接入该脚本。
