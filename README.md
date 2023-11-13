<div align="center">

<h1>NDDepth: Normal-Distance Assisted Monocular Depth Estimation and Completion</h1>

<div>
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=ecZHSVQAAAAJ' target='_blank'>Shuwei Shao</a><sup>1</sup>&emsp;
    <a target='_blank'>Zhongcai Pei</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=5PoZrcYAAAAJ' target='_blank'>Weihai Chen</a><sup>1</sup></sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=7E0QgKUAAAAJ' target='_blank'>Peter C. Y. Chen</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=LiUX7WQAAAAJ' target='_blank'>Zhengguo Li</a><sup>3</sup>
</div>
<div>
    <sup>1</sup>Beihang University, <sup>2</sup>National University of Singapore, <sup>3</sup>A*STAR
</div>

<div>
    <h4 align="center">
        • <a href="https://arxiv.org/abs/2309.10592" target='_blank'>Extended Version</a> •
    </h4>
</div>

<div>
    <h4 align="center">
        • <a href="https://arxiv.org/abs/2309.10592" target='_blank'>Conference Version [ICCV 2023(oral)]</a> •
    </h4>
</div>

<strong>In this paper, we introduce novel physics (geometry)-driven deep learning frameworks for these two tasks by assuming that 3D scenes are constituted with piece-wise planes. Instead of directly estimating the depth map or completing the sparse depth map, we propose to estimate the surface normal and plane-to-origin distance maps or complete the sparse surface normal and distance maps as intermediate outputs. To this end, we develop a normal-distance head that outputs pixel-level surface normal and distance. Meanwhile, the surface normal and distance maps are regularized by a developed plane-aware consistency constraint, which are then transformed into depth maps. Furthermore, we integrate an additional depth head to strengthen the robustness of the proposed frameworks. Extensive experiments on the NYU-Depth-v2, KITTI and SUN RGB-D datasets demonstrate that our method exceeds in performance prior state-of-the-art monocular depth estimation and completion competitors.</strong>

<div style="text-align:center">
<img src="assets/teaser.jpg"  width="80%" height="80%">
</div>

---

</div>

## Qualitative Depth and Point Cloud Results
You can download the qualitative depth results of [IEBins](https://arxiv.org/abs/2309.14137), [NDDepth](https://arxiv.org/abs/2309.10592), [NeWCRFs](https://openaccess.thecvf.com/content/CVPR2022/html/Yuan_Neural_Window_Fully-Connected_CRFs_for_Monocular_Depth_Estimation_CVPR_2022_paper.html), [PixelFormer](https://openaccess.thecvf.com/content/WACV2023/html/Agarwal_Attention_Attention_Everywhere_Monocular_Depth_Prediction_With_Skip_Attention_WACV_2023_paper.html), [AdaBins](https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_AdaBins_Depth_Estimation_Using_Adaptive_Bins_CVPR_2021_paper.html) and [BTS](https://arxiv.org/abs/1907.10326) on the test sets of NYUv2 and KITTI_Eigen from [here](https://pan.baidu.com/s/1zaFe40mwpQ5cvdDlLZRrCQ?pwd=vfxd) and download the qualitative point cloud results of IEBins, NDDepth, NeWCRFS, PixelFormer, AdaBins and BTS on the NYUv2 test set from [here](https://pan.baidu.com/s/1WwpFuPBGBUaSGPEdThJ6Rw?pwd=n9rw). 

The source code is comming.
