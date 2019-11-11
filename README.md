# Image Super-Resolution Using Very Deep Residual Channel Attention Networks
This repository is for RCAN introduced in the following paper

[Yulun Zhang](http://yulunzhang.com/), [Kunpeng Li](https://kunpengli1994.github.io/), [Kai Li](http://kailigo.github.io/), [Lichen Wang](https://sites.google.com/site/lichenwang123/), [Bineng Zhong](https://scholar.google.de/citations?user=hvRBydsAAAAJ&hl=en), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/), "Image Super-Resolution Using Very Deep Residual Channel Attention Networks", ECCV 2018, [[arXiv]](https://arxiv.org/abs/1807.02758) 


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 14.04/16.04 environment (Python3.6, PyTorch_0.4.0, CUDA8.0, cuDNN5.1) with Titan X/1080Ti/Xp GPUs. RCAN model has also been merged into [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Convolutional neural network (CNN) depth is of crucial importance for image super-resolution (SR). However, we observe that deeper networks for image SR are more difficult to train. The low-resolution inputs and features contain abundant low-frequency information, which is treated equally across channels, hence hindering the representational ability of CNNs. To solve these problems, we propose the very deep residual channel attention networks (RCAN). Specifically, we propose a residual in residual (RIR) structure to form very deep network, which consists of several residual groups with long skip connections. Each residual group contains some residual blocks with short skip connections. Meanwhile, RIR allows abundant low-frequency information to be bypassed through multiple skip connections, making the main network focus on learning high-frequency information. Furthermore, we propose a channel attention mechanism to adaptively rescale channel-wise features by considering interdependencies among channels. Extensive experiments show that our RCAN achieves better accuracy and visual improvements against state-of-the-art methods.

![CA](/Figs/CA.PNG)
Channel attention (CA) architecture.
![RCAB](/Figs/RCAB.PNG)
Residual channel attention block (RCAB) architecture.
![RCAN](/Figs/RCAN.PNG)
The architecture of our proposed residual channel attention network (RCAN).

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. (optional) Download models for our paper and place them in '/RCAN_TrainCode/experiment/model'.

    All the models (BIX2/3/4/8, BDX3) can be downloaded from [Dropbox](https://www.dropbox.com/s/qm9vc0p0w9i4s0n/models_ECCV2018RCAN.zip?dl=0), [BaiduYun](https://pan.baidu.com/s/1bkoJKmdOcvLhOFXHVkFlKA), or [GoogleDrive](https://drive.google.com/file/d/10bEK-NxVtOS9-XSeyOZyaRmxUTX3iIRa/view?usp=sharing).

2. Cd to 'RCAN_TrainCode/code', run the following scripts to train models.

    **You can use scripts in file 'TrainRCAN_scripts' to train models for our paper.**

    ```bash
    # BI, scale 2, 3, 4, 8
    # RCAN_BIX2_G10R20P48, input=48x48, output=96x96
    python main.py --model RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96

    # RCAN_BIX3_G10R20P48, input=48x48, output=144x144
    python main.py --model RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt

    # RCAN_BIX4_G10R20P48, input=48x48, output=192x192
    python main.py --model RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt

    # RCAN_BIX8_G10R20P48, input=48x48, output=384x384
    python main.py --model RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt
    
    # RCAN_BDX3_G10R20P48, input=48x48, output=144x144
    # specify '--dir_data' to the path of BD training data
    python main.py --model RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt

    ```

## Test
### Quick start
1. Download models for our paper and place them in '/RCAN_TestCode/model'.

    All the models (BIX2/3/4/8, BDX3) can be downloaded from [Dropbox](https://www.dropbox.com/s/qm9vc0p0w9i4s0n/models_ECCV2018RCAN.zip?dl=0), [BaiduYun](https://pan.baidu.com/s/1bkoJKmdOcvLhOFXHVkFlKA), or [GoogleDrive](https://drive.google.com/file/d/10bEK-NxVtOS9-XSeyOZyaRmxUTX3iIRa/view?usp=sharing).

2. Cd to '/RCAN_TestCode/code', run the following scripts.

    **You can use scripts in file 'TestRCAN_scripts' to produce results for our paper.**

    ```bash
    # No self-ensemble: RCAN
    # BI degradation model, X2, X3, X4, X8
    # RCAN_BIX2
    python main.py --data_test MyImage --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX2.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5
    # RCAN_BIX3
    python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX3.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5
    # RCAN_BIX4
    python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX4.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5
    # RCAN_BIX8
    python main.py --data_test MyImage --scale 8 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX8.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5
    # BD degradation model, X3
    # RCAN_BDX3
    python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BDX3.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBD --degradation BD --testset Set5
    # With self-ensemble: RCAN+
    # RCANplus_BIX2
    python main.py --data_test MyImage --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath ../LR/LRBI --testset Set5
    # RCANplus_BIX3
    python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath ../LR/LRBI --testset Set5
    # RCANplus_BIX4
    python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath ../LR/LRBI --testset Set5
    # RCANplus_BIX8
    python main.py --data_test MyImage --scale 8 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath ../LR/LRBI --testset Set5
    # BD degradation model, X3
    # RCANplus_BDX3
    python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BDX3.pt --test_only --save_results --chop --self_ensemble  --save 'RCANplus' --testpath ../LR/LRBD --degradation BD --testset Set5
    ```

### The whole test pipeline
1. Prepare test data.

    Place the original test sets (e.g., Set5, other test sets are available from [GoogleDrive](https://drive.google.com/drive/folders/1xyiuTr6ga6ni-yfTP7kyPHRmfBakWovo?usp=sharing) or [Baidu](https://pan.baidu.com/s/1yBI_-rknXT2lm1UAAB_bag)) in 'OriginalTestData'.

    Run 'Prepare_TestData_HR_LR.m' in Matlab to generate HR/LR images with different degradation models.
2. Conduct image SR. 

    See **Quick start**
3. Evaluate the results.

    Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.



## Results
### Quantitative Results
![PSNR_SSIM_BI](/Figs/psnr_bi_1.PNG)
![PSNR_SSIM_BI](/Figs/psnr_bi_2.PNG)
![PSNR_SSIM_BI](/Figs/psnr_bi_3.PNG)
Quantitative results with BI degradation model. Best and second best results are highlighted and underlined

For more results, please refer to our [main papar](https://arxiv.org/abs/1807.02758) and [supplementary file](http://yulunzhang.com/papers/ECCV-2018-RCAN_supp.pdf).
### Visual Results
![Visual_PSNR_SSIM_BI](/Figs/fig1_visual_bi_x4.PNG)
Visual results with Bicubic (BI) degradation (4×) on “img 074” from Urban100


![Visual_PSNR_SSIM_BI](/Figs/fig5_visual_psnr_ssim_bi_x4.PNG)
![Visual_PSNR_SSIM_BI](/Figs/supp_fig1_visual_psnr_ssim_bi_x4_1.PNG)
![Visual_PSNR_SSIM_BI](/Figs/supp_fig1_visual_psnr_ssim_bi_x4_2.PNG)
![Visual_PSNR_SSIM_BI](/Figs/supp_fig1_visual_psnr_ssim_bi_x4_3.PNG)
Visual comparison for 4× SR with BI model

![Visual_PSNR_SSIM_BI](/Figs/fig6_visual_psnr_ssim_bi_x8.PNG)
Visual comparison for 8× SR with BI model

![Visual_PSNR_SSIM_BD](/Figs/fig7_visual_psnr_ssim_bd_x3.PNG)
Visual comparison for 3× SR with BD model

![Visual_Compare_GAN_PSNR_SSIM_BD](/Figs/supp_fig1_visual_compare_gan_psnr_ssim_bi_x4_1.PNG)
![Visual_Compare_GAN_PSNR_SSIM_BD](/Figs/supp_fig1_visual_compare_gan_psnr_ssim_bi_x4_2.PNG)
![Visual_Compare_GAN_PSNR_SSIM_BD](/Figs/supp_fig1_visual_compare_gan_psnr_ssim_bi_x4_3.PNG)
Visual comparison for 4× SR with BI model on Set14 and B100 datasets.
The best results are highlighted. SRResNet, SRResNet VGG22, SRGAN MSE, SR-
GAN VGG22, and SRGAN VGG54 are proposed in [CVPR2017SRGAN], ENet E and ENet PAT are
proposed in [ICCV2017EnhanceNet]. These comparisons mainly show the eﬀectiveness of our proposed
RCAN against GAN based methods

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}

@inproceedings{zhang2018rcan,
    title={Image Super-Resolution Using Very Deep Residual Channel Attention Networks},
    author={Zhang, Yulun and Li, Kunpeng and Li, Kai and Wang, Lichen and Zhong, Bineng and Fu, Yun},
    booktitle={ECCV},
    year={2018}
}
```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes of EDSR [Torch version](https://github.com/LimBee/NTIRE2017) and [PyTorch version](https://github.com/thstkdgus35/EDSR-PyTorch).

