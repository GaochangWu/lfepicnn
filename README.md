***********************************************************************************************************
Matlab demo code for "Light Field Reconstruction Using Deep Convolutional Network on EPI" (CVPR 2017)
***********************************************************************************************************

Note:
The restoration kernels include SCN [1], SRSC [2], SRCNN [3], VDSR [4] and FSRCNN [5]. The non-blind deblur code is by Pan et al. [6].

Please cite our paper if you use this code, thank you! 

@inproceedings{EPICNN17,
	author = {Gaochang Wu and Mandan Zhao and Liangyong Wang and Qionghai Dai and Tianyou Chai and Yebin Liu},
	title = {Light Field Reconstruction Using Deep Convolutional Network on EPI},
	booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017},
	year = {2017},
}

[1] Zhaowen Wang, Ding Liu, Wei Han, Jianchao Yang and Thomas S. Huang, Deep Networks for Image Super-Resolution with Sparse Prior. International Conference on Computer Vision (ICCV), 2015

[2] J. Yang et al. Image super-resolution via sparse representation. IEEE Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010

[3] Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2015

[4] Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee, Accurate Image Super-Resolution Using Very Deep Convolutional Networks, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

[5] Chao Dong, Chen Change Loy, Xiaoou Tang. Accelerating the Super-Resolution Convolutional Neural Network, in Proceedings of European Conference on Computer Vision (ECCV), 2016

[6] Jinshan Pan, Zhe Hu, Zhixun Su, and Ming-Hsuan Yang, Deblurring Text Images via L0-Regularized Intensity and Gradient Prior, CVPR 2014

***********************************************************************************************************

Usage:
1. Please download Lytro data at "http://lightfields.stanford.edu/", and save the data under the file named "Data".
2. Before testing the code, please install "matconvnet" by running "install.m".
3. Make sure the 'utils', 'non-blind deconvolution', './matconvnet' are in your path.
4. Demo code is "main.m".
5. Batch processing code is "main_batchProcessing.m".

***********************************************************************************************************
