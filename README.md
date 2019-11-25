# lightDSFD

By [Jian Li](https://lijiannuist.github.io/)

## Introduction
We propose lightDSFD based on [DSFD](https://arxiv.org/abs/1810.10220).
the original DSFD code is [here](https://github.com/TencentYoutuResearch/FaceDetection-DSFD).

On Nvidia Tesla P40，the time of network is 13ms. 


模型 NMS_Params|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
lightDSFD,thresh=0.05,number=500|0.891 |0.864       |0.469
