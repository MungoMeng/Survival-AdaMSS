# AdaMSS: Adaptive Multi-Modality Segmentation-to-Survival Learning for Survival Outcome Prediction from PET/CT Images
In this study, we propose an Adaptive Multi-modality Segmentation-to-Survival model (AdaMSS) for survival prediction from PET/CT images. We propose a novel Segmentation-to-Survival Learning (SSL) strategy, where our AdaMSS is trained for tumor segmentation and survival prediction sequentially in two stages. This strategy enables the AdaMSS to focus on tumor regions in the first stage and gradually expand its focus to include other prognosis-related regions in the second stage. We also propose a data-driven strategy to fuse multi-modality information, which realizes adaptive optimization of fusion strategies based on training data during training. With the SSL and data-driven fusion strategies, our AdaMSS is designed as an adaptive model that can self-adapt its focus regions and fusion strategy for different training stages. Our AdaMSS is also capable of incorporating conventional radiomics features as an enhancement, where handcrafted features can be extracted from the AdaMSS-segmented tumor regions and then integrated into the AdaMSS through cooperative training and inference. Extensive experiments show that our AdaMSS outperforms state-of-the-art survival prediction methods.  
**For more details, please refer to our paper. [[arXiv](https://arxiv.org/abs/2305.09946)]**

## Overview
![workflow](https://github.com/MungoMeng/Survival-AdaMSS/blob/master/Figure/Overview.png)
![architecture](https://github.com/MungoMeng/Survival-AdaMSS/blob/master/Figure/Architecture.png)

## Publication
* **Mingyuan Meng, Bingxin Gu, Michael Fulham, Shaoli Song, Dagan Feng, Lei Bi, Jinman Kim, "AdaMSS: Adaptive Multi-Modality Segmentation-to-Survival Learning for Survival Outcome Prediction from PET/CT Images," arXiv preprint, arXiv:2305.09946, 2023. [[arXiv](https://arxiv.org/abs/2305.09946)]**
