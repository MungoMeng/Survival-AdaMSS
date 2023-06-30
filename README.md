# AdaMSS: Adaptive Multi-Modality Segmentation-to-Survival Learning for Survival Outcome Prediction from PET/CT Images
In this study, we propose a Deep Multi-modality Segmentation-to-Survival model (DeepMSS) for survival prediction from PET/CT images. We propose a novel Segmentation-to-Survival Learning (SSL) strategy, where our DeepMSS is trained for tumor segmentation and survival prediction sequentially. This strategy enables the DeepMSS to initially focus on tumor regions and gradually expand its focus to include other prognosis-related regions. We also propose a data-driven strategy to fuse multi-modality image information, which realizes automatic optimization of fusion strategies based on training data during training and also improves the adaptability of DeepMSS to different training targets. Our DeepMSS is also capable of incorporating conventional radiomics features as an enhancement, where handcrafted features can be extracted from the DeepMSS-segmented tumor regions and cooperatively integrated into the DeepMSS’s training and inference. Extensive experiments show that our DeepMSS outperforms state-of-the-art survival prediction methods.  
**For more details, please refer to our paper. [[arXiv](https://arxiv.org/abs/2305.09946)]**

## Overview
![workflow](https://github.com/MungoMeng/Survival-DeepMSS/blob/master/Figure/Overview.png)
![architecture](https://github.com/MungoMeng/Survival-DeepMSS/blob/master/Figure/Architecture.png)

## Publication
* **Mingyuan Meng, Bingxin Gu, Michael Fulham, Shaoli Song, Dagan Feng, Lei Bi, Jinman Kim, "AdaMSS: Adaptive Multi-Modality Segmentation-to-Survival Learning for Survival Outcome Prediction from PET/CT Images," Under review, 2023. [[arXiv](https://arxiv.org/abs/2305.09946)]**
