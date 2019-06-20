# Adaptive Weighting Multi-Field-of-View CNN for Semantic Segmentation in Pathology
By Hiroki Tokunaga, Yuki Teramoto, Akihiko Yoshizawa, Ryoma Bise  
### Paper
- http://openaccess.thecvf.com/content_CVPR_2019/papers/Tokunaga_Adaptive_Weighting_Multi-Field-Of-View_CNN_for_Semantic_Segmentation_in_Pathology_CVPR_2019_paper.pdf  
- https://arxiv.org/abs/1904.06040  
### Supplementary Material
https://github.com/t-hrk155/AWMF-CNN/blob/master/Supplementary_Materials.pdf
### Poster
https://github.com/t-hrk155/AWMF-CNN/blob/master/cvpr19_poster_tokunaga.pdf

## Abstract  
Automated digital histopathology image segmentation is an important task to help pathologists diagnose tumors and cancer subtypes. For pathological diagnosis of cancer subtypes, pathologists usually change the magnification of whole-slide images (WSI) viewers. A key assumption is that the importance of the magnifications depends on the characteristics of the input image, such as cancer subtypes. In this paper, we propose a novel semantic segmentation method, called Adaptive-Weighting-Multi-Field-of-View-CNN (AWMF-CNN), that can adaptively use image features from images with different magnifications to segment multiple cancer subtype regions in the input image. The proposed method aggregates several expert CNNs for images of different magnifications by adaptively changing the weight of each expert depending on the input image. It leverages information in the images with different magnifications that might be useful for identifying the subtypes. It outperformed other state-of-the-art methods in experiments.

## Results
![Results](https://github.com/t-hrk155/AWMF-CNN/blob/master/Results.PNG)

## Requirements: software
Requirements for Keras==2.2.4 and Tensorflow-gpu==1.8.0
