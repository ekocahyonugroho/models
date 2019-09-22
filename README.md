# PaddlePaddle Models

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle provides a rich set of computational units to enable users to adopt a modular approach to solving various learning problems. In this Repo, we demonstrate how to use PaddlePaddle to solve common machine learning tasks, providing several different neural network model that anyone can easily learn and use.

PaddlePaddle provides a rich set of computing units that allow users to solve a variety of learning problems in a modular way. In this Repo, we show how to use PaddlePaddle to solve common machine learning tasks and provide several different easy-to-learn neural network models. PaddlePaddle users can receive ** free Tesla V100 online computing resources**, efficient training model, ** daily login is 12 hours**, ** five days of running plus 48 hours**, [go to free use Computation] (http://ai.baidu.com/support/news?action=detail&id=981).

## table of Contents
* [Smart Vision (PaddleCV)] (#PaddleCV)
   * [Image Classification] (#Image Classification)
   * [Target Detection] (#Target Detection)
   * [Image Segmentation] (#Image Segmentation)
   * [key point detection] (#key point detection)
   * [Image Generation] (#Image Generation)
   * [Scene text recognition] (# scene text recognition)
   * [Metrics Learning] (#Metrics Learning)
   * [Video Classification and Motion Positioning] (#Video Classification and Motion Positioning)
* [Smart Text Processing (PaddleNLP)] (#PaddleNLP)
   * [Basic Model (Glossary & Language Model)] (#Basic Model)
   * [Text Understanding (Text Classification & Reading Comprehension)] (#Text Understanding)
   * [Semantic Model (Semantic Representation & Semantic Matching)] (#Semantic Model)
   * [Text generation (machine translation & dialog generation)] (#text generation)
* [Smart Recommendation (PaddleRec)] (#PaddleRec)
* [other models] (# other models)

## PaddleCV

### Image Classification

Image classification is based on the semantic information of images to distinguish different types of images. It is an important basic problem in computer vision. It is the basis of other high-level visual tasks such as object detection, image segmentation, object tracking, behavior analysis, face recognition, etc. The field has a wide range of applications. Such as: face recognition and intelligent video analysis in the security field, traffic scene recognition in the traffic field, content-based image retrieval and automatic classification of albums in the Internet field, image recognition in the medical field.

| **Model Name** | **Model Introduction** | **Data Set** | **Evaluation Indicators top-1/top-5 accuracy** |
| - | - | - | - |
[AlexNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | For the first time successfully applied ReLU, Dropout and LRN in CNN and use GPU for computational acceleration | ImageNet-2012 Verification set | 56.72%/79.17% |
[VGG19](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | Using 3*3 small convolution kernels based on AlexNet to increase network depth, with good generalization Ability | ImageNet-2012 Verification Set | 72.56%/90.93% |
[GoogLeNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | Increased network depth and width without increasing computational load, performance is superior | ImageNet-2012 Verification Set | 70.70%/89.66% |
[ResNet50](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | Residual Network introduces a new residual structure that solves the problem of decreasing accuracy as the network deepens | ImageNet-2012 Verification Set | 76.50%/93.00% |
[ResNet200_vd](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | Fusion of multiple improvements to ResNet, top1 accuracy of ResNet200_vd reaches 80.93% | ImageNet-2012 Verification Set | 80.93 %/95.33% |
[Inceptionv4](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | Combining the Inception module with Residual Connection, greatly speeding up training and gaining performance through ResNet's structure | ImageNet -2012 verification set | 80.77%/95.26% |
[MobileNetV1](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | Transforming a traditional convolutional structure into a two-layer convolutional structure without prejudice to accuracy Significantly reduces computation time, more suitable for mobile and embedded vision applications | ImageNet-2012 Validation Set | 70.99%/89.68% |
[MobileNetV2](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | The fine-tuning of the MobileNet structure, the skip learning connection on the thinner bottleneck layer and the ReLu non on the bottleneck layer Linear processing can achieve better results | ImageNet-2012 Verification Set | 72.15%/90.65% |
[SENet154_vd](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | Added SE (Sequeeze-and-Excitation) module on the basis of ResNeXt to improve recognition accuracy. First place in the classification project of ILSVRC 2017 | ImageNet-2012 Verification Set | 81.40%/95.48% |
[ShuffleNetV2](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | ECCV2018, a lightweight CNN network that strikes a good balance between speed and accuracy. More complex than ShuffleNet and MobileNetv2, more suitable for mobile and unmanned vehicles in the same complexity | ImageNet-2012 Verification Set | 70.03%/89.17% |

For more image classification models, please refer to [Image Classification] (https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)

### Target Detection

The goal of the target detection task is to give an image or a video frame, let the computer find the location of all the targets, and give the specific category of each target. For a computer, what can be "seen" is the number after the image is encoded, but it is difficult to solve the high-level semantic concept such as human or object in the image or video frame, and it is more difficult to locate the target in the image. Which area is in it.

Model Name | Model Introduction | Data Sets | Evaluation Indicators mAP |
| ------------------------------------------------- ----------- | -------------------------------------- ---------------------- | ---------- | ---------------- --------------------------------------- |
[SSD](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | Good inheritance of MobileNet's fast prediction and easy deployment, it can be well on multiple devices Complete image target detection task | VOC07 test | mAP = 73.32% |
[Faster-RCNN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | Creatively use the convolutional network to generate the suggestion box yourself, and share the convolution network with the target detection network. Reduced number of frames and improved quality | MS-COCO | Based on ResNet 50 mAP (0.50:0.95) = 36.7% |
[Mask-RCNN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | A classic two-stage framework that adds splitting branches based on the Faster R-CNN model to get masked results , the solution of the mask and category prediction relationship is realized, and the detection result at the pixel level can be obtained. MS-COCO | Based on ResNet 50 Mask mAP(0.50:0.95) = 31.4% |
[RetinaNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | A classic one-stage framework consisting of a backbone network, an FPN structure, and two separate regression object locations and predictions Subnetwork composition of object categories. The use of Focal Loss in the training process solves the problem that the traditional one-stage detector has a foreground background imbalance, and further improves the accuracy of the one-stage detector. | MS-COCO | Based on ResNet 50 mAP (0.50:0.95) = 36% |
[YOLOv3](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | Speed ​​and precision balanced target detection network, compared to the original author YOLO v3 implementation in darknet, PaddlePaddle implementation reference The paper [Bag of Tricks for Image Classification with Convolutional Neural Networks] (https://arxiv.org/pdf/1812.01187.pdf) added mixup, label_smooth and other processing, accuracy (mAP (0.50:0.95)) compared to the original The author increased the 4.7 absolute percentage points and added synchronized batch normalization. The final accuracy was 5.9 percentage points higher than the original author. MS-COCO | Based on DarkNet mAP(0.50:0.95)= 38.9% |
[PyramidBox](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/face_detection) | **PyramidBox** **Model is Baidu's self-developed face detection model**, solved with context information The problem of detecting difficult faces is high in network expression and strong in robustness. First place in the WIDER Face dataset in March 18 | WIDER FACE | mAP (Easy/Medium/Hard set) = 96.0% / 94.8% / 88.8% |

### Image segmentation

Image Semantic Separation As the name suggests, image pixels are grouped/segmented according to the semantic meaning of the expression. Image semantics refers to the understanding of image content. For example, it can describe what objects are doing things, etc. Segmentation refers to the image. Each pixel in the label is labeled, and the label belongs to which category. In recent years, it has been used in the driving technology of unmanned vehicles to separate street scenes to avoid pedestrians and vehicles, and auxiliary diagnosis in medical image analysis.

Model Name | Model Introduction | Data Sets | Evaluation Indicators |
| ------------------------------------------------- ----------- | -------------------------------------- ---------------------- | --------- | --------------- |
[ICNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/icnet) | Mainly used for real-time semantic segmentation of images, able to balance speed and accuracy, easy to deploy online | Cityscape | Mean IoU =67.0% |
[DeepLab V3+](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/deeplabv3%2B) | Multi-scale information fusion via encoder-decoder while preserving the original hole convolution and ASSP Layer, its backbone network uses the Xception model to improve the robustness and speed of semantic segmentation | Cityscape | Mean IoU=78.81% |

### Critical point detection

Pose Estimation, a key point detection of human bones, mainly detects some key points of the human body, such as joints and facial features, and describes human bone information through key points. Critical detection of human bones is essential for describing human posture and predicting human behavior. It is the basis of many computer vision tasks, such as motion classification, abnormal behavior detection, and automatic driving.

Model Name | Model Introduction | Data Sets | Evaluation Indicators |
| ------------------------------------------------- ----------- | -------------------------------------- ---------------------- | ------------ | ------------ |
[Simple Baselines](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/human_pose_estimation) | coco2018 key point detection project runner-up program, network structure is very simple, the effect reaches state of the art | COCO val2017 | AP = 72.7% |

### Image Generation

Image generation refers to generating a target image based on an input vector. The input vector here can be random noise or a user-specified condition vector. Specific application scenarios include: handwriting generation, face synthesis, style migration, image restoration, and the like.

Model Name | Model Introduction | Data Set |
| ------------------------------------------------- ----------- | -------------------------------------- ---------------------- | ---------- |
[CGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | Conditional Build Against Grid, a conditionally constrained GAN that uses additional information to add conditions to the model to guide the data Generation process | Mnist |
[DCGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | Deep convolution generation against the network, combining GAN and convolution networks to solve the problem of unstable GAN training | Mnist |
[Pix2Pix](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | Image translation, converting a certain type of image into another type of image through paired images, which can be used for style migration| Cityscapes |
[CycleGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | Image translation, which can convert a certain type of image into another type of image through unpaired images, which can be used Style migration | Cityscapes |
[StarGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | Multi-domain attribute migration, introducing auxiliary classifications to help a single discriminator determine multiple attributes, can be used for face attribute conversion | Celeba |
[AttGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | Using classification loss and reconstruction loss to ensure that specific attributes are changed, can be used for face-specific attribute conversion | Celeba |
[STGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | Face-specific attribute conversion, only input changed labels, introduce GRU structure, better select changed attributes | Celeba |

### Scene text recognition

Scene text recognition is a process of converting image information into a sequence of characters in the case of complex image background, low resolution, diverse fonts, random distribution, etc. It can be considered as a special translation process: translation of image input into natural language output. .

Model Name | Model Introduction | Data Sets | Evaluation Indicators |
| ------------------------------------------------- ----------- | -------------------------------------- ---------------------- | -------------------------- | -------------- |
[CRNN-CTC](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition) | Using the CTC model to identify single-line English characters in images, for end-to-end text line image recognition methods | Indefinite length of English string picture | Error rate = 22.3% |
[OCR Attention](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition) | Use attention to identify single-line English characters in images for end-to-end natural scene text recognition, | Single line variable length English string picture | Error rate = 15.8% |

### 测学习

Metric learning is also called distance metric learning and similarity learning. Through the distance between learning objects, metric learning can be used to analyze the association and comparison of object time. It can be applied to practical problems and can be applied to auxiliary classification and aggregation. Class problems are also widely used in areas such as image retrieval and face recognition.

Model Name | Model Introduction | Data Set | Evaluation Indicators Recall@Rank-1 (using arcmargin training) |
| ------------------------------------------------- ----------- | -------------------------------------- ------------------- | ------------------------------ | --------------------------------------------- |
[ResNet50 not fine-tuned] (https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | Feature model trained with arcmargin loss | Stanford Online Product(SOP) | 78.11% |
[ResNet50 uses triplet trimming] (https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | Based on arcmargin loss, using the feature model of triplet loss fine tuning | Stanford Online Product(SOP) | 79.21% |
[ResNet50 uses quadruplet fine-tuning] (https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | Based on arcmargin loss, using quadruplet loss fine-tuned feature model | Stanford Online Product(SOP) | 79.59% |
[ResNet50 uses eml fine-tuning] (https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | Based on arcmargin loss, feature model with eml loss fine-tuning | Stanford Online Product(SOP) | 80.11% |
[ResNet50 uses npairs fine-tuning] (https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | Based on arcmargin loss, using npairs loss fine-tuned feature model | Stanford Online Product(SOP) | 79.81% |

### 视频分类和动作定位

视频分类和动作定位是视频理解任务的基础。视频数据包含语音、图像等多种信息，因此理解视频任务不仅需要处理语音和图像，还需要提取视频帧时间序列中的上下文信息。视频分类模型提供了提取全局时序特征的方法，主要方式有卷积神经网络(C3D,I3D,C2D等)，神经网络和传统图像算法结合(VLAD等)，循环神经网络等建模方法。视频动作定位模型需要同时识别视频动作的类别和起止时间点，通常采用类似于图像目标检测中的算法在时间维度上进行建模。

| 模型名称                                                     | 模型简介                                                     | 数据集                     | 评估指标    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | ----------- |
| [TSN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | ECCV'16提出的基于2D-CNN经典解决方案 | Kinetics-400               | Top-1 = 67% |
| [Non-Local](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 视频非局部关联建模模型 | Kinetics-400               | Top-1 = 74% |
| [stNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | AAAI'19提出的视频联合时空建模方法 | Kinetics-400               | Top-1 = 69% |
| [TSM](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 基于时序移位的简单高效视频时空建模方法 | Kinetics-400               | Top-1 = 70% |
| [Attention   LSTM](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 常用模型，速度快精度高 | Youtube-8M                 | GAP   = 86% |
| [Attention   Cluster](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | CVPR'18提出的视频多模态特征注意力聚簇融合方法 | Youtube-8M                 | GAP   = 84% |
| [NeXtVlad](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 2nd-Youtube-8M比赛第3名的模型 | Youtube-8M                 | GAP   = 87% |
| [C-TCN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 2018年ActivityNet夺冠方案 | ActivityNet1.3 | MAP=31%    |
| [BSN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 为视频动作定位问题提供高效的proposal生成方法 | ActivityNet1.3 | AUC=66.64%    |
| [BMN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 2019年ActivityNet夺冠方案 | ActivityNet1.3 | AUC=67.19%    |

## PaddleNLP

### 基础模型

#### 词法分析

[LAC(Lexical Analysis of Chinese)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis)百度自主研发中文特色模型词法分析任务，**输入是一个字符串，而输出是句子中的词边界和词性、实体类别。

| **模型**         | **Precision** | **Recall** | **F1-score** |
| ---------------- | ------------- | ---------- | ------------ |
| Lexical Analysis | 88.0%         | 88.7%      | 88.4%        |
| BERT finetuned   | 90.2%         | 90.4%      | 90.3%        |
| ERNIE finetuned  | 92.0%         | 92.0%      | 92.0%        |

#### 语言模型

[基于LSTM的语言模型任务](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/language_model)，给定一个输入词序列（中文分词、英文tokenize），计算其PPL（语言模型困惑度，用户表示句子的流利程度）。

| **large config** | **train** | **valid** | **test** |
| ---------------- | --------- | --------- | -------- |
| paddle           | 37.221    | 82.358    | 78.137   |
| tensorflow       | 38.342    | 82.311    | 78.121   |

### 文本理解

#### 情感分析

[Senta(Sentiment Classification)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/sentiment_classification)百度AI开放平台中情感倾向分析模型、百度自主研发的中文情感分析特色模型。

| **模型**      | **dev** | **test** | **模型（****finetune****）** | **dev** | **test** |
| ------------- | ------- | -------- | ---------------------------- | ------- | -------- |
| BOW           | 89.8%   | 90.0%    | BOW                          | 91.3%   | 90.6%    |
| CNN           | 90.6%   | 89.9%    | CNN                          | 92.4%   | 91.8%    |
| LSTM          | 90.0%   | 91.0%    | LSTM                         | 93.3%   | 92.2%    |
| GRU           | 90.0%   | 89.8%    | GRU                          | 93.3%   | 93.2%    |
| BI-LSTM       | 88.5%   | 88.3%    | BI-LSTM                      | 92.8%   | 91.4%    |
| ERNIE         | 95.1%   | 95.4%    | ERNIE                        | 95.4%   | 95.5%    |
| ERNIE+BI-LSTM | 95.3%   | 95.2%    | ERNIE+BI-LSTM                | 95.7%   | 95.6%    |

#### 对话情绪识别

[EmoTect(Emotion Detection)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/emotion_detection)专注于识别智能对话场景中用户的情绪识别，并开源基于百度海量数据训练好的预训练模型。

| **模型** | **闲聊** | **客服** | **微博** |
| -------- | -------- | -------- | -------- |
| BOW      | 90.2%    | 87.6%    | 74.2%    |
| LSTM     | 91.4%    | 90.1%    | 73.8%    |
| Bi-LSTM  | 91.2%    | 89.9%    | 73.6%    |
| CNN      | 90.8%    | 90.7%    | 76.3%    |
| TextCNN  | 91.1%    | 91.0%    | 76.8%    |
| BERT     | 93.6%    | 92.3%    | 78.6%    |
| ERNIE    | 94.4%    | 94.0%    | 80.6%    |

#### Reading Comprehension

[MRC (Machine Reading Comprehension)] (https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2018-DuReader) Machine Reading Comprehension (MRC) is a key task in Natural Language Processing (NLP) One, open source DuReader upgraded the classic reading comprehension BiDAF model, removed the char level embedding, used [pointer network] in the prediction layer (https://arxiv.org/abs/1506.03134), and referenced [ Some network structures in R-NET] (https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) have greatly improved the effect

| **Model** | **Dev ROUGE-L** | **Test ROUGE-L** |
| ------------------------------------------------- ------- | --------------- | ---------------- |
| BiDAF (Original [Thesis] (https://arxiv.org/abs/1711.05073) Baseline) | 39.29 | 45.90 |
| The Baseline System | 47.68 | 54.66 |

### Semantic model

#### ERNIE

[ERNIE (Enhanced Representation from kNowledge IntEgration)] (https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) Baidu's self-study semantic representation model, by modeling words, entities and entity relationships in massive data , learn the semantic knowledge of the real world. Compared with BERT to learn the original language signal, ERNIE directly models the prior semantic knowledge unit, which enhances the semantic representation of the model.
<table border="1" cellspacing="0" cellpadding="0" width="0">
  <tr>
    <td width="66"><p align="center">数据集 </p></td>
    <td width="180" colspan="2"><p align="center">XNLI</p></td>
    <td width="196" colspan="2"><p align="center">LCQMC</p></td>
    <td width="196" colspan="2"><p align="center">MSRA-NER<br />
        (SIGHAN 2006)</p></td>
    <td width="196" colspan="2"><p align="center">ChnSentiCorp</p></td>
    <td width="392" colspan="4"><p align="center">nlpcc-dbqa</p></td>
  </tr>
  <tr>
    <td width="66" rowspan="2"><p align="center">评估<br />指标</p></td>
    <td width="180" colspan="2"><p align="center">acc</p></td>
    <td width="196" colspan="2"><p align="center">acc</p></td>
    <td width="196" colspan="2"><p align="center">f1-score</p></td>
    <td width="196" colspan="2"><p align="center">acc</p></td>
    <td width="196" colspan="2"><p align="center">mrr</p></td>
    <td width="196" colspan="2"><p align="center">f1-score</p></td>
  </tr>
  <tr>
    <td width="82"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
  </tr>
  <tr>
    <td width="66"><p align="center">BERT</p></td>
    <td width="82"><p align="center">78.1</p></td>
    <td width="98"><p align="center">77.2</p></td>
    <td width="98"><p align="center">88.8</p></td>
    <td width="98"><p align="center">87</p></td>
    <td width="98"><p align="center">94.0</p></td>
    <td width="98"><p align="center">92.6</p></td>
    <td width="98"><p align="center">94.6</p></td>
    <td width="98"><p align="center">94.3</p></td>
    <td width="98"><p align="center">94.7</p></td>
    <td width="98"><p align="center">94.6</p></td>
    <td width="98"><p align="center">80.7</p></td>
    <td width="98"><p align="center">80.8</p></td>
  </tr>
  <tr>
    <td width="66"><p align="center">ERNIE</p></td>
    <td width="82"><p>79.9(+1.8)</p></td>
    <td width="98"><p>78.4(+1.2)</p></td>
    <td width="98"><p>89.7(+0.9)</p></td>
    <td width="98"><p>87.4(+0.4)</p></td>
    <td width="98"><p>95.0(+1.0)</p></td>
    <td width="98"><p>93.8(+1.2)</p></td>
    <td width="98"><p>95.2(+0.6)</p></td>
    <td width="98"><p>95.4(+1.1)</p></td>
    <td width="98"><p>95.0(+0.3)</p></td>
    <td width="98"><p>95.1(+0.5)</p></td>
    <td width="98"><p>82.3(+1.6)</p></td>
    <td width="98"><p>82.7(+1.9)</p></td>
  </tr>
</table>

#### BERT

[BERT(Bidirectional Encoder Representation from Transformers)] (https://github.com/PaddlePaddle/LARK/tree/develop/BERT) is a universal semantic representation model with strong migration capability. The Transformer is the basic component of the network and is bidirectional. The Masked Language Model and Next Sentence Prediction are training targets. The pre-training results in a common semantic representation. Combined with a simple output layer, it is applied to downstream NLP tasks and achieves SOTA results on multiple tasks.

#### ELMo

[ELMo (Embeddings from Language Models)] (https://github.com/PaddlePaddle/LARK/tree/develop/ELMo) is one of the important universal semantic representation models, with bidirectional LSTM as the basic component of the network, with the Language Model For the training goal, the general semantic representation is obtained through pre-training, and the general semantic representation is migrated as a Feature to the downstream NLP task, which will significantly improve the model performance of the downstream task.

#### SimNet

[SimNet(Similarity Net)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/similarity_net) A framework for calculating the similarity of short texts, which can be calculated based on two texts entered by the user. Score.

**Model** | **Baidu knows ** | **ECOM** | **QQSIM** | **UNICOM** | **LCQMC** |
| ------------ | ------------ | -------- | --------- | ---- ------ | --------- |
| | AUC | AUC | AUC | Positive and Reverse Order | Accuracy |
BOW_Pairwise | 0.6767 | 0.7329 | 0.7650 | 1.5630 | 0.7532 |

### Text Generation

#### machine translation

[MT(machine translation)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/neural_machine_translation/transformer) Machine translation is the use of computers to convert a natural language (source language) into another nature The process of the language (target language), the input is the source language sentence, and the output is the sentence of the corresponding target language.

| **Test Set** | **newstest2014** | **newstest2015** | **newstest2016** |
| ---------- | ---------------- | ---------------- | ---- ------------ |
| Base | 26.35 | 29.07 | 33.30 |
Big | 27.07 | 30.09 | 34.38 |

#### Conversation automatic evaluation

[Auto Dialogue Evaluation] (https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/dialogue_model_toolkit/auto_dialogue_evaluation) is mainly used to assess the quality of responses in the open field dialogue system and can help companies or individuals Quickly assess the quality of the response to the dialogue system and reduce the cost of manual evaluation.

After fine-tuning a small amount of annotated data, the scorman correlation coefficient of scoring and manual scoring is automatically evaluated, as shown in the following table.

**/** | **seq2seq_naive** | **seq2seq_att** | **keywords** | **human** |
| ----- | ----------------- | --------------- | --------- --- | --------- |
| cor | 0.474 | 0.477 | 0.443 | 0.378 |

#### Dialogue General Understanding

[DGU(Dialogue General Understanding)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/dialogue_model_toolkit/dialogue_general_understanding) Dialogue General Understanding Developed a related model training process for datasets, supporting classification, multi-labeling Tasks such as classification, sequence labeling, etc., users can customize the relevant models for their own data sets.

**ask_name** | **udc** | **udc** | **udc** | **atis_slot** | **dstc2** | **atis_intent** | **swda** | * *mrda** |
| ------------ | ------- | ------- | ------- | ------------ - | ---------- | --------------- | -------- | -------- |
| Dialog Task | Match | Match | Match | Slot Analysis | DST | Intent Recognition | DA | DA |
Task Type | Classification | Classification | Classification | Sequence Labeling | Multi-label Classification | Classification |
| task name | udc | udc | udc | atis_slot | dstc2 | atis_intent | swda | mrda |
| Evaluation Indicators | R1@10 | R2@10 | R5@10 | F1 | JOINT ACC | ACC | ACC | ACC |
SOTA | 76.70% | 87.40% | 96.90% | 96.89% | 74.50% | 98.32% | 81.30% | 91.70% |
DGU | 82.02% | 90.43% | 97.75% | 97.10% | 89.57% | 97.65% | 80.19% | 91.43% |

#### DAM

[Deep Attention Maching] (https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2018-DAM) is an open domain multi-round dialogue matching model. Sort the most appropriate responses based on multiple rounds of conversation history and candidate responses.

| | Ubuntu Corpus | Douban Conversation Corpus | | | | | | | |
| ---- | ------------- | -------------------------- | --- -- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| R2@1 | R10@1 | R10@2 | R10@5 | MAP | MRR | P@1 | R10@1 | R10@2 | R10@5 |
| DAM | 93.8% | 76.7% | 87.4% | 96.9% | 55.0% | 60.1% | 42.7% | 25.4% | 41.0% | 75.7% |

#### Knowledge Driven Dialogue

[New conversation task for knowledge-driven dialogue] (https://github.com/baidu/knowledge-driven-dialogue/tree/master) where the machine talks to people based on the constructed knowledge map. It is designed to test the machine's ability to perform human-like conversations.

| **baseline system** | **F1/BLEU1/BLEU2** | **DISTINCT1/DISTINCT2** |
| ------------------- | ------------------ | ---------- ------------- |
| retrieval-based | 31.72/0.291/0.156 | 0.118/0.373 |
Generation-based | 32.65/0.300/0.168 | 0.062/0.128 |

## PaddleRec

Personalized recommendation is playing an increasingly important role in current Internet services. At present, most e-commerce systems, social networks, advertisement recommendation, and search engines all use various forms of personalized recommendation technology to varying degrees. Help users quickly find the information they want.

| Model Name | Model Introduction |
| ------------------------------------------------- ----------- | -------------------------------------- ---------------------- |
[TagSpace](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | Applicable to industrial-grade label recommendations, specific application scenarios include feed news tag recommendations, etc. |
[GRU4Rec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | The first time RNN (GRU) is applied to session-based recommendations, compared to traditional KNN and matrix decomposition, the effect is obvious Promotion
[SequenceSemanticRetrieval](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | Using the ideas in the reference paper to predict user behavior using multiple time granularities |
[DeepCTR](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | Only implemented the DNN part of the model described in the DeepFM paper, DeepFM will be given in other examples |
[Multiview-Simnet](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | Combine multiple functional views of users and projects into one unified model based on multivariate views |
[Word2Vec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | word2vector model for skip-gram mode |
[GraphNeuralNetwork](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | A recommendation system based on the session-based graph neural network model, which can better mine the rich transformation characteristics of the item and generate accurate Potential user vector representation |
[DeepInterestNetwork](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | DIN uses an information activation module (Activation Unit) to activate the user's history click item with the information of the estimated target Candidate ADs. In order to extract the user's interest related to the current estimated target. |


## Other models

| Model Name | Model Introduction |
| ------------------------------------------------- ----------- | -------------------------------------- ---------------------- |
[DeepASR](https://github.com/PaddlePaddle/models/blob/develop/PaddleSpeech/DeepASR/README_cn.md) | Configure and train acoustic models in speech recognition using the Fluid framework and integrate Kaldi's decoder |
[DQN](https://github.com/PaddlePaddle/models/blob/develop/legacy/PaddleRL/DeepQNetwork/README_cn.md) | value based reinforcement learning algorithm, the first successful combination of deep learning and reinforcement learning Model up |
[DoubleDQN](https://github.com/PaddlePaddle/models/blob/develop/legacy/PaddleRL/DeepQNetwork/README_cn.md) | Applying the idea of ​​Double Q to DQN to solve optimization problems |
[DuelingDQN](https://github.com/PaddlePaddle/models/blob/develop/legacy/PaddleRL/DeepQNetwork/README_cn.md) | Improved DQN model for improved model performance |

## License
This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](LICENSE).


## License
This wizard is contributed by [PaddlePaddle] (https://github.com/PaddlePaddle/Paddle) and is certified by [Apache-2.0 license] (LICENSE).
