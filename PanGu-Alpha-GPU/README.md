# PanGu-Alpha-GPU

 

### 描述

本项目是  [Pangu-alpha](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) 的 GPU 版本，关于  [Pangu-alpha](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) 的原理、数据集等信息请查看原项目。该项目现阶段主要是让 Pangu-alpha 模型能在 GPU 上进行推理和训练，让更多人体验到大模型的魅力。开放的宗旨就是要集思广益、抛砖引玉、挖掘大模型应用潜力，同时发现存在的问题，以指导我们未来的创新研究和突破。



# mindspore 推理、Finetune、预训练

1. [请查看](inference_mindspore_gpu/README.md)：该部分代码只支持推理，如果只想体验一下盘古α的话，推荐使用这个页面下的《三分钟实现推理教程》。
2. [请查看](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/pangu_alpha)：如果想在盘古α上开发的话，推荐使用 mindspore 提供的训练和推理代码。mindspore 官网的 model_zoo 提供了推理、Finetune、预训练全流程。

# pytorch 推理、Finetune、预训练

[请查看](panguAlpha_pytorch/README.md)：基于 Megatron-1.1开发的盘古α的推理、Finetune、预训练全流程。
