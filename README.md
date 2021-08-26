# PanGu TensorFlow Version

[盘古项目Repo](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)

[盘古GPU版本Repo](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha-GPU)

本Repo参考与修改自以上两个Repo中的内容，本Repo内容保持原Repo所写的Apache协议

2.6B的 float16 版本，在A6000上大概占用9GB显存

13B的 float16 版本，在A6000上大概占用34GB显存

注意float16版本尽量不要在CPU上运行，特别慢

float32版本，没有在显卡上测试过，建议2.6B在32GB内存的CPU上运行，13B在超过96GB内存的CPU上运行

本项目的姊妹篇，[智源CPM-TF版本地址](https://github.com/qhduan/CPM-LM-TF2)，这两个模型是有一定区别的，具体可以参考上面PanGu官方Repo链接里面的论文

## Demo

具体请查看目录下面这几个ipynb：


- [prediction_13.ipynb](prediction_13.ipynb)  CPU上跑13B fp32的结果
- [prediction_2.6.ipynb](prediction_2.6.ipynb) CPU上跑2.6B fp32的结果
- [test-gpu-13B-fp16.ipynb](test-gpu-13B-fp16.ipynb) GPU上跑13B fp16的结果
- [test-gpu-2.6B-fp16.ipynb](test-gpu-2.6B-fp16.ipynb) GPU上跑2.6B fp16的结果

可运行的Colab，2.6B fp16：[https://colab.research.google.com/drive/12VYofmlZCnJqd2cW-dCnNci9edXuYIlG?usp=sharing](https://colab.research.google.com/drive/12VYofmlZCnJqd2cW-dCnNci9edXuYIlG?usp=sharing)

## Download

注：百度里面没有13B的fp32，因为太大了传不上去

百度：

链接: https://pan.baidu.com/s/1PQp6bU7StZ84o9fCGsks9w 提取码: 1pbd

GDrive:

https://drive.google.com/drive/folders/1332wY_01r67u9BASS3RghTB3T8xIrTl3?usp=sharing
