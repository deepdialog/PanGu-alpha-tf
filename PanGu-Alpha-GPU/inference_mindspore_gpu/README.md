# PanGu-Alpha-GPU

 

### 描述

本项目是  [Pangu-alpha](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) 的 GPU 推理版本，关于  [Pangu-alpha](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) 的原理、数据集等信息请查看原项目。该项目现阶段主要是让 Pangu-alpha 模型能在 GPU 上进行推理和训练，让更多人体验到大模型的魅力。开放的宗旨就是要集思广益、抛砖引玉、挖掘大模型应用潜力，同时发现存在的问题，以指导我们未来的创新研究和突破。



### 模型 ckpt

| 模型                                                         | MD5                              | 精度 |
| ------------------------------------------------------------ | -------------------------------- | ---- |
| [Pangu-alpha_2.6B.ckpt](https://git.openi.org.cn/attachments/27234961-4d2c-463b-9052-0240cc7ff29b?type=0) | da404a985671f1b5ad913631a4e52219 | fp32 |
| [ PanguAlpha_13b_fp16.ckpt](https://git.openi.org.cn/attachments/650711d6-6310-4dc2-90f8-153552e59c7a?type=0) | f2734649b9b859ff4cf62d496291249a | fp16 |
| [PanguAlpha_2.6B_fp16.ckpt](https://git.openi.org.cn/attachments/7ff30c2f-e9e4-44be-8eaa-23c9d617b781?type=0) | 3a14e8bf50548a717160e89df7c14b63 | fp16 |

[Pangu-alpha_2.6B.ckpt](https://git.openi.org.cn/attachments/27234961-4d2c-463b-9052-0240cc7ff29b?type=0) 可以用于 `fp16` 和 `fp32` 的 2.6B 模型 的加载，因为在模型加载阶段会进行精度转换

[ PanguAlpha_13b_fp16.ckpt](https://git.openi.org.cn/attachments/650711d6-6310-4dc2-90f8-153552e59c7a?type=0) 只能用于 `fp16` 的13B 模型的加载

[PanguAlpha_2.6B_fp16.ckpt](https://git.openi.org.cn/attachments/7ff30c2f-e9e4-44be-8eaa-23c9d617b781?type=0) 可以用于 `fp16` 的 2.6B 模型 的加载，效果和[Pangu-alpha_2.6B.ckpt](https://git.openi.org.cn/attachments/27234961-4d2c-463b-9052-0240cc7ff29b?type=0) 是一样的，但该 ckpt 消耗的内存更小，约20g。

### 显存占用情况

| 模型      | 显存占用  |
| --------- | --------- |
| 2.6B_fp16 | 6728 MiB  |
| 2.6B_fp32 | 17214 MiB |
| 13B_fp16  | 26430 MiB |

可以根据显卡显存大小运行不同的模型

`2.6B_fp16` 模型应该可以大多数显卡上运行

已经在 T4 成功运行 `2.6B_fp16` 模型，在  v100 上成功运行 `2.6B_fp16` `2.6B_fp32` `13B_fp16`模型

### 推理

###**想快速体验？？请查看[3分钟实现推理教程](3-minus-inference.md)！！可以白嫖 T4 服务器哦！！！**


##### 环境

可以直接使用现成的 docker 镜像

```
docker pull yands/mindspore_pangu-alpha:1.2.0
```
python 路径 `/usr/local/bin/python`

如果不喜欢使用这个镜像，也可以使用 `mindspore:1.2.0` 版本，可以直接运行 `2.6B_fp32` 模型。想运行 `fp16` 的模型需要修改几处 mindspore 源码，请查看[附录](#源码修改)。

##### 依赖

```
pip install jieba
pip install sentencepiece==0.1.94
```

##### 运行

```
python run_inference.py --model=13B_fp16 --load_ckpt_path=/xxx/PanguAlpha_13b_fp16.ckpt
python run_inference.py --model=2B6_fp16 --load_ckpt_path=/xxx/PanguAlpha_2_6b.ckpt
python run_inference.py --model=2B6 --load_ckpt_path=/xxx/PanguAlpha_2_6b.ckpt
```

##### 结果

注：该结果限制输出 tokens 长度为50，没有针对不同任务做后处理

```
Input is: 上联：瑞风播福泽，事业具昌盛千家乐

2.6B_fp16 : 求下联
上联:瑞风播福泽,事业具昌盛千家乐求对下联:瑞风播德泽,事业成伟业万家乐

2.6B_fp32 :;下联:瑞雪迎瑞雪,事业兴盛万家乐。
横批:瑞雪兆丰年。
【瑞雪】
上联:瑞雪兆丰年,事业具昌盛千家乐

13B_fp16  :下联怎么对
上联:瑞风播福泽,事业具昌盛千家乐下联:福泽播善政,福泽流天下万民欢
```
```
Input is: 四川的省会是?

2.6B_fp16 : 省会是成都,四川省的省会是成都市,简称:成 都,是中华人民共和国四川省的首府,位于四川盆地西南边缘,成都平原中部,成都市区东南部,东与德阳市、绵阳

2.6B_fp32 :成都。。。。。。
好的,谢谢啦

13B_fp16  :成都。
```
```
Input is: 上联：春雨润人间，社会和谐万象新

2.6B_fp16 : 下联:春风暖人间,社会和谐万象新。横批:春雨润大地

2.6B_fp32 :下联:春晖照世界,世界和平百花盛开。
春雨滋润着大地,社会和谐万象更新,春晖照耀着世界。

13B_fp16  :下联:春风吹大地,大地春意盎然
```
```
Input is: 书生：羌笛何须怨杨柳，春风不度玉门关。
飞云：（这诗怎么这么耳熟？且过去跟他聊聊如何。）
书生：小兄弟，要不要一起喝一杯？
飞云：你请我呀？你若是请我，我便和你喝一杯；你若不请我，我便一个人去喝。
书生：小兄弟，看你年纪轻轻，不至于这么势利吧？
飞云：

2.6B_fp16 : 你怎么知道我不势利,你若是不势利,为什么要请我喝酒?
书生:小兄弟,你这话问的好奇怪哦!
飞云:你说什么?
书生:

2.6B_fp32 :我这人最看重的便是信誉。
书生:那是自然。
(这是在下和小兄弟的对话,小兄弟可不要介意。)
飞云:你看我年纪轻轻,不

13B_fp16  :你不知道吗,我这可是为了我自己,才这样做的。
书生:你这不是为了我自己,是为了你自己的前途呀!
飞云:你这是什么话?
书生:
```
```
Input is: 张无忌拿出屠龙宝刀，手起刀落，周芷若掉了一颗门牙，身旁的赵敏喜极而泣，

2.6B_fp16 : 周芷若却只是微微一笑,并不言语。张无忌又拿出倚天剑和屠龙刀,倚天剑和屠龙刀都是他亲手打造,刀刀都锋利无比,倚天剑和屠龙刀的刀身都是

2.6B_fp32 :张无忌又道:“这刀名叫倚天,你拿去给人做鞘,这倚天剑我送给你了。”周芷若喜道:“谢大侠。”张无忌道:“那倚天剑

13B_fp16  :周芷若也流下了幸福的眼泪。 张无忌和周芷若回到了峨嵋派。峨嵋弟子正准备迎接新掌门,却发现掌门被一个蒙面人所杀。张无忌和赵敏急忙赶到峨嵋
```
```
Input is: 人工智能成为国际竞争的新焦点。人工智能是引领未来的战略性技术，世界主要发达国家把发展人工智能作为提升国家竞争力、维护国家安全的重大战略，加紧出台规划和政策，围绕核心技术、顶尖人才、标准规范等强化部署，力图在新一轮国际科技竞争中掌握主导权。当前，

2.6B_fp16 : 人工智能技术已经成为全球创新最活跃、应用最广泛、产业链最长的技术之一。
在我国,人工智能技术也已进入快速发展的新阶段,人工智能技术在教育、医疗、交通、金融

2.6B_fp32 :人工智能正从理论研究向产业应用加速推进,从基础研究向技术研发和产品开发加速推进,从产业应用向民生服务加速推进。人工智能已成为全球创新最活跃、产业发展最具活力、国际竞争

13B_fp16  :人工智能发展呈现出深度感知、人机协同、群智开放、自主操控等趋势和特点,正在深刻改变人类的生产生活方式,成为推动经济社会发展的新引擎。
“人工智能”的英文单词是
```
```
Input is: 中国和美国和日本和法国和加拿大和澳大利亚的首都分别是哪里？

2.6B_fp16 : 中国:北京、上海、广州、香港、重庆、成都。

2.6B_fp32 :北京 (Beijing);上海 (Shanghai);广州 (Guangzhou);重庆 (Chongqing);成都 ( Chengdu);

13B_fp16  :中国的首都是北京,美国的首都是纽约,日本的首都是东京,法国的首都是巴黎。
```

### 报错

- ​	有时会报下面两个错误，如果出现，重新运行

```
FileNotFoundError: [Errno 2] No such file or directory: '/userhome/pclproject/gpt/PanGu-Alpha-GPU/inference/cuda_meta_105367/NotEqual_9416851550248092999.json'
```

```
ModuleNotFoundError: No module named 'tvm'
```



### 附录

#### 源码修改

1、mindspore/train/serialization.py 的 load_param_into_net() 函数

```
def load_param_into_net(net, parameter_dict, strict_load=False,):
    """
    Loads parameters into network.

    Args:
        net (Cell): Cell network.
        parameter_dict (dict): Parameter dictionary.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                           in the param_dict into net with the same suffix. Default: False

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dictionary.

    Examples:
        >>> net = Net()
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> param_not_load = load_param_into_net(net, param_dict)
        >>> print(param_not_load)
        ['conv1.weight']
    """
    if not isinstance(net, nn.Cell):
        logger.error("Failed to combine the net and the parameters.")
        msg = ("Argument net should be a Cell, but got {}.".format(type(net)))
        raise TypeError(msg)

    if not isinstance(parameter_dict, dict):
        logger.error("Failed to combine the net and the parameters.")
        msg = ("Argument parameter_dict should be a dict, but got {}.".format(type(parameter_dict)))
        raise TypeError(msg)

    strict_load = Validator.check_bool(strict_load)
    logger.info("Execute the process of loading parameters into net.")
    net.init_parameters_data()
    param_not_load = []
    for _, param in net.parameters_and_names():
        if param.name in parameter_dict:
            new_param = parameter_dict[param.name]
            new_param = Parameter(Tensor(new_param.asnumpy(), param.dtype), name=param.name)
            if not isinstance(new_param, Parameter):
                logger.error("Failed to combine the net and the parameters.")
                msg = ("Argument parameter_dict element should be a Parameter, but got {}.".format(type(new_param)))
                raise TypeError(msg)
            _update_param(param, new_param)
        else:
            param_not_load.append(param.name)

    if param_not_load and not strict_load:
        _load_dismatch_prefix_params(net, parameter_dict, param_not_load)

    logger.debug("Params not matched(in net but not in parameter_dict):")
    for param_name in param_not_load:
        logger.debug("%s", param_name)

    logger.info("Loading parameters into net is finished.")
    if param_not_load:
        logger.warning("{} parameters in the net are not loaded.".format(len(param_not_load)))
    return param_not_load

```



2、mindspore/nn/layer/basic.py 的 class Dense() 

```
class Dense(Cell):
    r"""
    The dense connected layer.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{kernel} + \text{bias}),

    where :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the inputs created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the inputs created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'.Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If `activation` is not one of str, Cell, Primitive, None.
        ValueError: If length of shape of `weight_init` is not equal to 2 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
        >>> net = nn.Dense(3, 4)
        >>> output = net(input)
        >>> print(output.shape)
        (2, 4)
    """

    @cell_attr_register(attrs=['has_bias', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 dtype=mstype.float32):
        super(Dense, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        self.reshape = P.Reshape()
        self.shape_op = P.Shape()


        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("Weight init shape error.")
        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels], dtype), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("Bias init shape error.")
            self.bias = Parameter(initializer(bias_init, [out_channels], dtype), name="bias")
            self.bias_add = P.BiasAdd()

        self.matmul = P.MatMul(transpose_b=True)
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError("The activation must be str or Cell or Primitive,"" but got {}.".format(activation))
        self.activation_flag = self.activation is not None

    def construct(self, x):
        x_shape = self.shape_op(x)
        check_dense_input_shape(x_shape)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.weight)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)
        return x

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s
```



