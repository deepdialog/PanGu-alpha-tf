from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
import mindspore as ms

from tokenization_jieba import JIEBATokenizer
from generate import generate
import os
import numpy as np
import argparse

from pangu_dropout_recompute_eos_fp16 import PANGUALPHA as PANGUALPHA_fp16
from pangu_dropout_recompute_eos_fp16 import EvalNet_p
from pangu_wrapcell_gradient_scale_eos import VirtualDatasetOneInputCell
from utils_fix import PANGUALPHAConfig

from pangu_dropout_recompute_eos import PANGUALPHA as PANGUALPHA_fp32

def get_model_13b_fp16(args):
    # model_parallel_num = 8
    model_parallel_num = 1
    data_parallel_num = int(1 / model_parallel_num)
    per_batch_size = 1
    batch_size = per_batch_size * data_parallel_num
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=1024,
        vocab_size=40000,
        embedding_size=5120,  # 5120,  # 353M   8B
        num_layers=40,  # 40,
        num_heads=40,  # ,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,  # 0.0,
        compute_dtype=ms.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        word_emb_dp=True,
        eod_reset=False)
    print("===config is: ", config, flush=True)
    pangu_ = PANGUALPHA_fp16(config)
    # pangu_ = VirtualDatasetOneInputCell(pangu_)
    eval_pangu = EvalNet_p(pangu_, generate=True)
    eval_pangu.set_train(False)
    model = Model(eval_pangu)

    param_dict = load_checkpoint(args.load_ckpt_path)
    load_param_into_net(eval_pangu, param_dict)

    print('#### Load ckpt success!!! ####')
    return model


def get_model_2b6_fp16(args):

    eod_reset = False
    model_parallel_num = 1
    data_parallel_num = int(1 / model_parallel_num)
    per_batch_size = 1
    batch_size = per_batch_size * data_parallel_num

    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=1024,
        vocab_size=40000,
        embedding_size=2560,  # 353M   8B
        num_layers=32,
        num_heads=32,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        compute_dtype=ms.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        word_emb_dp=True,
        eod_reset=eod_reset,
        use_recompute=False)
    print("===config is: ", config, flush=True)

    pangu_ = PANGUALPHA_fp16(config)
    # pangu_ = VirtualDatasetOneInputCell(pangu_)
    eval_pangu = EvalNet_p(pangu_, generate=True)
    eval_pangu.set_train(False)
    model = Model(eval_pangu)

    param_dict = load_checkpoint(args.load_ckpt_path)
    load_param_into_net(eval_pangu, param_dict)

    print('#### Load ckpt success!!! ####')
    return model


def get_model_2b6(args):

    eod_reset = False
    model_parallel_num = 1
    data_parallel_num = int(1 / model_parallel_num)
    per_batch_size = 1
    batch_size = per_batch_size * data_parallel_num

    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=1024,
        vocab_size=40000,
        embedding_size=2560,  # 353M   8B
        num_layers=32,
        num_heads=32,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        compute_dtype=ms.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        word_emb_dp=True,
        eod_reset=eod_reset)
    print("===config is: ", config, flush=True)

    pangu_ = PANGUALPHA_fp32(config)
    # pangu_ = VirtualDatasetOneInputCell(pangu_)
    eval_pangu = EvalNet_p(pangu_, generate=True)
    eval_pangu.set_train(False)
    model = Model(eval_pangu)

    param_dict = load_checkpoint(args.load_ckpt_path)
    load_param_into_net(eval_pangu, param_dict)

    print('#### Load ckpt success!!! ####')
    return model

def run_eval(args):

    ms.context.set_context(save_graphs=False, mode=ms.context.PYNATIVE_MODE, device_target="GPU") #GRAPH_MODE PYNATIVE_MODE

    if args.model == '13B_fp16':
        model_predict = get_model_13b_fp16(args)
    if args.model == '2B6_fp16':
        model_predict = get_model_2b6_fp16(args)
    if args.model == '2B6':
        model_predict = get_model_2b6(args)

    tokenizer_path = os.getcwd() + "/tokenizer"
    tokenizer = JIEBATokenizer(os.path.join(tokenizer_path, 'vocab.vocab'),
                               os.path.join(tokenizer_path, 'vocab.model'))

    samples = ['上联：瑞风播福泽，事业具昌盛千家乐',
               '四川的省会是?',
               '上联：春雨润人间，社会和谐万象新',
               '''书生：羌笛何须怨杨柳，春风不度玉门关。
飞云：（这诗怎么这么耳熟？且过去跟他聊聊如何。）
书生：小兄弟，要不要一起喝一杯？
飞云：你请我呀？你若是请我，我便和你喝一杯；你若不请我，我便一个人去喝。
书生：小兄弟，看你年纪轻轻，不至于这么势利吧？
飞云：''',
               '张无忌拿出屠龙宝刀，手起刀落，周芷若掉了一颗门牙，身旁的赵敏喜极而泣，',
               '人工智能成为国际竞争的新焦点。人工智能是引领未来的战略性技术，世界主要发达国家把发展人工智能作为提升国家竞争力、维护国家安全的重大战略，加紧出台规划和政策，围绕核心技术、顶尖人才、标准规范等强化部署，力图在新一轮国际科技竞争中掌握主导权。当前，',
               '中国和美国和日本和法国和加拿大和澳大利亚的首都分别是哪里？']
    for sample in samples:
        # sample = input("Tell Pangu-alpha what you want to generate:")
        tokenized_token = tokenizer.tokenize(sample)
        start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
        input_ids = np.array(start_sentence).reshape(1, -1)
        output_ids = generate(model_predict, input_ids, 1024, 9)
        output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('Input is:', sample)
        print('Output is:', output_samples[len(sample):], flush=True)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    while 1:
        sample = input("Tell Pangu-alpha what you want to generate:")
        tokenized_token = tokenizer.tokenize(sample)
        start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
        input_ids = np.array(start_sentence).reshape(1, -1)
        output_ids = generate(model_predict, input_ids, 1024, 9, TOPK=1)
        output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('Input is:', sample)
        print('Output is:', output_samples[len(sample):], flush=True)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    return

if __name__ == "__main__":
    #python run_inference.py --model=13B_fp16 --load_ckpt_path=/userhome/temp/PanguAlpha_13b_fp16.ckpt
    #python run_inference.py --model=2B6_fp16 --load_ckpt_path=/userhome/temp/PanguAlpha_2_6b.ckpt
    #python run_inference.py --model=2B6 --load_ckpt_path=/userhome/temp/PanguAlpha_2_6b.ckpt

    parser = argparse.ArgumentParser(description="PANGUALPHA predicting")
    parser.add_argument("--model",
                        type=str,
                        default="13B_fp16",
                        choices=["13B_fp16", "2B6_fp16", "2B6"])
    parser.add_argument("--load_ckpt_path",
                        type=str,
                        default='/userhome/temp/PanguAlpha_13b_fp16.ckpt', #/userhome/temp/PanguAlpha_2_6b.ckpt
                        help="ckpt file path.")

    args_opt = parser.parse_args()
    run_eval(args_opt)


