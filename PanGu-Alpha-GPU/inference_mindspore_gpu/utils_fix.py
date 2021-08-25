"""
network config setting, gradient clip function and dynamic learning rate function
"""
from multiprocessing import Process
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
import numpy as np
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord
from mindspore.parallel._utils import _get_global_rank
from mindspore.communication.management import get_rank, get_group_size
import uuid


class PANGUALPHAConfig:
    """
    PANGUALPHA config class which defines the model size
    """
    def __init__(self,
                 data_parallel_num,
                 model_parallel_num,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=50257,
                 embedding_size=768,
                 num_layers=12,
                 num_heads=12,
                 expand_ratio=4,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 self_layernorm=True,
                 forward_reduce_scatter=True,
                 word_emb_dp=True,
                 stage_num=16,
                 micro_size=32,
                 eod_reset=False,
                 use_top_query_attention=True,
                 use_recompute=True,
                 word_emb_path='',
                 position_emb_path='',
                 top_query_path=''):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.post_layernorm_residual = post_layernorm_residual
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype
        self.use_past = use_past
        self.dp = data_parallel_num
        self.mp = model_parallel_num
        self.self_layernorm = self_layernorm
        self.forward_reduce_scatter = forward_reduce_scatter
        self.stage_num = stage_num
        self.micro_size = micro_size
        self.word_emb_dp = word_emb_dp
        self.eod_reset = eod_reset
        self.use_recompute = use_recompute
        self.use_top_query_attention = use_top_query_attention
        self.word_emb_path = word_emb_path
        self.position_emb_path = position_emb_path
        self.top_query_path = top_query_path


    def __str__(self):
        info = "[PANGUALPHAConfig]" + '===' * 10 + '\n'
        for k, v in self.__dict__.items():
            var_info = "{}:{}\n".format(k, v)
            info += var_info
        info += '=' * 10
        return info


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Tensor")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad ) / value, ())
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    grad = grad * clip_norm / global_norm
    return grad


class GlobalNorm(nn.Cell):
    """

    Calculate the global norm value of given tensors

    """
    def __init__(self, params):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.allreduce_filter = tuple("projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table"
                                      not in x.name for x in params)
        self.length = len(params)
        self.values = []
        self.group_size = get_group_size()
        for item in self.allreduce_filter:
            if item:
                self.values.append(Tensor([1.0], mstype.float32))
            else:
                self.values.append(Tensor([self.group_size*1.0], mstype.float32))
        self.values = tuple(self.values)
    def construct(self, grads):
        square_sum_dp = self.hyper_map(get_square_sum, grads, self.values)
        global_norms = F.sqrt(P.AllReduce()(F.addn(square_sum_dp)))
        return global_norms


class ClipByGlobalNorm(nn.Cell):
    """

    Clip grads by global norm

    """
    def __init__(self, params, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads, global_norm_value


def _get_model_parallel_group(dp, mp):
    rank = _get_global_rank()
    group = range(0, mp)
    index = rank // dp
    return [x + index * mp for x in group]


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PANGUALPHA network.
    """
    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True,
                 lr_scale=0.125):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine
        self.lr_scale = lr_scale

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr * self.lr_scale



class LossSummaryCallback(Callback):
    def __init__(self, summary_dir, local_rank=0, has_trained_epoch=0, has_trained_step=0,
                    bucket='obs://mindspore-file/loss_file/summary/', syn_times=100):
        self._summary_dir = summary_dir
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step

        self.bucket = bucket
        self.syn_times = syn_times

        print("entering")
        self.summary_record = SummaryRecord(self._summary_dir)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.has_trained_step
        # create a confusion matric image, and record it to summary file
        print("writing")
        self.summary_record.add_value('scalar', 'loss', cb_params.net_outputs[0])
        self.summary_record.add_value('scalar', 'scale', cb_params.net_outputs[2])
        if len(cb_params.net_outputs) >3:
            self.summary_record.add_value('scalar', 'global_norm', cb_params.net_outputs[3])
        if len(cb_params.net_outputs) >4:
            self.summary_record.add_value('scalar', 'learning rate', cb_params.net_outputs[4])
        self.summary_record.record(cur_step)

        print("writing finished...",cur_step, self.syn_times)
        if cur_step % self.syn_times == 0:
            print("Copying summary to the bueckets start", flush=True)
            self.summary_record.flush()
            self.syn_files()
            print("Copying summary to the bueckets ends", flush=True)


class StrategySaveCallback(Callback):
    def __init__(self, strategy_path, local_rank=0, has_trained_epoch=0, has_trained_step=0,
                    bucket='obs://mindspore-file/strategy_ckpt/', sym_step=100):
        self.strategy_path = strategy_path
        self.local_rank = local_rank
        self.sym_step = sym_step
        self.bucket = bucket
        self.has_trained_step = has_trained_step
        self.has_send=False
        self.file_name = strategy_path.split('/')[-1]

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.has_trained_step
        if self.has_send is False:
            print("Send strategy file to the obs")
            self.syn_files()
            self.has_send = True


if __name__ == '__main__':

    import os
    class config():
        def __init__(self):
            self.cur_step_num = 100
            self.net_outputs = [Tensor(1), Tensor(1), Tensor(2)]
    class tmp2():
        def original_args(self):
            return config()
    class tmp():
        def __init__(self):
            self.run_context = tmp2()

    strategy_path = '/tmp/startegy.ckpt'
    os.system('touch {}'.format(strategy_path))
    sub_dir = 'exp00/'
    callback = StrategySaveCallback(strategy_path, local_rank=0, has_trained_epoch=0, has_trained_step=0,
                                            bucket='obs://mindspore-file/strategy_ckpt_13b/' + sub_dir, sym_step=4)

    run = tmp2()

    callback.step_end(run)
    callback.step_end(run)
