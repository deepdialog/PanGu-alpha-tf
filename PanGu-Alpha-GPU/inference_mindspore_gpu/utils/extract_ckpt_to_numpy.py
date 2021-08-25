
from mindspore import save_checkpoint, load_checkpoint, build_searched_strategy, merge_sliced_parameter
from mindspore import ops
import numpy as np
import os

from mindspore import context

if __name__ == '__main__':

    context.set_context(mode=context.PYNATIVE_MODE)

    pwd = '/userhome/model/'
    # save = '/userhome/model/panguAlpha_13b_fp16_NumpyCkpt/'
    save = '/userhome/model/panguAlpha_2.6b_NumpyCkpt/'
    # save = '/userhome/model/tmp/'
    # file_name = pwd + 'PanguAlpha_2.6B_fp16.ckpt'
    # file_name = pwd + 'PanguAlpha_13b_fp16.ckpt'
    file_name = pwd + 'PanguAlpha_2.6B.ckpt'
    # file_name = '/Users/sam/Downloads/PanguAlpha_2.6B_fp16.ckpt'
    param_dict1 = load_checkpoint(file_name)
    if not os.path.exists(save):
        os.mkdir(save)

    keys_same = []
    for key in param_dict1.keys():
        equal_count = ops.EqualCount()
        parameter = param_dict1[key]
        parameter_name = parameter.name
        parameter_shape = parameter.data.shape
        parameter_shape_length = len(parameter_shape)
        # np.save(save + parameter_name +'.npy', parameter.data.asnumpy())
        print(type(param_dict1[key]))
        print(param_dict1[key])

    pass