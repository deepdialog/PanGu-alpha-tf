"""
TopK for text generation
"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
import mindspore as ms


def top_k_logits(logits, top_k=0, top_p=0.9, filter_value=-float(0)):
    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k

        p_args = logits.argsort()[::-1][:top_k]
        mask = np.ones(logits.shape) * filter_value
        mask[p_args] = 1
        logits = logits * mask

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_indices = np.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]

        # cumulative_probs = np.cumsum(softmax(sorted_logits), axis=-1)
        cumulative_probs = np.cumsum(sorted_logits, axis=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def generate(model, origin_inputs, seq_length, end_token=50256, TOPK = 5, max_num=50):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """

    pad_id = 6
    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, pad_id))
    # print("input_ids is ", input_ids)
    cnt = 0
    while valid_length < seq_length:
        inputs = Tensor(input_ids, mstype.int32)
        logits = model.predict(inputs).asnumpy()
        logits = logits.reshape(bs, seq_length, -1)
        probs = logits[0, valid_length-1, :]
        p_args = probs.argsort()[::-1][:TOPK]

        p = probs[p_args]
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        if p_args[target_index] == end_token or valid_length == seq_length-1 or cnt>=max_num:
            outputs = input_ids
            break
        input_ids[0][valid_length] = p_args[target_index]
        valid_length += 1
        cnt += 1

    length = np.sum(outputs != pad_id)
    outputs = outputs[0][:length]
    return outputs



