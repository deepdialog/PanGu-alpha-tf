
import os
import jieba
import numpy as np
from scipy.special import softmax
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from tokenization_jieba import JIEBATokenizer


def create_model_for_provider(model_path: str, provider: str= 'CPUExecutionProvider') -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('NUM_THREADS', 4))
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session


print('model loading...')
tokenizer = JIEBATokenizer(
    'tokenizer/vocab.vocab',
    'tokenizer/vocab.model')
pangu_kv = create_model_for_provider('./onnx_kv_q/pangu.onnx')
jieba.initialize()
kv_cache_start = np.load('kv_cache.npy')
print('model green')


def generate(
    text,
    max_len = 100,
    temperature = 1.0,
    top_p = 0.95,
    top_k = 50,
    eod=None,
    additional_eod=[],
    ban = []
):
    if eod is None:
        eod = [tokenizer.eod_id, tokenizer.eot_id]
    ids = tokenizer.encode(text)
    kv_cache = None

    for i in range(max_len):
        if i == 0:
            logits, kv_cache = pangu_kv.run(None, {
                "input_ids": np.array([ids], dtype=np.int64),
                'kv_cache': kv_cache_start,
            })
        else:
            logits, new_kv = pangu_kv.run(None, {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                'kv_cache': kv_cache,
            })
            kv_cache = np.concatenate([kv_cache, new_kv], axis=-2)

        for x in ban:
            logits[:, -1, x] = -9999

        logits = logits / temperature
        scores = softmax(logits[:, -1, :])
        next_probs = np.sort(scores)[:, ::-1]
        if top_p > 0.0 and top_p < 1.0:
            next_probs = next_probs[:, :int(next_probs.shape[1] * (1 - top_p))]
        if top_k > 0 and top_k < next_probs.shape[1]:
            next_probs = next_probs[:, :top_k]
        next_probs_1 = next_probs / next_probs.sum(axis=1).reshape((-1, 1))

        next_tokens = np.argsort(scores)[:, ::-1]
        if top_p > 0.0 and top_p < 1.0:
            next_tokens = next_tokens[:, :int(next_tokens.shape[1] * (1 - top_p))]
        if top_k > 0 and top_k < next_tokens.shape[1]:
            next_tokens = next_tokens[:, :top_k]

        next_token = np.random.choice(next_tokens[0], p=next_probs_1[0])
        if eod is not None and next_token in eod:
            break
        if next_token in additional_eod or tokenizer.decode([int(next_token)]) in additional_eod:
            break
        ids.append(next_token)
    return tokenizer.decode([int(x) for x in ids])


if __name__ == '__main__':
    print(generate('天下是否太平，取决于'))
