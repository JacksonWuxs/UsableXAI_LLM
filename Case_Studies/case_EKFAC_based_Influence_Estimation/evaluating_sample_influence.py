# Author: Yaochen Zhu (uqp4qh@virginia.edu) and Xuansheng Wu (wuxsmail@163.com) 
# Last Modify: 2024-01-11
# Description: Computing the EK-FAC approximated influence function over a corpus.
import os
import re
import json
import tqdm
import nltk
import random
import argparse

import pickle as pkl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch as tc
from libs.core.generator import Generator, zero_grad
from libs.core.hooks import MLPHookController
from libs.core.EKFAC_influence import CovarianceEstimator, InfluenceEstimator
from libs.utils import batchit, CorpusSearchIndex


seed = 12345
random.seed(seed)
tc.manual_seed(seed)
if tc.cuda.is_available():
    tc.cuda.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
    tc.backends.cudnn.benchmark = True

def compute_LM_loss(ids, masks, probs):
    bs, ts = ids.shape
    probs = probs[:, :-1, :].reshape(bs * (ts - 1), -1)
    probs = probs[tc.arange(bs * (ts - 1)), ids[:, 1:].flatten()].reshape(bs, ts - 1)
    return -(masks[:, :-1] * tc.log2(probs)).sum(axis=1) #/ (1e-9 + masks.sum(axis=1))


def compute_pseudo_loss(masks, logits):
    bs, ts = masks.shape
    ids = logits.argmax(dim=-1) # assuming that the pseudo labels are greedy-search generated    
    probs = tc.softmax(logits, -1).reshape(bs * ts, -1)
    probs = probs[tc.arange(bs * ts), ids.flatten()].reshape(bs, ts)
    return -(masks * tc.log2(probs)).sum(axis=1)


def get_sample_indices(num_samples, num_neg, i):
    indices = list(range(num_samples))
    indices.remove(i)
    neg_indices = random.sample(indices, num_neg)
    return [i] + neg_indices


def sentence_tokenize(s):
    return re.split(r'(?<=[^A-Z].[.!?]) +(?=[A-Z])', s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
        help="specify the name of the model")
    parser.add_argument("--chech_point", type=str,
        help="specify the checkpoint for EKFAC")
    args = parser.parse_args()

    model_name = args.model_name
    chech_point = args.chech_point

    root = f"/shared/user/xai"
    inf_root = os.path.join(root, "results", model_name)
    inf_estimator = InfluenceEstimator.load_from_disk(inf_root)
    
    corpus = CorpusSearchIndex("../../datasets/scifact/corpus.txt")
    generator = Generator(f"./outputs/pretrain_scifact_{model_name}/checkpoint-{chech_point}", device="cuda")

    if "gpt2" in model_name:
        hooker = MLPHookController.GPT2(generator._model)
    elif "llama" in model_name:
        hooker = MLPHookController.LLaMA(generator._model)

    topks = [1, 5, 10, 20, 50, 100]
    num_negs = [100, 200, 400, 500]
    recalls = [[[] for _ in topks] for _ in num_negs]

    num_samples = len(corpus)
    num_neg = 499
    
    # Loop over all the queries
    for i, sample in enumerate(corpus):
        sample = sentence_tokenize(sample)
        query, completion = u" ".join(sample[:3]).strip(), u" ".join(sample[3:]).strip()
        completion = generator.generate([query])[0]
        
        # Prepare the input to the LLM
        query, completion, full_text = generator.prepare4explain([query], [completion])
        completion = [completion[0][:1024 - len(query[0])]]
        
        # Forward propagation
        ids = tc.tensor([generator._tokenizer.convert_tokens_to_ids(query[0] + completion[0])])
        probs = tc.softmax(generator._model(input_ids=ids.to(generator._model.device)).logits, dim=-1)
        out_mask = tc.tensor([[0] * len(query[0]) + [1] * len(completion[0])]).to(generator._model.device)
        
        # Calculte the log probability
        query_loss = compute_LM_loss(ids, out_mask, probs)[0]
        query_loss.backward()
        
        # Backward propagation
        with tc.no_grad():
            ### Remove all the weightsn
            query_grads = hooker.collect_weight_grads()
            # use .copy, otherwise zero_grad will remove the grad
            query_grads = {layer:grad.clone() for layer, (_, grad) in query_grads.items()}
        zero_grad(generator._model)
        
        # Calculate the HVP for the query
        query_hvps = inf_estimator.calculate_hvp(query_grads)
        
        influences = []
        bar = tqdm.tqdm(total=len(corpus), desc="Influence for Query=%d" % i)

        sample_idxes = get_sample_indices(num_samples, num_neg, i)
        save_root = os.path.join(root, "results", model_name, "samples")
        os.makedirs(save_root, exist_ok=True)

        with open(os.path.join(save_root, f"{i}.pkl"), "wb") as f:
            pkl.dump(sample_idxes, f)

        for j in sample_idxes:
            # Forward propagation
            sample = corpus[j]
            inputs, outputs = generator.forward([sample])
            losses = compute_pseudo_loss(inputs["attention_mask"], outputs.logits)
            for loss in losses:
                loss.backward(retain_graph=True)
            
            # Backward propagation
            with tc.no_grad():
                grads = hooker.collect_weight_grads()
                inf = inf_estimator.calculate_total_influence(query_hvps, grads)
                influences.append((j, float(inf.cpu().numpy())))
            zero_grad(generator._model)
            bar.update(1)

        for l, num_neg in enumerate(num_negs):
            results_file = os.path.join(root, "results", model_name, f"results_{num_neg}.txt")
            
            pred_rank = [_[0] for _ in sorted(influences[:num_neg], key=lambda x: x[1], reverse=True)]
            for topk, hit in zip(topks, recalls[l]):
                hit.append(1.0 if i in pred_rank[:topk] else 0.0)

            if i % 5 == 0:
                info = "Cases=%d | %s" % (i, u" | ".join("top-%d=%.4f" % (k, sum(hit) / len(hit)) for k, hit in zip(topks, recalls[l])))
                print(f"neg_{num_neg}: " + info)

                ### This is a new run
                if i == 0:
                    with open(results_file, "w") as f:
                        f.write(f"{i}: " + info + "\n")

                    if num_neg == 100:
                        print(influences)
                        inf_path = os.path.join(root, "results", model_name, f"inf_{i}.pkl")
                        with open(inf_path, "wb") as f:
                            pkl.dump(influences, f)

                ### Otherwise we append
                else:
                    with open(results_file, "a") as f:
                        f.write(f"{i}: " + info + "\n")
