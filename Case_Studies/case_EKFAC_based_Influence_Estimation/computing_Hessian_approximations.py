# Author: Xuansheng Wu (wuxsmail@163.com) and Yaochen Zhu (uqp4qh@virginia.edu)
# Last Modify: 2024-03-16
# Description: Computing the EK-FAC approximated influence function over a corpus.
import os
import json
import tqdm
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch as tc

from libs.core.generator import Generator, zero_grad
from libs.core.hooks import MLPHookController
from libs.core.EKFAC_influence import CovarianceEstimator, InfluenceEstimator
from libs.utils import batchit, CorpusSearchIndex


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

    batch_size = 2
    corpus = CorpusSearchIndex("../../datasets/scifact/corpus.txt")
    generator = Generator(f"./outputs/pretrain_scifact_{model_name}/checkpoint-{chech_point}"), device="cuda")

    if "gpt2" in model_name:
        hooker = MLPHookController.GPT2(generator._model)
    elif "llama" in model_name:
        hooker = MLPHookController.LLaMA(generator._model)

    estimator = CovarianceEstimator()
    bar_cov = tqdm.tqdm(total=len(corpus), desc="EstimatingCovariance")
    device = generator._model.device
    
    ### Estimating S and A
    for i, texts in enumerate(batchit(corpus, batch_size)):
        texts = [_ for _ in texts if len(_.strip()) > 0]
        if len(texts) == 0:
            continue
        zero_grad(generator._model)
        inputs, outputs = generator.forward(texts)
        losses = compute_pseudo_loss(inputs["attention_mask"],
                                   outputs.logits)
        for loss in losses:
            loss.backward(retain_graph=True)
        with tc.no_grad():
            estimator.update_cov(hooker.collect_states(),
                                 inputs["attention_mask"].to(generator._device))
        bar_cov.update(len(texts))

        if i>=500:
            break
    
    ### Calculating the SVD decomposition of S and A
    estimator.calculate_eigenvalues_and_vectors()

    ### Estimating Lambda
    batch_size = 1 
    bar_lambda = tqdm.tqdm(total=len(corpus), desc="EstimatingLambda")
    for i, texts in enumerate(batchit(corpus, batch_size)):
        texts = [_ for _ in texts if len(_.strip()) > 0]
        if len(texts) == 0:
            continue
        zero_grad(generator._model)
        inputs, outputs = generator.forward(texts)
        losses = compute_pseudo_loss(inputs["attention_mask"], outputs.logits)
        for loss in losses:
            loss.backward(retain_graph=True)
        with tc.no_grad():
            estimator.update_lambdas(hooker.collect_states(),
                                     inputs["attention_mask"].to(generator._device))
        bar_lambda.update(len(texts))
        
        if i>=500:
            break

    save_root = os.path.join(root, "results", model_name)
    os.makedirs(save_root, exist_ok=True)
    estimator.save_to_disk(save_root)
