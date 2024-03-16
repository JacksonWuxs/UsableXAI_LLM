import os
import re
import sys
import string
import pickle as pkl

import nltk
import numpy as np
import torch as tc
import transformers as trf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from pathlib import Path
parent = Path(__file__).parent.parent.parent
sys.path.append(str(parent))

from libs.utils.strings import KMPSearch, tokens2text
from useful import construct_prompt, MultiRCDataset


stop_words = set(nltk.corpus.stopwords.words("english") + list(string.punctuation))


def check_exactly_match(answer, truth):
    answer = re.sub(r"[^a-zA-Z ]+", "", answer).lower()
    truth = re.sub(r"[^a-zA-Z ]+", "", truth).lower()
    return truth in answer

def drop_stopwords(s):
    s = s.strip().replace("\n", " ").replace("-", " ")
    tokens = []
    for w in s.split(" "):
        if len(w) == 0:
            continue
        if w[-1] in string.punctuation:
            tokens.append(w[:-1])
            tokens.append(w[-1])
        elif w[0] in string.punctuation:
            tokens.append(w[0])
            tokens.append(w[1:])
        else:
            tokens.append(w)
    return " ".join([_ for _ in tokens if _.lower() not in stop_words])


def match_query(prompt, instruct, tokens, tokenizer):
    masks = [0.] * len(tokens)
    for sid, span in enumerate(instruct.split("|||"), 1):
        if len(span) == 0:
            continue
        assert span in prompt, span + "---//--->" + prompt
        if prompt[max(0,  prompt.index(span) - 1)] == "\n":
            span = "\n" + span
        span = tokenizer.tokenize(span)
        while len(span) > 0 and span[0] in {u"\u2581", "<0x0A>", u'\u2581"'}:
            span = span[1:]
        while len(span) > 0 and span[-1] in {u"\u2581", "<0x0A>", u'\u2581"'}:
            span = span[:-1]
        if len(span) > 0:
            idx = KMPSearch(span, tokens)
            assert idx >= 0, str(span) + str(tokens)
            masks[idx:idx + len(span)] = [1.0] * len(span)
    return np.array(masks)


def compute_density(x, eps=2, p=5):
    x = np.maximum(np.zeros_like(x), x)
    x = x / (x.max(axis=0, keepdims=True) + 1e-9)
    x = np.ceil(x * 10.)
    x = np.where(x <= eps, 0.0, x) 
    l1 = x.sum(axis=-1)
    lp = (x ** p).sum(axis=-1) ** (1.0 / p) + 1e-9
    x = l1 / lp
    return x


def compute_density(x, eps=4, p=5):
    x = np.maximum(np.zeros_like(x), x)
    x = x / (x.max(axis=0, keepdims=True) + 1e-9)
    x = np.ceil(x * 10.) 
    x = np.where(x <= eps, 0.0, x) 
    l1 = x.sum(axis=-1)
    lp = (x ** p).sum(axis=-1) ** (1.0 / p) + 1e-9
    return l1 / lp


def highlight_sentences(explains, tokens, special=u"\u2581", skips={"<0x0A>", ".", "!", "?"}, topK=2):
    assert len(explains) == len(tokens)
    words, scores = [], []
    current_word, current_score = [], []
    for i, (e, w) in enumerate(zip(explains, tokens)):
        if w.startswith(special) or w in skips:
            if len(current_word) > 0:
                words.append(u"".join(current_word).replace(special, ""))
                scores.append(np.max(current_score))
                current_word, current_score = [], []

        current_word.append(w)
        current_score.append(e)
    if len(current_word) > 0:
        words.append(u"".join(current_word))
        scores.append(np.max(current_score))
    truncate_words, truncate_scores, flag = [], [], False
    for i, (word, score) in enumerate(zip(words, scores)):
        if flag:
            truncate_words.append(word)
            truncate_scores.append(score)
            if " ".join(words[i+1:i+4]) == "<0x0A>My question is:":
                break
        if " ".join(words[i-3:i+1]) == 'Reading the following paragraph:':
            flag = True

    assert len(truncate_words) == len(truncate_scores)


    sentences, scores, tmp_w, tmp_s = [], [], [], []
    all_sentences = iter(nltk.sent_tokenize(" ".join(truncate_words)))
    current_sent = next(all_sentences)
    for w, s in zip(truncate_words, truncate_scores):
        tmp_w.append(w)
        tmp_s.append(s)
        if " ".join(tmp_w) == current_sent:
            sentences.append(" ".join(tmp_w))
            scores.append(np.mean(tmp_s))
            try:
                current_sent = next(all_sentences)
            except StopIteration:
                pass
            tmp_w.clear()
            tmp_s.clear()
    assert len(tmp_w) == 0
    sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [_ for _, __ in sentences[:topK]]
    



class CustomTrainer(trf.Trainer):

    def __init__(self, pos_weight, **kwrds):
        self.pos_weight = pos_weight
        trf.Trainer.__init__(self, **kwrds)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = tc.nn.BCEWithLogitsLoss(weight=tc.tensor([self.pos_weight], device=model.device))
        loss = loss_fct(logits, labels.unsqueeze(-1))
        return (loss, outputs) if return_outputs else loss


def build_bert_classifier(X_train, Y_train, X_test, seed, name="distilbert-base-uncased"):
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, set_seed,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(device)
    ds = Dataset.from_dict({"text": X_train, "label": Y_train}).map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    args = TrainingArguments(
        output_dir="./results/",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        seed=seed,
        metric_for_best_model="accuracy",
        weight_decay=1e-3)
    trainer = CustomTrainer(
        pos_weight=len(Y_train) / sum(Y_train),
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds.select(range(10)),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )
    model.train()
    trainer.train()
    model.eval()
    scores = []
    with tc.no_grad():
        for x in X_test:
            inputs = tokenizer(x, truncation=True, padding=True, return_tensors="pt")
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            scores.append(tc.sigmoid(outputs.logits)[0].item())
    return np.array(scores)


def classification_report(probs, reals, step=0.005):
    best_scores, best_thres = (None, None, -float("inf")), None
    for thres in np.arange(step, 1.0 + step, step):
        p, r, f = precision_recall_fscore_support(reals, np.where(probs >= thres, 1.0, 0.0),
                                                  average="macro", zero_division=0.0)[:3]
        if f > best_scores[-1]:
            best_scores, best_thres = (p, r, f), thres
    auc = roc_auc_score(reals, probs)
    best_scores = precision_recall_fscore_support(reals, np.where(probs >= best_thres, 1.0, 0.0),
                                                  average="macro", zero_division=0.0)[:3]
    return best_scores + (auc,)

def evaluate(row, rslt, tokenizer):
    prompt = construct_prompt(row)
    mask = match_query(prompt, row[0], rslt["Input"], tokenizer)
    response = tokens2text(tokenizer, rslt["Output"])
    explain = compute_density(rslt["Explain"] * mask.reshape(-1, 1))
    highlight = highlight_sentences(explain, rslt["Input"], topK=len(row[-1]))
    return {"Input": prompt, "Output": response, "Highlight": highlight, "Rationale": row[-1],
            "Query": row[1], "Correct": row[2], "Wrong": row[3], "Document": row[0]}


def exactly_match(query, response, correct, wrong, delta=0):
    errors = 0.0
    response = drop_stopwords(response)
    correct = sorted([drop_stopwords(_) for _ in correct], key=len)
    duplicate = []
    for i, word in enumerate(correct):
        word = word.lower()
        for each in correct[:i]:
            if word in each.lower():
                duplicate.append(i)
    for idx in duplicate[::-1]:
        correct.pop(idx)
    for each_candidate in correct:
        if not check_exactly_match(response, each_candidate):
            errors += 1

    wrong = sorted([drop_stopwords(_) for _ in wrong], key=len)
    duplicate = []
    for i, word in enumerate(wrong):
        word = word.lower()
        for each in wrong[:i]:
            if word in each.lower():
                duplicate.append(i)
    for idx in duplicate[::-1]:
        wrong.pop(idx)
    for each_candidate in wrong:
        if check_exactly_match(response, each_candidate):
            errors += 1
    return 1. if errors <= delta else 0.


def evaluation(dataset, dtype, seed, tokenizer):
    X, Y = [], []
    for i, src in enumerate(dataset, 1):
        if not os.path.isfile(results_path + str(i) + ".pkl"):
            continue
        case = evaluate(src, pkl.load(open(results_path + str(i) + ".pkl", "rb")), tokenizer)
        if dtype == "Rationale":
            X.append(" ".join(case["Rationale"]) + "\n" + case["Output"])
        elif dtype == "Highlight":
            X.append(" ".join(case["Highlight"]) + "\n" + case["Output"])
        elif dtype == "Document":
            X.append(case["Document"] + "\n" + case["Output"])
        Y.append(exactly_match(case["Query"], case["Output"], case["Correct"], case["Wrong"]))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    Y_hat = build_bert_classifier(X_train, Y_train, X_test, seed)
    best_scores = classification_report(Y_hat, Y_test)
    print("Input=%s | P=%.4f | R=%.4f | F1=%.4f | AUC=%.4f" % ((dtype,) + best_scores))
    if dtype == "Rationale":
        np.random.seed(seed)
        Y_hat = np.random.random(len(Y_test))
        best_scores = classification_report(Y_hat, Y_test)
        dtype = "Random"
        print("Input=%s | P=%.4f | R=%.4f | F1=%.4f | AUC=%.4f" % ((dtype,) + best_scores))


if __name__ == "__main__":
    device = "cuda"
    results_path = r"./Results/vicuna_7b_v1.1/multiRC_"
    dataset_path = r"../../datasets/MultiRC/"
    data = MultiRCDataset(dataset_path)
    tokenizer = trf.AutoTokenizer.from_pretrained(r"lmsys/vicuna-7b-v1.5", use_fast=False)
    for seed in [0, 1, 2, 3, 4]:
        print("Seed:", seed)
        for dtype in ["Rationale", "Highlight", "Document"]:
            evaluation(data, dtype, seed, tokenizer)
