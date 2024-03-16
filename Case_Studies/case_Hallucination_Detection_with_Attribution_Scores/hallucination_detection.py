import os
import string
import pickle as pkl

import nltk
import numpy as np
import transformers as trf
from sentence_transformers import CrossEncoder
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

from pathlib import Path
parent = Path(__file__).parent.parent.parent
sys.path.append(str(parent))

from libs.utils.strings import KMPSearch, tokens2text
from useful import construct_prompt, HallEval2Dataset


POS_IDS = ["EMPTY"]


def compute_pos_mask(tokenizer, tokens, split_func=None):
    types = {}
    s = tokens2text(tokenizer, tokens)
    s = split_func(s) if split_func else s
    for s in nltk.sent_tokenize(s):
        for word, pos in nltk.pos_tag(nltk.word_tokenize(s)):
            if pos not in POS_IDS:
                POS_IDS.append(pos)
            types[word] = POS_IDS.index(pos)

    def match(masks, current_word):
        word = u"".join(current_word).replace(u"\u2581", "")
        label = types.get(word, 0)
        masks.extend([label] * len(current_word))
        current_word.clear()
    
    masks = []
    current_word = []
    for token in tokens:
        if token == "<0x0A>":
            match(masks, current_word)
            masks.append(0)
            continue
        if token.startswith(u"\u2581") or token in string.punctuation:
            match(masks, current_word)
        current_word.append(token)
    if len(current_word) > 0:
        match(masks, current_word)
    assert len(masks) == len(tokens)
    return np.array(masks)
        


def match_spans(prompt, span, tokens, tokenizer):
    masks = [0.] * len(tokens)
    assert span in prompt, span + "---//--->" + prompt
    if prompt[max(0,  prompt.index(span) - 1)] == "\n":
        span = "\n" + span
    span = tokenizer.tokenize(span)
    while len(span) > 0 and span[0] in {u"\u2581", "<0x0A>"}:
        span = span[1:]
    while len(span) > 0 and span[-1] in {u"\u2581", "<0x0A>"}:
        span = span[:-1]
    if len(span) > 0:
        idx = KMPSearch(span, tokens)
        assert idx >= 0, str(span) + str(tokens)
        masks[idx:idx + len(span)] = [1.0] * len(span)
    return np.array(masks)


def compute_density(x, eps=4, p=5):
    x = np.maximum(np.zeros_like(x), x)
    x = x / (x.max(axis=0, keepdims=True) + 1e-9)
    x = np.ceil(x * 10.) 
    x = np.where(x <= eps, 0.0, x) 
    l1 = x.sum(axis=-1)
    lp = (x ** p).sum(axis=-1) ** (1.0 / p) + 1e-9
    return l1 / lp



def evaluate(row, rslt, tokenizer, split_func):
    prompt = construct_prompt(row)
    response = tokens2text(tokenizer, rslt["Output"])
    mask = match_spans(prompt, row[1], rslt["Input"], tokenizer)
    explain = compute_density(rslt["Explain"]* mask.reshape(-1, 1))
    explain = explain / explain.max()
    pos_masks = compute_pos_mask(tokenizer, rslt["Input"], split_func)
    scores = [0] * 120
    for i in range(1, len(POS_IDS) + 1):
        temp_mask = mask * np.where(pos_masks == i, 1., 0.)
        span = explain[temp_mask == 1]
        if len(span) > 0:
            scores[i*2] = np.mean(span)
            scores[i*2+1] = np.max(span)
    return {"Input": prompt, "Output": response, "Scores": scores, "Hallucination": row[3]}



def classification_report(probs, reals, steps=100, thres=0.5):
    reals = np.array(reals)
    acc = accuracy_score(reals, np.where(probs >= thres, 1.0, 0.0))
    best_scores = precision_recall_fscore_support(reals, np.where(probs >= thres, 1.0, 0.0),
                                                  average="binary", zero_division=0.0)[:3]
    return best_scores + (acc,)


def evaluation_ours(dataset, explain_folder, tokenizer, seed):
    X, Y = [], []
    for file in sorted(os.listdir(explain_folder)):
        expl = pkl.load(open(explain_folder + '/' + file, "rb"))
        query = tokenizer.convert_tokens_to_string(expl["Input"])
        query = query.split("USER:")[1].split("ASSISTANT:")[0].strip()
        match, src = 0, None
        for item in dataset:
            if query == item[1]:
                match += 1
                src = item
        if match == 0:
            continue
        assert match == 1
        case = evaluate(src, expl, tokenizer, lambda s: s[s.index("USER:")+5:-11])
        X.append(case["Scores"])
        Y.append(1. if src[3] else 0.)

    X, Y = np.array(X), np.array(Y)
    cols = [i for i, v in enumerate(abs(X).sum(axis=0).tolist()) if v > 0]
    X_train, X_test, Y_train, Y_test = train_test_split(X[:, cols], Y, test_size=0.4, random_state=seed)
    
    X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-9)
    X_test = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-9)
    X_train, Y_train = ADASYN(random_state=seed).fit_resample(X_train, Y_train)

    Y_hat = SVC(probability=True, kernel="rbf", degree=2, C=0.05, random_state=seed).fit(X_train, Y_train).predict_proba(X_test)[:, 1]
    best_scores = classification_report(Y_hat, Y_test)
    print("AttrScore P=%.4f | R=%.4f | F1=%.4f | Acc=%.4f" % best_scores)

    np.random.seed(seed)
    Y_hat = np.random.random(len(Y_hat))
    best_scores = classification_report(Y_hat, Y_test)
    print("Random P=%.4f | R=%.4f | F1=%.4f | Acc=%.4f" % best_scores)


def evaluation_vectara(dataset, explain_folder, tokenizer, seed):
    X, Y = [], []
    for file in sorted(os.listdir(explain_folder)):
        expl = pkl.load(open(explain_folder + '/' + file, "rb"))
        query = tokenizer.convert_tokens_to_string(expl["Input"])
        query = query.split("USER:")[1].split("ASSISTANT:")[0].strip()
        match, src = 0, None
        for item in dataset:
            if query == item[1]:
                match += 1
                src = item
        if match == 0:
            continue
        assert match == 1
        X.append([src[1], src[2]])
        Y.append(1. if src[3] else 0.)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=seed)
    model = CrossEncoder('vectara/hallucination_evaluation_model', device="cuda")
    Y_hat = 1. - model.predict(X_test)
    best_scores = classification_report(Y_hat, Y_test)
    print("Vectara P=%.4f | R=%.4f | F1=%.4f | Acc=%.4f" % best_scores)


def evaluation_factool(explain_folder, golden_data, predict_data, tokenizer, seed):
    X, Y, P = [], [], []
    for file in sorted(os.listdir(explain_folder)):
        expl = pkl.load(open(explain_folder + '/' + file, "rb"))
        query = tokenizer.convert_tokens_to_string(expl["Input"])
        query = query.split("USER:")[1].split("ASSISTANT:")[0].strip()
        match, src = 0, None
        for item in golden_data:
            if query == item[1]:
                match += 1
                src = item
        if match == 0:
            continue
        assert match == 1
        X.append(src[1] + src[2]) # we don't actually use it
        Y.append(1 if src[3] else 0.)
        
        match, src = 0, None
        for item in predict_data:
            if query == item[1]:
                match += 1
                src = item
        if match == 0:
            continue
        assert match == 1
        P.append(src[3])

    P_train, P_test, Y_train, Y_test = train_test_split(P, Y, test_size=0.4, random_state=seed)
    best_scores = classification_report(np.array(P_test), np.array(Y_test))
    print("FacTool P=%.4f | R=%.4f | F1=%.4f | Acc=%.4f" % best_scores)


if __name__ == "__main__":
    results_path = r"./Results/vicuna_7b_v1.5/"
    dataset_path = r"../../datasets/HallEval2/annotation/"
    data = HallEval2Dataset(dataset_path)
    tokenizer = trf.AutoTokenizer.from_pretrained(r"lmsys/vicuna-7b-v1.5", use_fast=False)
    for seed in [1, 2, 3, 4, 5]:
        print("Random Seed:", seed)
        evaluation_ours(data, results_path, tokenizer, seed)
        evaluation_vectara(data, results_path, tokenizer, seed)
        evaluation_factool(results_path, data,
                           HallEval2Dataset(r"../../datasets/HallEval2/prediction"),
                           tokenizer, seed)
