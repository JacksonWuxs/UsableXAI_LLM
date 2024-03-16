import re
import difflib
import string
from collections import Counter, defaultdict

import nltk


def KMPSearch(pat, txt):
    M = len(pat)
    N = len(txt)

    lps = [0] * M
    j = 0

    computeLPSArray(pat, M, lps)

    i = 0
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            return i - j

        elif i < N and pat[j] != txt[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return -1


def computeLPSArray(pat, M, lps):
    length = 0
    i = 1

    while i < M:
        if pat[i] == pat[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
                length = lps[length-1]
        else:
            lps[i] = 0
            i += 1


def check_exactly_match(answer, truth):
    answer = re.sub(r"[^a-zA-Z ]+", "", answer).lower()
    truth = re.sub(r"[^a-zA-Z ]+", "", truth).lower()
    return truth in answer


def check_exactly_match_nostopwords(answer, truth):
    answer = re.sub(r"[^a-zA-Z ]+", "", answer).lower()
    truth = re.sub(r"[^a-zA-Z ]+", "", truth).lower()
    return truth in answer


def clean(tokens):
    new = []
    for token in tokens:
        for p in ["##", "\u0120", "\u2581"]:
            if len(token) > len(p) and token.startswith(p):
                token = token.replace(p, " ")
                break
        new.append(token)
    return new

def drop_stopwords(x):
    tokens = simple_tokenize(x)
    return " ".join([_ for _ in tokens if _.lower() not in stop_words])

# Function to generate N-grams
def generate_ngrams(words, n):
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def find_top_k_ngrams_optimized(words, scores, n, k):
    # Generate N-grams and their corresponding scores
    ngrams = generate_ngrams(words, n)
    ngram_scores = [sum(scores[i:i+n])/n for i in range(len(words) - n + 1)]

    # Pair N-grams with their average scores and count occurrences
    ngram_score_pairs = list(zip(ngrams, ngram_scores))
    ngram_avg_scores = Counter()
    for ngram, score in ngram_score_pairs:
        ngram_avg_scores[ngram] += score

    # Average scores for N-grams appearing multiple times
    for ngram in ngram_avg_scores:
        occurrences = ngrams.count(ngram)
        ngram_avg_scores[ngram] /= occurrences

    return ngram_avg_scores.most_common(k)


def simple_tokenize(s):
    s = s.strip().replace("\n", " ").replace("-", " ")
    words = []
    for w in s.split(" "):
        if len(w) == 0:
            continue
        if w[-1] in string.punctuation:
            words.append(w[:-1])
            words.append(w[-1])
        elif w[0] in string.punctuation:
            words.append(w[0])
            words.append(w[1:])
        else:
            words.append(w)
    return words


def IOU(ngram1, ngram2, threshold=None):
    if isinstance(ngram1, str):
        ngram1 = nltk.word_tokenize(ngram1)
    if isinstance(ngram2, str):
        ngram2 = nltk.word_tokenize(ngram2)
    ngram1 = {_.lower() for _ in ngram1 if _.lower() not in stop_words}
    ngram2 = {_.lower() for _ in ngram2 if _.lower() not in stop_words}
    assert len(ngram1) + len(ngram2) > 0
    score = len(ngram1 & ngram2) / len(ngram1 | ngram2)
    if threshold is None:
        return score
    return score >= threshold


def is_subsequence(smaller, larger):
    # Check if the smaller n-gram is a subsequence of the larger n-gram
    it = iter(larger)
    return all(item in it for item in smaller)


stop_words = set(nltk.corpus.stopwords.words("english") + list(string.punctuation))
def is_similar(ngram1, ngram2, thres1=0.2, thres2=0.1):
    d = difflib.SequenceMatcher(None, ngram1, ngram2)
    d = max(d.get_matching_blocks(), key=lambda x: x[2]).size
    if d / min(len(ngram1), len(ngram2)) >= thres1:
        return True
    ngram1 = {_ for _ in ngram1 if _.lower() not in stop_words}
    ngram2 = {_ for _ in ngram2 if _.lower() not in stop_words}
    if len(ngram1) + len(ngram2) == 0:
        return True
    return len(ngram1 & ngram2) / len(ngram1 | ngram2) >= thres2


def top_k_ngrams(words, scores, K, N):
    # Check if the lists are of the same length
    if len(words) != len(scores):
        raise ValueError("The lengths of words and scores must be the same.")

    # Creating N-grams
    ngram_scores = defaultdict(list)
    for i in range(len(words) - N + 1):
        ngram = tuple(words[i:i+N])
        avg_score = sum(scores[i:i+N]) / N
        ngram_scores[ngram] = avg_score

    # Sorting the N-grams by their average score
    sorted_ngrams = sorted(ngram_scores.items(), key=lambda x: x[1], reverse=True)

    # Returning the top K N-grams
    topK = []
    i = 0
    while len(topK) < K and i < len(sorted_ngrams):
        item = sorted_ngrams[i][0]
        if not any(is_similar(item, exist[0]) for exist in topK):
            topK.append(sorted_ngrams[i])
        i += 1
    return topK


def collect_topK_range_Ngrams(words, scores, K, minN, maxN):
    def update(item, score, finalists):
        #if score > 0.:
        min_item, min_score = None, float('inf')
        for i, s in finalists.items():
            if s < min_score:
                min_item, min_score = i, s
        if score > min_score or len(finalists) < K:
            tmp = "".join(item)
            if not any(tmp in "".join(_) for _ in finalists):
                if len(finalists) == K:
                    del finalists[min_item]
                finalists[item] = score
        
    final = {}
    for n in range(maxN, minN - 1, -1):
        for (item, score) in top_k_ngrams(words, scores, K, n):
            update(item, score, final)
    return [_[0] for _ in sorted(final.items(), key=lambda x: x[1], reverse=True)]



def tokens2text(tokenizer, tokens):
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokenizer.decode(ids, clean_up_tokenization_spaces=True)

