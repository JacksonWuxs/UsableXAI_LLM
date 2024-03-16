import os
import json


class MultiRCDataset:
    def __init__(self, root, subset="test"):
        self._root = root
        self._load_docs(root + "/docs")
        self._load_data(root + "/" + subset + ".jsonl")
        # ["doc", "query", "answer", "evidence"]

    def __iter__(self):
        for row in self._data:
            yield row

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def _clean_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split(" ")

        full, flag = [], False
        for i, token in enumerate(tokens):
            if flag is True and (full[-1] == '"' or full[-1] == "'"):
                full[-1] = full[-1] + token
                continue
            if i > 0 and len(tokens) > i + 1 and (
                token in '&/-' or (token == '.' and\
                                   tokens[i-1][-1].isdigit() and tokens[i+1][0].isdigit())):
                full[-1] = "".join(tokens[i-1:i+2])
                continue
            if i > 0 and token in '.,!?' or token == "'s":
                full[-1] = tokens[i-1] + token
                continue
            if token == '"' or token == "'":
                if flag is False:
                    flag = True
                    full.append(token)
                else:
                    flag = False
                    full[-1] = full[-1] + token
                continue
            full.append(token)
        return " ".join(full)

    def _load_docs(self, doc_folder):
        self._docs = {}
        for each_file in os.listdir(doc_folder):
            with open(doc_folder + '/' + each_file, encoding="utf8") as f:
                tokens = f.read().replace("\n", " ")
            self._docs[each_file] = self._clean_tokens(tokens)

    def _load_data(self, file_path):
        answers, evidences = {}, {}
        with open(file_path, encoding="utf8") as f:
            for row in f:
                row = json.loads(row.strip())
                d = row["annotation_id"].split(":")[0]
                q, a = row["query"].split("||")
                q = d + "|||" + self._clean_tokens(q.strip())
                if q not in answers:
                    answers[q] = ([], [])

                if row["classification"] in (True, "True"):
                    answers[q][0].append(a.strip())
                    if q not in evidences:
                        evidences[q] = [self._clean_tokens(_["text"]) for _ in row["evidences"][0]]
                else:
                    answers[q][1].append(a.strip())

        self._data = []
        for query in sorted(answers):
            d, q = query.split("|||")
            if len(answers[query][0]) == 0:
                continue
            self._data.append((
                self._docs[d], q,
                answers[query][0],
                answers[query][1],
                evidences[query]
                ))



def construct_prompt(row):
    SYSTEM_TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions.\n\nUSER: %s\nASSISTANT:"
    return SYSTEM_TEMPLATE % ("Reading the following paragraph: %s\nMy question is: %s\nKeep your answer simple and use original content if it is possible." % (row[0], row[1]))

