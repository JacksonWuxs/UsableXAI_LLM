import json
import os

import numpy as np


class HallEval2Dataset:
    def __init__(self, root):
        self._root = root
        self._data = []
        self._load_data(root)

    def __iter__(self):
        for row in self._data:
            yield row

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def _load_data(self, folder_path):
        for file in sorted(os.listdir(folder_path)):
            if not file.endswith(".json"):
                continue
            samples = json.load(open(folder_path + '/' + file, encoding="utf8"))
            key = "human_judge" if "human_judge" in samples[0] else "chatgpt_judge"
            
            for row in samples:
                if key == "human_judge":
                    if len(row[key]) == 0:
                        continue
                    truth = [1. if "true" == _ else 0.0 for _ in row[key]]
                    hardness = max(1. - np.mean(truth), np.mean(truth))
                    if min(truth.count(1.), truth.count(0.)) > 1:
                        continue
                    self._data.append([
                        row["id"],
                        row["user_query"],
                        row["chatgpt_response"],
                        any("false" in _ for _ in row[key]),
                        file[:-5]
                        ])
                else:
                    truth = [1. if "true" in _ else 0.0 for _ in row[key]]
                    self._data.append([
                        row["id"],
                        row["user_query"],
                        row["chatgpt_response"],
                        1.0 - np.mean(truth),
                        file[:-5]
                        ])


def construct_prompt(row):
    SYSTEM_TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. " +\
                      "The assistant gives helpful, detailed, and polite answers to the user's questions. " +\
                      "USER: %s ASSISTANT:"
    return SYSTEM_TEMPLATE % (row[1],)
