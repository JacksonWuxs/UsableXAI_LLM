import os


def init_folder(file_path):
    folder = os.path.split(file_path)[0]
    if len(folder) == 0:
        folder = "./"
    os.makedirs(folder, exist_ok=True)


def batchit(X, bs=1, droplast=False):
    batch = []
    for x in X:
        batch.append(x)
        if len(batch) == bs:
            yield batch
            batch.clear()
    if not droplast and len(batch) > 0:
        yield batch


class CorpusSearchIndex:
    def __init__(self, file_path, encoding="utf8", cache_freq=1, sampling=None):
        init_folder(file_path)
        self.fpath = file_path
        self._encoding = encoding
        self._cache_freq = cache_freq
        self._sampling = sampling
        self._build_index()

    def _build_index(self):
        self._lookup, self._numrow, self._doc2idx = [0], 0, {}
        with open(self.fpath, "a+", encoding=self._encoding) as f:
            f.seek(0)
            while self._numrow != self._sampling:
                row = f.readline()
                if len(row) == 0:
                    break
                self._doc2idx[row] = 0
                self._numrow += 1
                if self._numrow % self._cache_freq == 0:
                    self._lookup.append(f.tell())

    def __contains__(self, element):
        return element in self._doc2idx

    def __iter__(self):
        with open(self.fpath, encoding=self._encoding) as f:
            for idx, row in enumerate(f, 1):
                yield row.strip()
                if idx == self._numrow:
                    break

    def __len__(self):
        return self._numrow

    def __getitem__(self, index):
        with open(self.fpath, encoding=self._encoding) as f:
            cacheid = index // self._cache_freq
            f.seek(self._lookup[cacheid])
            for idx, row in enumerate(f, cacheid * self._cache_freq):
                if idx == index:
                    return row.strip()
        raise IndexError("Index %d is out of boundary" % index)

    def append(self, document):
        with open(self.fpath, "a+", encoding=self._encoding) as f:
            f.write(document.replace("\n", "") + "\n")
            self._numrow += 1
            self._doc2idx[document] = len(self._doc2idx)
            if self._numrow % self._cache_freq == 0:
                self._lookup.append(f.tell())

    def clear(self):
        os.remove(self.fpath)
        self._build_index()
