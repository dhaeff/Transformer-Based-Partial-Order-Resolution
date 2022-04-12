
import numpy as np

import util


class Vocabulary():
    def __init__(self, path, transform=None):
        self.index_to_act = {0: "<PAD>", 1: "<SOT>", 2: "<EOT>", 3: "<MASK>", 4: "<SOS>", 5: "<EOS>"}
        self.act_to_index = {"<PAD>": 0, "<SOT>": 1, "<EOT>": 2, "<MASK>": 3, "<SOS>": 4, "<EOS>": 5}
        self.path = path
    def __len__(self):
        return len(self.index_to_act)

    @staticmethod
    def tokenizer(fileName, path, REBUILD_DATA):
        if REBUILD_DATA:
            tokenized_traces = util.load_synthetic_collection(fileName + '.xes')
            np.save(f'{path}/tokenized_traces.npy', tokenized_traces)

        else:
            tokenized_traces = np.load(f"{path}/tokenized_traces.npy", allow_pickle=True)

        return tokenized_traces

    def build_vocabulary(self, traces):
        idx = len(self)
        for sentence in traces:
            for word in sentence:
                if word not in self.act_to_index:
                    self.act_to_index[word] = idx
                    self.index_to_act[idx] = word
                    idx += 1

    def numericalize(self, traces):
        numTraces = []
        for trace in traces:
            events = []
            for event in trace:
                events.append(self.act_to_index[event])
            numTraces.append(events)
        return numTraces

