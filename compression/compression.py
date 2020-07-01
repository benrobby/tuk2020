import os
import enum
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json as json


class EncodingType(enum.Enum):
    DictionaryFSBA = 0
    DictionarySIMDBP128 = 1
    FrameOfReferenceFSBA = 2
    FrameOfReferenceSIMDBP128 = 3
    FixedStringDictionaryFSBA = 4
    FixedStringDictionarySIMDBP128 = 5
    Unencoded = 6
    RunLength = 7
    LZ4SIMDBP128 = 8


@dataclass
class EncodedColumn:
    t_id: int
    a_id: int
    t_name: str
    a_name: str
    current_encoding: np.ndarray
    sizes: np.ndarray
    metrics: list  # sorted

    def get_current_size(self) -> float:
        return self.sizes[self.current_encoding]

    def get_best_metric(self) -> float:
        if len(self.metrics) == 0:
            return 0.0
        return self.metrics[0][0]

    # since it's greedy anyways, we only have to look at the best metric for each column. All other will be worse and
    # will not be considered later on because it's greedy.
    def get_best_encoding(self) -> int:
        if len(self.metrics) == 0:
            return 0
        return self.metrics[0][1]


def run():
    with open("runtimes.pickle", 'rb') as input:
        runtimes = pickle.load(input)
    with open("sizes.pickle", 'rb') as input:
        sizes = pickle.load(input)
    assert np.shape(runtimes) == np.shape(sizes)
    attributes = pd.read_csv("attribute_meta_data.csv")

    # it contains negative sizes (doesn't make sense)
    sizes = np.maximum(sizes, 0)

    budgets_factors = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    # need smaller factors. For 0.7, we just take the best metric for each column
    metrics = [
        ('performance', lambda runtime_gains, sizes_losses: runtime_gains),
        ('performance-size', lambda runtime_gains, sizes_losses: runtime_gains / (sizes_losses)),
    ]

    compression_names = {
        'FSBA': "Fixed-size byte-aligned",
        'SIMDBP128': "SIMD-BP128"
    }

    if not os.path.exists("config"):
        os.makedirs("config")

    for metric in metrics:
        for b in budgets_factors:

            all_dict_encoding_size = 0

            encoded_columns = []
            for _, a in attributes.iterrows():
                ids = a['ATTRIBUTE_ID'].split('_')
                t_id = int(ids[0])
                a_id = int(ids[1])

                sizes_for_column = sizes[t_id, a_id]
                runtimes_for_column = runtimes[t_id, a_id]

                indices = np.arange(len(sizes_for_column))
                invalid_encoding_indices = indices[
                    np.logical_and(sizes_for_column == np.finfo(np.float64).max, runtimes_for_column == 0.0)]

                smallest_encoding = np.argmin(sizes_for_column)

                runtime_gains = runtimes_for_column[smallest_encoding] - runtimes_for_column
                sizes_losses = sizes_for_column - sizes_for_column[smallest_encoding]
                metrics = [(m, i) for i, m in enumerate(list(metric[1](runtime_gains, sizes_losses)))]
                metrics = sorted(metrics, key=lambda m: m[0], reverse=True)
                metrics = list(filter(lambda m: m[1] not in invalid_encoding_indices, metrics))

                encoded_columns.append(
                    EncodedColumn(t_id, a_id, a['TABLE_NAME'], a['COLUMN_NAME'], smallest_encoding, sizes_for_column,
                                  metrics))

                all_dict_encoding_size += sizes[t_id, a_id, EncodingType.DictionaryFSBA.value]

            budget_size = b * all_dict_encoding_size

            print(
                f"greedily choosing the encodings with the best metrics. budget {b}, max size {budget_size}, "
                f"metric {metric[0]}")

            current_size = sum(e.get_current_size() for e in encoded_columns)

            metrics = [(e, e.get_best_metric()) for e in encoded_columns]
            metrics = sorted(metrics, key=lambda x: x[1], reverse=True)

            i = 0
            while current_size < budget_size and i < len(metrics):
                # print(current_size)
                metrics[i][0].current_encoding = metrics[i][0].get_best_encoding()
                current_size = sum(e.get_current_size() for e in encoded_columns)
                i += 1

            print(f"stopped at {i} of {len(metrics)}. total size {sum(e.get_current_size() for e in encoded_columns)}")

            output_encodings = {}
            for e in encoded_columns:
                if output_encodings.get(e.t_name) is None:
                    output_encodings[e.t_name] = {}
                encoding = EncodingType(e.current_encoding).name
                compression = None
                for c in ['FSBA', 'SIMDBP128']:
                    if encoding.endswith(c):
                        encoding = encoding[:len(encoding) - len(c)]
                        compression = compression_names.get(c)
                        break

                output_encodings[e.t_name][e.a_name] = {
                    'encoding': encoding
                }

                if compression is not None:
                    output_encodings[e.t_name][e.a_name]['compression'] = compression

            with open("config/encoding_" + metric[0] + "_" + str(b) + ".json", 'w', newline='\n') as f:
                f.write(json.dumps({ 'default': { 'encoding': 'Dictionary' }, 'custom': output_encodings }, indent=4))

    return


if __name__ == '__main__':
    run()
