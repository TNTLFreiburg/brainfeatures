import pandas as pd
import numpy as np

from brainfeatures.data_set.tuh_abnormal import _read_all_file_names
from brainfeatures.utils.file_util import pandas_store_as_h5


def aggregate_main(in_dir, out_dir, agg_f):
    files = _read_all_file_names(in_dir, extension=".h5", key="natural")
    for file in files:
        data_df = pd.read_hdf(file, key="data")
        info_df = pd.read_hdf(file, key="info")

        cols = data_df.columns
        data = agg_f(data_df.as_matrix(), axis=0, keepdims=True)
        data_df = pd.DataFrame(data, columns=cols)
        pandas_store_as_h5(file.replace(in_dir, out_dir), data_df, key_="data")
        pandas_store_as_h5(file.replace(in_dir, out_dir), info_df, key_="info")


if __name__ == '__main__':
    agg_f = np.median
    in_dir = "/data/schirrmr/gemeinl/tuh-abnormal-eeg/feats/unagged/"
    out_dir = in_dir.replace("unagged", agg_f.__name__)
    aggregate_main(in_dir, out_dir, agg_f)
