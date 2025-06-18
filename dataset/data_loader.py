import numpy as np

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def load_data(config, base_filename, query_filename):
    data = read_fvecs(base_filename)
    queries = read_fvecs(query_filename)
    config.dim = data.shape[1]
    config.num_elements = data.shape[0]
    config.num_queries = queries.shape[0]
    return data, queries
