import argparse
import numpy as np
from scipy.sparse import csr_matrix



def build_parser():
    parser = argparse.ArgumentParser(description="derive statistics for labels in labels file")
    parser.add_argument('-f', metavar='file_name', type=str, default='',
                        help='path of input file')
    parser.add_argument('-n', metavar='number of labels', type=int, default=-1,
                        help='total number of labels')
    parser.add_argument('-o', metavar='out_file', type=str, default=-1,
                        help='output .spmat file')

    return vars(parser.parse_args())

def parse_label_file(fname):
    with open(fname, "r") as fd:
        labels = []
        unique_labels = set()
        for line in fd:
            curr_labels = line.rstrip().split(',')
            if len(curr_labels) == 1:
                curr_labels = line.rstrip().split('&')
            curr_labels = [int(label) for label in curr_labels]
            labels.append(curr_labels)
            unique_labels.update(curr_labels)
    return labels, unique_labels

def setup_csr_mat(labels, num_total_labels):
    npspmat = np.zeros((len(labels), num_total_labels))
    for i in range(len(npspmat)):
        for label in labels[i]:
            npspmat[i][label-1] = 1
    return csr_matrix(npspmat)

def write_sparse_matrix(mat, fname):
    """ write a CSR matrix in the spmat format """
    with open(fname, "wb") as f:
        sizes = np.array([mat.shape[0], mat.shape[1], mat.nnz], dtype='int64')
        sizes.tofile(f)
        indptr = mat.indptr.astype('int64')
        indptr.tofile(f)
        mat.indices.astype('int32').tofile(f)
        mat.data.astype('float32').tofile(f)

if __name__ == "__main__":
    args = build_parser()
    in_fname, num_lbls, out_fname = args['f'], args['n'], args['o']
    labels, unique_lbls = parse_label_file(in_fname)
    print(len(labels), unique_lbls)
    if num_lbls != len(unique_lbls):
        print(f"Warning: input labels was {num_lbls} while computed num labels was {len(unique_lbls)}")
        print("(Ignore this if you're converting query labels)")

    csr_mat = setup_csr_mat(labels, num_lbls)
    write_sparse_matrix(csr_mat, out_fname)
    print(f"wrote csr matrix to {out_fname}")
