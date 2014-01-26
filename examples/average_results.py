import argparse
import functools

import h5py
import numpy as np


def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new-fname',
                        help='File in which to save the averaged results dataset',
                        dest='new_fname',
                        required=True,
                        action='store') 
    parser.add_argument('-f', '--fnames',
                        help='File names from which to load the existing results datasets',
                        dest='fnames',
                        metavar='FNAME',
                        required=True,
                        nargs='+',
                        action='store')

    return parser


def save_dataset_names(dset_list, name, obj):
    if isinstance(obj, h5py.Dataset):
        if not name in dset_list:
            dset_list.append(name)

def average_datasets(name, new_file, files):
    try:
        shapes = [f[name].shape for f in files]
    except KeyError:
        print 'Error: files do not have the same structure'
        print 'Exiting...'
        cleanup(new_file, files)
        exit()

    ok = True
    for i, shape in enumerate(shapes):
        for other_shape in shapes[i+1:]:
            ok = (shape == other_shape) and ok

    if not ok:
        print 'Error: shape mismatch for dataset {}'.name
        print 'Exiting...'
        cleanup(new_file, files)
        exit()

    x = np.zeros(files[0][name].shape)
    for _file in files:
        x = x + (1.0/n)*_file[name][...]
    new_file[name] = x

def cleanup(new_file, files):
    new_file.close()
    for _file in files:
        _file.close()


if __name__ == '__main__':
    parser = construct_parser()

    args = parser.parse_args()

    n = float(len(args.fnames))
    files = []
    for fname in args.fnames:
        files.append(h5py.File(fname, 'r'))

    # Collect dataset names from all the files.
    dset_keys = []
    f = functools.partial(save_dataset_names, dset_keys)
    for _file in files:
        _file.visititems(f)

    # Average each dataset.
    new_file = h5py.File(args.new_fname, 'w')
    for name in dset_keys:
        average_datasets(name, new_file, files)

    # Close files.
    cleanup(new_file, files)
