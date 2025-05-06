import os
import vector
import numpy as np
import awkward as ak
import pandas as pd
import uproot
import torch
from collections.abc import Mapping
from functools import reduce
from torch.utils.data import Dataset


class Data(Mapping):
    """ Cache for loading data from tree """

    def __init__(self, name, files, treenames, N=None):
        self.name = name
        self.trees = []
        self.entries = []
        self.files = files if isinstance(files, (list, tuple)) else [files]
        self.treenames = treenames if isinstance(
            treenames, (list, tuple)) else [treenames]
        self.data = {'file': np.array([]), 'tree': np.array(
            []), 'sample': np.array([])}
        self.idx = None
        for f in self.files:
            if isinstance(f, uproot.reading.ReadOnlyDirectory):
                F = f
            elif isinstance(f, str):
                if not os.path.isfile(f):
                    raise RuntimeError(f'File {f} does not exist')
                F = uproot.open(f)
            else:
                raise ValueError
            for treename in self.treenames:
                if treename not in F.keys():
                    print(f'Tree {treename} is not in file {f}')
                    continue
                tree = F[treename]
                n_tree = tree.num_entries if N is None else min(
                    N, tree.num_entries)
                self.trees.append(tree)
                self.entries.append(n_tree)
                self.data['file'] = np.concatenate(
                    (self.data['file'], np.array([F.file_path]*n_tree)), axis=0)
                self.data['tree'] = np.concatenate(
                    (self.data['tree'], np.array([tree.name]*n_tree)), axis=0)
                self.data['sample'] = np.concatenate(
                    (self.data['sample'], np.array([os.path.basename(F.file_path)]*n_tree)), axis=0)

    def __getitem__(self, key):
        if key not in self.keys():
            arrays = []
            for entries, tree in zip(self.entries, self.trees):
                if key not in tree.keys():
                    raise KeyError(
                        f'Key {key} is not present in tree {tree.name} of file {tree.file.file_path}')
                arrays.append(tree[key].array()[:entries])
            self.data[key] = np.array(ak.concatenate(arrays, axis=0))
            if self.idx is not None:
                self.data[key] = self.data[key][self.idx]
        return self.data[key]

    def __setitem__(self, key, val):
        if key in self.keys():
            print(f'Key {key} already in the data, will modify it')
        self.data[key] = val

    def __len__(self):
        return sum(self.entries)

    def __iter__(self):
        return self.data.__iter__()

    def delete(self, key):
        if key not in self.keys():
            print(f'Key {key} is not registered')
        else:
            del self.data[key]

    def cut(self, mask):
        idx = np.where(mask)[0]
        self._apply_idx(idx)
        # Need to keep track of the total index for further loading
        self._store_idx(idx)

    def _apply_idx(self, idx):
        for key in self.keys():
            self[key] = self[key][idx]

    def _store_idx(self, idx):
        if self.idx is None:
            self.idx = idx
        else:
            self.idx = np.intersect1d(idx, self.idx)

    @property
    def branches(self):
        return list(set([k for t in self.trees for k in t.keys()]))

    @property
    def N(self):
        return sum(self.entries)

    def keys(self):
        return self.data.keys()

    def make_particles(self, name, px, py, pz, E):
        if name in self.keys():
            print(f'{name} is already in data, will not modify it')
            return
        for br in [px, py, pz, E]:
            if br not in self.branches:
                raise RuntimeError(f'Branch {br} not found in file')
        self.data[name] = vector.arr(
            {
                "x": self[px],
                "y": self[py],
                "z": self[pz],
                "E": self[E],
            },
        )

    def select_branches(self, name, branches, index):
        if name in self.keys():
            print(f'{name} is already in data, will not modify it')
            return
        if not isinstance(index, np.ndarray):
            index = np.array(index)
        values = [self[br] for br in branches]
        if all([isinstance(val, ak.Array) for val in values]):
            concat = np.concatenate([np.array(val).reshape(-1, 1)
                                    for val in values], axis=1)
            self[name] = ak.from_numpy(np.take_along_axis(
                concat, index[:, None], axis=1).ravel())
        elif all([isinstance(val, np.ndarray) for val in values]):
            if all([isinstance(val, vector.MomentumNumpy4D) for val in values]):
                pxs = np.concatenate([val.px.reshape(-1, 1)
                                     for val in values], axis=1)
                pys = np.concatenate([val.py.reshape(-1, 1)
                                     for val in values], axis=1)
                pzs = np.concatenate([val.pz.reshape(-1, 1)
                                     for val in values], axis=1)
                es = np.concatenate([val.e.reshape(-1, 1)
                                    for val in values], axis=1)
                self.data[name] = vector.arr(
                    {
                        "x": np.take_along_axis(pxs, index[:, None], axis=1),
                        "y": np.take_along_axis(pys, index[:, None], axis=1),
                        "z": np.take_along_axis(pzs, index[:, None], axis=1),
                        "E": np.take_along_axis(es,  index[:, None], axis=1),
                    },
                )
            else:
                concat = np.concatenate([val.reshape(-1, 1)
                                        for val in values], axis=1)
                self[name] = np.take_along_axis(
                    concat, index[:, None], axis=1).ravel()
        else:
            raise RuntimeError(
                f'Mix of types : {[type(val) for val in values ]}')

    def merge(self, suffix, datas, suffixes, merge_branch, branches_to_merge):
        # We want to compare the merged_branch, but needs to be done per file and tree
        # ie, to avoid cases where there is coverage between samples
        # Here find the common samples and trees between datasets
        common_samples = list(
            set([s for data in [self]+datas for s in np.unique(data['sample'])]))
        common_trees = list(
            set([t for data in [self]+datas for t in np.unique(data['tree'])]))
        # Make the common indices arrays #
        # self[merge_branch][(self['tree'] == common_trees[0]) & (self['sample'] == common_samples[0])]
        idx = np.array([], dtype=np.int64)  # for self object
        idxs = [np.array([], dtype=np.int64)
                for _ in range(len(datas))]  # for to merged datasets
        for sample in common_samples:
            for tree in common_trees:
                # Find the common values for the same file and tree #
                masks = [(data['tree'] == tree) & (data['sample'] == sample)
                         for data in [self]+datas]
                matched = reduce(np.intersect1d, [
                                 data[merge_branch][mask] for data, mask in zip([self]+datas, masks)])
                # Find indices for each of the datas
                idx_tmp = np.nonzero(np.in1d(self[merge_branch], matched))[0]
                idx = np.concatenate((idx, idx_tmp[masks[0][idx_tmp]]), axis=0)
                for i in range(len(idxs)):
                    idx_tmp = np.nonzero(
                        np.in1d(datas[i][merge_branch], matched))[0]
                    idxs[i] = np.concatenate(
                        (idxs[i], idx_tmp[masks[i+1][idx_tmp]]), axis=0)
        # Safety check that merged branch is the same #
        for i in range(len(datas)):
            if np.any(self[merge_branch][idx] != datas[i][merge_branch][idxs[i]]):
                raise RuntimeError(
                    f'Merged branch {merge_branch} does not match between current data and entry {i}')
        # Need to restrict the current loaded branches on the intersection index now
        self._apply_idx(idx)
        # Modify the branch name to fix the suffix
        for br in branches_to_merge:
            # In case branch is not loaded, call it to be sure and select with index
            self[br] = self[br][idx]
            # Change the key name
            self[f'{br}_{suffix}'] = self.data.pop(br)
            print(f'Changing branch name : {br:20s} -> {f"{br}_{suffix}":20s}')
        # Add the branches to be merged
        for i in range(len(datas)):
            for br in branches_to_merge:
                self[f'{br}_{suffixes[i]}'] = datas[i][br][idxs[i]]
                print(
                    f'Merging branch from merge entry {i} : {br:20s} -> {f"{br}_{suffixes[i]}":20s}')
        # Record the current object index for further loading
        self._store_idx(idx)

    @property
    def get_df(self):
        return pd.DataFrame({key: self[key] for key in self.keys()})


class TransformerDataset(Dataset):
    def __init__(self, df, variable_sets, device):
        self.variable_vecs = torch.stack(
            [torch.tensor(df[var].values, dtype=torch.float32).to(device) for var in variable_sets], dim=1
        )

    def __len__(self):
        return self.variable_vecs.shape[0]

    def __getitem__(self, idx):
        return self.variable_vecs[idx]
