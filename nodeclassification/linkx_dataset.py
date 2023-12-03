import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
# from torch_geometric.utils import one_hot

from typing import Optional

import torch
from torch import Tensor


def one_hot(
    index: Tensor,
    num_classes: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Taskes a one-dimensional :obj:`index` tensor and returns a one-hot
    encoded representation of it with shape :obj:`[*, num_classes]` that has
    zeros everywhere except where the index of last dimension matches the
    corresponding value of the input tensor, in which case it will be :obj:`1`.

    .. note::
        This is a more memory-efficient version of
        :meth:`torch.nn.functional.one_hot` as you can customize the output
        :obj:`dtype`.

    Args:
        index (torch.Tensor): The one-dimensional input tensor.
        num_classes (int, optional): The total number of classes. If set to
            :obj:`None`, the number of classes will be inferred as one greater
            than the largest class value in the input tensor.
            (default: :obj:`None`)
        dtype (torch.dtype, optional): The :obj:`dtype` of the output tensor.
    """
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype,
                      device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)

class LINKXDataset(InMemoryDataset):
    r"""A variety of non-homophilous graph datasets from the `"Large Scale
    Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple
    Methods" <https://arxiv.org/abs/2110.14446>`_ paper.

    .. note::
        Some of the datasets provided in :class:`LINKXDataset` are from other
        sources, but have been updated with new features and/or labels.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"penn94"`, :obj:`"reed98"`,
            :obj:`"amherst41"`, :obj:`"cornell5"`, :obj:`"johnshopkins55"`,
            :obj:`"genius"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    github_url = ('https://github.com/CUAI/Non-Homophily-Large-Scale/'
                  'raw/master/data')
    gdrive_url = 'https://drive.google.com/uc?confirm=t&'

    facebook_datasets = [
        'penn94', 'reed98', 'amherst41', 'cornell5', 'johnshopkins55'
    ]

    datasets = {
        'penn94': {
            'Penn94.mat': f'{github_url}/facebook100/Penn94.mat'
        },
        'reed98': {
            'Reed98.mat': f'{github_url}/facebook100/Reed98.mat'
        },
        'amherst41': {
            'Amherst41.mat': f'{github_url}/facebook100/Amherst41.mat',
        },
        'cornell5': {
            'Cornell5.mat': f'{github_url}/facebook100/Cornell5.mat'
        },
        'johnshopkins55': {
            'Johns%20Hopkins55.mat': f'{github_url}/facebook100/Johns%20Hopkins55.mat'
        },
        'genius': {
            'genius.mat': f'{github_url}/genius.mat'
        },
        'wiki': {
            'wiki_views2M.pt':
            f'{gdrive_url}id=1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',
            'wiki_edges2M.pt':
            f'{gdrive_url}id=14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',
            'wiki_features2M.pt':
            f'{gdrive_url}id=1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'
        }
    }

    splits = {
        'penn94': f'{github_url}/splits/fb100-Penn94-splits.npy',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = list(self.datasets[self.name].keys())
        if self.name in self.splits:
            names += [self.splits[self.name].split('/')[-1]]
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for filename, path in self.datasets[self.name].items():
            download_url(path, self.raw_dir)#, filename=filename)
        if self.name in self.splits:
            download_url(self.splits[self.name], self.raw_dir)

    def _process_wiki(self):

        paths = {x.split('/')[-1]: x for x in self.raw_paths}
        x = torch.load(paths['wiki_features2M.pt'])
        edge_index = torch.load(paths['wiki_edges2M.pt']).t().contiguous()
        y = torch.load(paths['wiki_views2M.pt'])

        return Data(x=x, edge_index=edge_index, y=y)

    def _process_facebook(self):
        from scipy.io import loadmat

        mat = loadmat(self.raw_paths[0])

        A = mat['A'].tocsr().tocoo()
        row = torch.from_numpy(A.row).to(torch.long)
        col = torch.from_numpy(A.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        metadata = torch.from_numpy(mat['local_info'].astype('int64'))

        xs = []
        y = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        x = torch.cat([metadata[:, :1], metadata[:, 2:]], dim=-1)
        for i in range(x.size(1)):
            _, out = x[:, i].unique(return_inverse=True)
            xs.append(one_hot(out))
        x = torch.cat(xs, dim=-1)

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.name in self.splits:
            splits = np.load(self.raw_paths[1], allow_pickle=True)
            sizes = (data.num_nodes, len(splits))
            data.train_mask = torch.zeros(sizes, dtype=torch.bool)
            data.val_mask = torch.zeros(sizes, dtype=torch.bool)
            data.test_mask = torch.zeros(sizes, dtype=torch.bool)

            for i, split in enumerate(splits):
                data.train_mask[:, i][torch.tensor(split['train'])] = True
                data.val_mask[:, i][torch.tensor(split['valid'])] = True
                data.test_mask[:, i][torch.tensor(split['test'])] = True

        return data

    def _process_genius(self):
        from scipy.io import loadmat

        mat = loadmat(self.raw_paths[0])
        edge_index = torch.from_numpy(mat['edge_index']).to(torch.long)
        x = torch.from_numpy(mat['node_feat']).to(torch.float)
        y = torch.from_numpy(mat['label']).squeeze().to(torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def process(self):
        if self.name in self.facebook_datasets:
            data = self._process_facebook()
        elif self.name == 'genius':
            data = self._process_genius()
        elif self.name == 'wiki':
            data = self._process_wiki()
        else:
            raise NotImplementedError(
                f"chosen dataset '{self.name}' is not implemented")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}({len(self)})'