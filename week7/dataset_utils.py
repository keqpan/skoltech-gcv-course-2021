import numpy as np
import pickle
import os
import torch.utils.data as data
from mesh import Mesh


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, dataroot, export_folder, device, batch_size=16,
                 phase='train', num_aug=1, max_dataset_size=float("inf")):

        self.batch_size = batch_size
        self.max_dataset_size = max_dataset_size

        self.dataset = MeshDataset(dataroot, export_folder, device, phase, num_aug)
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=phase == 'train',
            num_workers=3,
            collate_fn=self.collate_fn)
    
    def __len__(self):
        len_ = min(len(self.dataset), self.max_dataset_size)
        if len_ % self.batch_size != 0:
            return len_ // self.batch_size + 1
        return len_ // self.batch_size


    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_dataset_size:
                break
            yield data

    @staticmethod
    def collate_fn(batch):
        """Creates mini-batch tensors
        We should build custom collate_fn rather than using default collate_fn
        """
        meta = {}
        keys = batch[0].keys()
        for key in keys:
            meta.update({key: np.array([d[key] for d in batch])})
        return meta


class MeshDataset(data.Dataset):

    def __init__(self, dataroot, export_folder, device, phase='train',
                 num_aug=1, ninput_edges=750, do_scale_verts=False,
                 flip_edges_prct=0.2, slide_verts_prct=0.2):

        # percent vertices which will be shifted along the mesh surface
        # percent of edges to randomly flip
        # non-uniformly scale the mesh e.g., in x, y or z

        self.num_aug = num_aug
        self.do_scale_verts = do_scale_verts
        self.flip_edges_prct = flip_edges_prct
        self.slide_verts_prct = slide_verts_prct

        self.export_folder = export_folder
        self.ninput_edges = ninput_edges

        self.mean = 0
        self.std = 1
        self.ninput_channels = None

        self.device = device
        self.root = dataroot
        self.dir = os.path.join(dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, phase)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()

        super(data.Dataset, self).__init__()

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(self.root, 'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.num_aug
            self.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
                              'ninput_channels': len(mean)}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            self.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, hold_history=False, export_folder=self.export_folder,
                    num_aug=self.num_aug, do_scale_verts=self.do_scale_verts,
                    flip_edges_prct=self.flip_edges_prct,
                    slide_verts_prct=self.slide_verts_prct)

        meta = {'mesh': mesh, 'label': label}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith('.obj') and (root.count(phase) == 1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes


def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)
