import os
from data import srdata


class THERMAL(srdata.SRData):
    def __init__(self, args, name='THERMAL', train=True, benchmark=False):
        super(THERMAL, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'thermal_dataset')
        split = 'train' if self.train else 'test'
        self.dir_hr = os.path.join(self.apath, split, 'HR')
        self.dir_lr = os.path.join(self.apath, split, 'LR')
        self.ext = ('.png', '.png')

    def _scan(self):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        names_hr = sorted([
            os.path.join(self.dir_hr, f)
            for f in os.listdir(self.dir_hr)
            if f.lower().endswith(exts)
        ])
        names_lr = [
            [os.path.join(self.dir_lr, os.path.basename(f)) for f in names_hr]
            for _ in self.scale
        ]
        return names_hr, names_lr