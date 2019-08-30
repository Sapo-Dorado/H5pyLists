from fastai.core import ifnone
from fastai.vision import ImageList, Path, Image, tensor
import pandas as pd
import numpy as np
import h5py

class H5pyList(ImageList):
    def __init__(self, *args, file=None, key=None, **kwargs):
        super().__init__(*args, **kwargs)
        if file == None or key == None:
            raise ValueError('file and key must be defined')
        self.file,self.key = file,key
        self.copy_new += ["file", "key"]
    
    def get(self, i):
        return Image(tensor(self.file[self.key][self.items[i]]))
 
    def label_from_key(self, key, **kwargs):
        return self._label_from_list([self.file[key][i] for i in self.items], **kwargs)
     
    @classmethod
    def from_file(cls, fpath, key=None, idxs=None, **kwargs):
        fpath = Path(fpath)
        file = h5py.File(fpath.name, 'r')
        key = ifnone(key, list(file.keys())[0])
        items = ifnone(idxs, list(range(len(file[key]))))
        return cls(items, path=fpath.parent, file=file, key=key, **kwargs)
