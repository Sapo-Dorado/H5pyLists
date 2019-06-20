from fastai.vision import get_files, ImageList, Path, Image, tensor, df_names_to_idx
import numpy as np
import pandas as pd
import os
import h5py

class H5pyList(ImageList):
    def open(self, fn, dsn):
        with h5py.File(fn, 'r') as f:
            return Image(tensor(f[dsn][()]))
    
    def get(self, idx):
        item = self.items[idx]
        res = self.open(item[0], item[1])
        self.sizes[idx] = res.size
        return res
    
    def _append_h5py_items(item_list, fns):
        for file in fns:
            with h5py.File(file, 'r') as f:
                for name in f:
                    item_list.append([Path(file),name])
        return item_list
    
    def add_prefix_suffix(fns, path, folder, suffix):
        pref = f'{path}{os.path.sep}'
        if folder is not None:
            pref+=f'{folder}{os.path.sep}'
        return np.char.add(np.char.add(pref, fns.astype(str)), suffix)

    @classmethod
    def from_file(cls, path, **kwargs):
        """This function makes an H5pyList from all of the datasets in a given file"""
        items = []
        path = Path(path)
        with h5py.File(path, 'r') as f:
            for name in f:
                items.append((path, name))
        return cls(items, path=path.parent, **kwargs)
    
    @classmethod
    def from_folder(cls, path, recurse=True, **kwargs):
        """This function makes a H5pyList from all the datasets in all the files of a given folder"""
        items = []
        path = Path(path)
        fns = get_files(path, recurse=recurse, extensions='.h5py')
        cls._append_h5py_items(items, fns)
        return cls(items, path=path, **kwargs)
    
    @classmethod
    def from_df(cls, df, path, cols,
                dsn_col=None, folder=None, suffix='', **kwargs):
        """Use dsn_col if your df contains an entry for the name of every dataset paired with its corresponding
           filename. The filenames must be in a single column. This function does not allow you to use
           label_from_df unless you include a dsn_col."""
        if dsn_col is not None:
            return cls.from_detailed_df(df, path, cols, dsn_col, folder, suffix, **kwargs)
        suffix = suffix or ''
        path = Path(path)
        inputs = df.iloc[:,df_names_to_idx(cols,df)]
        assert inputs.isna().sum().sum() == 0, f"You have NaN values in column(s) {cols} of your dataframe, please fix it."
        files = cls.add_prefix_suffix(inputs.values.flatten(), path, folder, suffix)
        items = []
        cls._append_h5py_items(items, files)
        return cls(items, path=path, **kwargs)
    
    @classmethod
    def from_detailed_df(cls, df, path, fn_col, dsn_col,
                        folder=None, suffix='', **kwargs):
        """This function is called by from_df if dsn_col is passed into it."""
        suffix = suffix or ''
        path = Path(path)
        if not (isinstance(fn_col, int) and isinstance(dsn_col, int)):
            raise "Filenames and dataset names must each be in a single column"
        fn_inputs = df.iloc[:,fn_col]
        ds_inputs = df.iloc[:, dsn_col]
        assert fn_inputs.isna().sum().sum() + ds_inputs.isna().sum().sum() == 0, f"You have NaN values in column(s) {[fn_col, dsn_col]} of your dataframe, please fix it."
        files = cls.add_prefix_suffix(fn_inputs.values.flatten(), path, folder, suffix)
        items = [[f, ds] for f,ds in zip(files, ds_inputs.values.flatten())]
        return cls(items, path=path, inner_df=df, **kwargs)
