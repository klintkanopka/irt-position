import torch
import os
import numpy as np
import pandas as pd

class IRTPDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, run_name):

        d = pd.read_csv(data_path)
        d = d.assign(seq = d['sequence_number'] -1)
        d['seq'] = d['seq'] / d['seq'].max()

        # zero-index people and generate key to match data later
        people = d['id'].unique()
        person_df = pd.DataFrame({
            'id' : people,
            'sid' : list(range(len(people)))
            })

        if run_name is not None:
            path_dir = data_path.split('/')[:-1]
            file_stem = data_path.split('/')[-1][:-4]
            path_dir.append(run_name)
            path_dir = '/'.join(path_dir)
            if not os.path.isdir(path_dir):
                os.mkdir(path_dir)
            path_stem = '/'.join([path_dir, file_stem])
        else:
            path_stem = data_path[:-4]

        person_path = path_stem + '_person_key.csv'
        item_path = path_stem + '_item_key.csv'

        person_df.to_csv(person_path, index=False)

        # zero-index items and generate key to match data later
        items = d['itemkey'].unique()
        item_df = pd.DataFrame({
            'itemkey' : items,
            'ik' : list(range(len(items)))
            })
        item_df.to_csv(item_path, index=False)

        # construct dataframe

        d = pd.merge(d, person_df, how='left', on='id')
        d = d.assign(sid = d['sid'].astype(int))

        d = pd.merge(d, item_df, how='left', on='itemkey')
        d = d.assign(ik = d['ik'].astype(int))

        d = d[['sid', 'ik', 'seq', 'resp']]

        self.path_stem = path_stem

        self.n_persons = len(person_df)
        self.n_items = len(item_df)
        self.responses = d 

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.responses.iloc[idx, 0:3]
        X = np.array(X)
        y = np.array(self.responses.iloc[idx, 3])

        return X, y


