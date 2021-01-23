import torch
import numpy as np
import pandas as pd

class IRTPDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):

        d = pd.read_csv(data_path)
        d = d.assign(seq = d['sequence_number'] -1)

        # zero-index people and generate key to match data later
        people = d['id'].unique()
        person_df = pd.DataFrame({
            'id' : people,
            'sid' : list(range(len(people)))
            })
        person_path = data_path[:-4] + '_person_key.csv'
        person_df.to_csv(person_path)

        # zero-index items and generate key to match data later
        items = d['itemkey'].unique()
        item_df = pd.DataFrame({
            'itemkey' : items,
            'ik' : list(range(len(items)))
            })
        item_path = data_path[:-4] + '_item_key.csv'
        item_df.to_csv(item_path)

        # construct dataframe

        d = pd.merge(d, person_df, how='left', on='id')
        d = d.assign(sid = d['sid'].astype(int))

        d = pd.merge(d, item_df, how='left', on='itemkey')
        d = d.assign(ik = d['ik'].astype(int))

        d = d[['sid', 'ik', 'seq', 'resp']]

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


