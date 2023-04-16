import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from tqdm import tqdm


class YouChooseClickDataset(InMemoryDataset):

    def __init__(self, transform=None, pre_transform=None):
        root = "C:\\Users\\evanh\\Documents\\Datasets\\You-choose"
        super(YouChooseClickDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.download_path = "https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z"

    @property
    def raw_file_names(self):
        return ['youchoose-clicks.dat', 'youchoose-buys.dat']

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def download(self):
        import urllib.request
        urllib.request.urlretrieve("http://www.example.com/songs/mp3.mp3", "mp3.mp3")

    def process(self):

        data_list = []

        click_df = pd.read_csv(self.raw_dir + 'youchoose-clicks.dat', header=None)
        click_df.columns = ['session_id', 'timestamp', 'item_id', 'category']

        buy_df = pd.read_csv(self.raw_dir + 'youchoose-buys.dat', header=None)
        buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

        # Transform the item_ids so that they start at 0
        label_encoder = LabelEncoder()
        click_df['item_id'] = label_encoder.fit_transform(click_df.item_id)

        # Set the labels by checking if each session in clicks is also in buys:
        click_df['label'] = click_df.session_id.isin(buy_df.session_id)

        # process by session ID
        grouped = click_df.groupby('session_id')
        for session_id, group in tqdm(grouped):

            sess_item_id = label_encoder.fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']]\
                .sort_values('sess_item_id')\
                .item_id.drop_duplicates().values
            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [datum for datum in data_list if self.pre_filter(datum)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(datum) for datum in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

