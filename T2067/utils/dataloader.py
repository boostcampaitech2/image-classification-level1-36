from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self,dataframe, transform):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        data = self.df.iloc[idx]
        img_path = data['path']
        image = Image.open(img_path)
        image = self.transform(image)
        label = data['class']
        return image,label


def get_loader(train_data, valid_data, transform, num_workers, batch_size):
    train_dataset = CustomDataset(train_data, transform)
    valid_dataset = CustomDataset(valid_data, transform)
    train_loader = DataLoader(train_dataset,
                              shuffle= True,
                              num_workers = num_workers,
                              batch_size = batch_size
                              )
    valid_loader = DataLoader(valid_dataset,
                              shuffle=False,
                              num_workers = num_workers
                              )
    return train_loader, valid_loader