
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChartImageDataset(Dataset):
    def __init__(self, labels_csv: str, transform=None):
        self.df = pd.read_csv(labels_csv, parse_dates=['date']).set_index('date')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert("RGB")
        y = int(row['label'])
        if self.transform is not None:
            img = self.transform(img)
        return img, y
