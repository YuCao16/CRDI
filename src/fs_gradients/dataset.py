from typing import Optional, Tuple
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class FewShotDataset(Dataset):
    def __init__(
        self, csv_file: str, transform: Optional[transforms.Compose] = None
    ) -> None:
        self.img_labels = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = idx

        if self.transform:
            image = self.transform(image)

        return image, label
