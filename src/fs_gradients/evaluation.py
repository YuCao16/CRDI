from tqdm import tqdm

import numpy as np

import torch
import lpips
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import FewShotDataset
from .fid_score import calc_fid_score


class Evaluator:
    def __init__(
        self,
        args,
        fake_images,
        fid_npz_path,
        cluster_size,
        device="cuda",
    ):
        assert len(fake_images.shape) == 4

        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.lpips_fn.eval()
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        dataset = FewShotDataset(csv_file=args.csv_file, transform=transform)
        self.real_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.fake_images = fake_images
        self.cluster_size = cluster_size
        self.fid_npz_path = fid_npz_path
        self.args = args

    def calc_intra_lpips(self, device: str = "cuda") -> float:
        cluster = {i: [] for i in range(10)}
        b, _, _, _ = self.fake_images.shape
        for i in tqdm(range(b)):
            dists = []
            for batch in self.real_loader:
                real_image, _ = batch
                if self.args.normalization:
                    real_image = real_image * 2 - 1
                real_image = real_image.to(device)
                with torch.no_grad():
                    dist = self.lpips_fn(
                        self.fake_images[i, :, :, :].unsqueeze(0).cuda(),
                        real_image,
                    )
                    dists.append(dist.item())
            cluster[int(np.argmin(dists))].append(i)

        dists = []
        cluster = {c: cluster[c][: self.cluster_size] for c in cluster}

        for c in tqdm(cluster):
            temp = []
            cluster_length = len(cluster[c])
            for i in tqdm(range(cluster_length)):
                img1 = (
                    self.fake_images[cluster[c][i], :, :, :].unsqueeze(0).cuda()
                )
                for j in range(i + 1, cluster_length):
                    img2 = (
                        self.fake_images[cluster[c][j], :, :, :]
                        .unsqueeze(0)
                        .cuda()
                    )
                    with torch.no_grad():
                        pairwise_dist = self.lpips_fn(img1, img2)
                        temp.append(pairwise_dist.item())
            if temp:
                dists.append(np.mean(temp))
            else:
                print("**************EMPTY****************")
                dists.append(0)
        dists = np.array(dists)
        intra_lpips = dists[~np.isnan(dists)].mean()
        return intra_lpips

    def calc_fid(self):
        return calc_fid_score(
            self.fake_images, self.fid_npz_path, num_workers=0
        )
