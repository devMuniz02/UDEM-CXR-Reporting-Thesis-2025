from typing import Callable, Optional
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from utils.data.mimic_dataset import MIMICDataset
from utils.data.chexpert_dataset import CHEXPERTDataset
from utils.processing import loader, image_transform

def create_dataloaders(chexpert_paths, 
                       mimic_paths, 
                       batch_size: int = 32,
                       split: str = "train", 
                       transform: Optional[Callable] = None,
                       sampling_ratio: float = 0.7,
                       findings_or_impression: str = "findings",
                       **kwargs) -> DataLoader:
    MIMIC_df, CHEXPERT_df = loader(chexpert_paths, mimic_paths, split=split)
    transform = image_transform(img_size=512) if transform is None else transform
    if findings_or_impression == "findings":
        mimic_ds = MIMICDataset(
            MIMIC_df, 
            mimic_paths["mimic_images_dir"], 
            mimic_paths["mimic_data_path"], 
            transform=transform)
            
        output_loader = DataLoader(
            mimic_ds,
            batch_size=batch_size,
            shuffle=True if split != "test" else False,
            **kwargs
        )

    elif findings_or_impression == "impression":  # "impression"
        chexpert_ds = CHEXPERTDataset(
            CHEXPERT_df,
            chexpert_paths["chexpert_data_path"],
            split=split,
            transform=transform
        )

        output_loader = DataLoader(
            chexpert_ds,
            batch_size=batch_size,
            shuffle=True if split != "test" else False,
            **kwargs
        )

    else:
        mimic_ds = MIMICDataset(
            MIMIC_df, 
            mimic_paths["mimic_images_dir"], 
            mimic_paths["mimic_data_path"], 
            transform=transform)
        chexpert_ds = CHEXPERTDataset(
            CHEXPERT_df,
            chexpert_paths["chexpert_data_path"],
            split=split,
            transform=transform
        )
        mixed = ConcatDataset([mimic_ds, chexpert_ds])
        n1, n2 = len(mimic_ds), len(chexpert_ds)

        p1, p2 = sampling_ratio, 1 - sampling_ratio  # desired sampling ratio

        # per-sample weights: higher weight → sampled more often
        w1 = torch.full((n1,), fill_value=p1 / max(n1, 1), dtype=torch.float)
        w2 = torch.full((n2,), fill_value=p2 / max(n2, 1), dtype=torch.float)
        weights = torch.cat([w1, w2])

        sampler = WeightedRandomSampler(weights, num_samples=n1 + n2, replacement=True)

        output_loader = DataLoader(
            mixed,
            batch_size=batch_size,
            shuffle=True if split != "test" else False,
            sampler=sampler,
            **kwargs
        )
        
    return output_loader
    

# from typing import Callable, Optional
# import torch
# from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
# from utils.data.mimic_dataset import MIMICDataset
# from utils.data.chexpert_dataset import CHEXPERTDataset
# from utils.processing import loader, image_transform

# def create_dataloaders(chexpert_paths, 
#                        mimic_paths, 
#                        batch_size: int = 32,
#                        split: str = "train", 
#                        transform: Optional[Callable] = None,
#                        sampling_ratio: float = 0.7,
#                        findings_or_impression: str = "findings",
#                        **kwargs) -> DataLoader:
#     MIMIC_df, CHEXPERT_df = loader(chexpert_paths, mimic_paths, split=split)
#     transform = image_transform(img_size=512) if transform is None else transform
#     if findings_or_impression not in ["findings", "impression"]:
#         findings_or_impression = "findings"
#         print(f"Invalid findings_or_impression value. Defaulting to 'findings'.")

#     if findings_or_impression == "findings":
#         chexpert_ds = CHEXPERTDataset(
#             CHEXPERT_df,
#             chexpert_paths["chexpert_data_path"],
#             split=split,
#             transform=transform,
#             text_col='section_findings',
#         )

#     elif findings_or_impression == "impression":  # "impression"
#         chexpert_ds = CHEXPERTDataset(
#             CHEXPERT_df,
#             chexpert_paths["chexpert_data_path"],
#             split=split,
#             transform=transform,
#             text_col='section_impression',
#         )

#     mimic_ds = MIMICDataset(
#         MIMIC_df, 
#         mimic_paths["mimic_images_dir"], 
#         mimic_paths["mimic_data_path"], 
#         transform=transform)
    
#     if split == "train":
#         print("Len MIMIC Dataset:", len(mimic_ds), "Len CheXpert Dataset:", len(chexpert_ds))
#         n1, n2 = len(mimic_ds), len(chexpert_ds)
#         sampler = None
#         if n1 / (n1 + n2) < sampling_ratio:  # n1/(n1+n2) is less than sampling ratio
#             p1, p2 = sampling_ratio, 1 - sampling_ratio  # desired sampling ratio

#             # per-sample weights: higher weight → sampled more often
#             w1 = torch.full((n1,), fill_value=p1 / max(n1, 1), dtype=torch.float)
#             w2 = torch.full((n2,), fill_value=p2 / max(n2, 1), dtype=torch.float)
#             weights = torch.cat([w1, w2])

#             sampler = WeightedRandomSampler(weights, num_samples=n1 + n2, replacement=True)

#         mixed = ConcatDataset([mimic_ds, chexpert_ds])

#         output_loader = DataLoader(
#             mixed,
#             batch_size=batch_size,
#             shuffle=False,
#             sampler=sampler,
#             **kwargs
#         )
        
#         return output_loader
#     else:
#         output_loader = DataLoader(
#             mimic_ds,
#             batch_size=batch_size,
#             shuffle=True if split != "test" else False,
#             **kwargs
#         )
            
#         return output_loader
    