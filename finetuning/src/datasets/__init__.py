from src.datasets.material import TDW_Material
from src.datasets.size import TDW_Size
from src.datasets.position import TDW_Position
from src.datasets.color import TDW_Color
from src.datasets.action import TDW_Action

dataset_types = {
    "material": TDW_Material,
    "size": TDW_Size,
    "position": TDW_Position,
    "color": TDW_Color,
    "action": TDW_Action,
}

if __name__ == "__main__":
    from torch.utils.data import DataLoader, ConcatDataset
    from torchvision.transforms import ToTensor
    from torch.utils.data import ConcatDataset

    transform = ToTensor()
    dataset = TDW_Size(transform=transform)
    dataset = ConcatDataset(
        [tdw_dataset(transform=transform) for tdw_dataset in dataset_types.values()]
    )
    loader = DataLoader(dataset, drop_last=True)
    breakpoint()
