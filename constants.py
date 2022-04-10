import torchvision.transforms as transforms
import os

mask_size = (64, 64)
transform = {
    "train": {
        "images": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(mask_size),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ]
        ),
        "maps": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(mask_size),
            ]
        ),
    },
    "test": {
        "images": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(mask_size),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ]
        ),
        "maps": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(mask_size),
            ]
        ),
    },
}
root_dir = os.path.dirname(__file__)

all_videos = [
    "Fits",
    # "Dogs",
    # "HospitalTrick",
    # "Domino",
    # "Sherbakov",
    # "Masyanya",
    #     "NaskaFigurist",
    #     "NaskaFigurist2",
    #     "DenkaPastuh",
    #     "DenkaPastuh2",
    #     "Sobaki",
    #     "Sobaki2",
    #     "Richi",
]
