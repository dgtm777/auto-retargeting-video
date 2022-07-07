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

all_videos = [  # list of videos that will be processed
    "Fits",
    "Dogs",
    "Domino",
    "HospitalTrick",
    "NaskaFigurist",
    "NaskaFigurist2",
    # "trailer",
    "DenkaPastuh",
    "DenkaPastuh2",
    "Sobaki",
    "Richi",
    # "Sherbakov",
    "Masyanya",
    "TheBoysScene",
    "Svaty1",
    "Svaty2",
    # "Svaty3",
    # "Yulka",
    # "Yulka2",
    "Denka1",
]

speed_error = float(0.001)
mask_coef = int(4)
fps_coef = float(1)
speed_coef = int(6000)
prev_speed_coef = float(0.9)
future_speed_coef = float(0.5)
jump_coef_img_size = float(0.25)
jump_coef_wrap_size = float(2 / 3)
jump_coef_mask_value = float(5)
jump_delay_coef = float(0.8)
moving_available_coef = float(1)
scene_detection_parameters = (250, 150)
