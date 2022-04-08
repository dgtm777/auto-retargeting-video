import subprocess
import sys
from upload import my_upload


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def get_nn():
    import segmentation_models_pytorch as smp

    model = my_upload()
    net = smp.Unet("mobilenet_v2", in_channels=3, encoder_weights="imagenet")
    state_dict = model["state_dict"]
    net.load_state_dict(state_dict)
    return net
