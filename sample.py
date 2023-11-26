import argparse
import runpy
from methods.utils import get_root_dir


parser = argparse.ArgumentParser()
parser.add_argument("--method", default='unet_gan', type=str, help="name of a method.")
opt = parser.parse_args()


if __name__ == '__main__':
    if opt.method == 'unet_gan':
        runpy.run_path(path_name=get_root_dir().joinpath('methods', 'unet_gan', 'sample.py'))
    else:
        raise ValueError
