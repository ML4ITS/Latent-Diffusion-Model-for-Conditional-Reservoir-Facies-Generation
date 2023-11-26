# import sys
import argparse
import runpy
from methods.utils import get_root_dir


# def parse_args_any(args):
#     pos = []
#     named = {}
#     key = None
#     for arg in args:
#         if key:
#             if arg.startswith('--'):
#                 named[key] = True
#                 key = arg[2:]
#             else:
#                 named[key] = arg
#                 key = None
#         elif arg.startswith('--'):
#             key = arg[2:]
#         else:
#             pos.append(arg)
#     if key:
#         named[key] = True
#     return (pos, named)


parser = argparse.ArgumentParser()
parser.add_argument("--method", default='unet_gan', type=str, help="name of a method.")
args = parser.parse_args()


if __name__ == '__main__':

    if args.method == 'unet_gan':
        runpy.run_path(path_name=str(get_root_dir().joinpath('methods', 'unet_gan', 'unet_gan.py')))
    elif args.method == 'ldm_stage1':
        runpy.run_path(path_name=str(get_root_dir().joinpath('methods', 'ldm', 'stage1.py')))
    elif args.method == 'ldm_stage2':
        runpy.run_path(path_name=str(get_root_dir().joinpath('methods', 'ldm', 'stage2.py')))
    elif args.method == 'ldm':
        runpy.run_path(path_name=str(get_root_dir().joinpath('methods', 'ldm', 'stage1.py')))
        runpy.run_path(path_name=str(get_root_dir().joinpath('methods', 'ldm', 'stage2.py')))
    else:
        raise ValueError
