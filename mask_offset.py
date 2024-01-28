import cv2
import numpy as np
import argparse
import os
from util.misc import refresh_dir

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--raw_mask_dir",type=str,default="dataset/taxim/marker/")
    io_parser.add_argument("--output_mask_dir",type=str,default="dataset/taxim/marker_offset/")
    para_parser = parser.add_argument_group()
    para_parser.add_argument("--offset",type=int,nargs=2,default=[20,20])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    raw_mask_files = sorted(os.listdir(args.raw_mask_dir))
    refresh_dir(args.output_mask_dir)
    for raw_file in raw_mask_files:
        raw_mask = cv2.imread(os.path.join(args.raw_mask_dir, raw_file), cv2.IMREAD_GRAYSCALE)
        H, W = raw_mask.shape
        mask_yx = np.nonzero(raw_mask)  # y,x
        new_mask_yx = [mask_yx[0]+args.offset[0], mask_yx[1]+args.offset[1]]
        rev = (new_mask_yx[0] >= 0) * (new_mask_yx[0] < H) * (new_mask_yx[1] >= 0) * (new_mask_yx[1] < W)
        new_mask_yx = [new_mask_yx[0][rev], new_mask_yx[1][rev]]
        new_mask = np.zeros_like(raw_mask, dtype=np.uint8)
        new_mask[new_mask_yx[0], new_mask_yx[1]] = 255
        new_mask[raw_mask > 0] = 0
        cv2.imwrite(os.path.join(args.output_mask_dir, raw_file), new_mask)
    