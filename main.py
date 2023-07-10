import os

import cv2
import argparse
from tqdm import tqdm

from tools import resize_image_with_pad, apply_midas


def main():
    parser = argparse.ArgumentParser(
        description="Batch converts images to depth map using MiDaS"
    )
    parser.add_argument("input_dir", help="input directory")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("--res", type=int, default=512, help="Resolution to process at")
    args = parser.parse_args()

    # make output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    img_ext = [".jpg", ".jpeg", ".png", ".webp"]

    image_paths = [img_path for img_path in os.listdir(args.input_dir) if
                   os.path.splitext(img_path)[1].lower() in img_ext]

    print(f"Found {len(image_paths)} images")
    for img_path in tqdm(image_paths):
        full_img_path = os.path.join(args.input_dir, img_path)

        img = cv2.imread(full_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # remove alpha channel if present
        if img.shape[2] == 4:
            img = img[:, :, :3]

        img, remove_pad = resize_image_with_pad(img, args.res)
        result = apply_midas(img)

        output = remove_pad(result)

        # make sure it is rgba
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

        output = output.astype('uint8')
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(args.output_dir, img_path), output)

    print("FIN")


if __name__ == "__main__":
    main()
