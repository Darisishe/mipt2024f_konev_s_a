import numpy as np
import os
import argparse
import cv2
from src.replacement import replacement
from src.parseVIA import ViaImgMetadataParser


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    """
    Main function to run replacement function over dataset and synthetic barcodes

    Command-line arguments:
    - -d, --data: Optional. The path to the directory with data (data/ by default)
    - -s, --synthetic: Optional. The path to the directory with synthetic barcode for replacement (synthetic/ by default)
    - -o, --output: Optional. The to the directory where to save results (output/ by default)
    - --no-autocolor: Optional. Tells program to use NO_AUTOCOLOR mode for replacement (by default AUTOCOLOR is used)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", type=str, help="Data directory path", required=False
    )
    parser.add_argument(
        "-s",
        "--synthetic",
        type=str,
        help="Synthetic barcodes directory path",
        required=False,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Directory path for results", required=False
    )
    parser.add_argument(
        "--no-autocolor",
        action="store_true",
        help="Don't try to adjust color schemes of barcodes",
        required=False,
    )
    args = parser.parse_args()

    data_dir = "data/"
    synthetic_dir = "synthetic/"
    output_dir = "output/"
    replacement_mode = replacement.Mode.AUTOCOLOR

    if args.data:
        data_dir = args.data

    if args.synthetic:
        synthetic_dir = args.synthetic

    if args.output:
        output_dir = args.output

    if args.no_autocolor:
        replacement_mode = replacement.Mode.NO_AUTOCOLOR

    synthetic = dict()
    for barcode_type in os.listdir(synthetic_dir):
        type_dir = os.path.join(synthetic_dir, barcode_type)
        synthetic[barcode_type] = ViaImgMetadataParser(
            os.path.join(type_dir, "annotations.json"), type_dir
        )

    data = ViaImgMetadataParser(
        os.path.join(data_dir, "barcode_annotations.json"), data_dir
    )

    for base_image, base_polygon, barcode_type, region_id in data:
        for new_image, new_polygon, _, _ in synthetic[barcode_type]:
            seamless_result = replacement.seamless_replace(
                cv2.imread(base_image),
                base_polygon,
                cv2.imread(new_image),
                new_polygon,
                replacement_mode,
            )

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    filename(new_image) + "_on_" + filename(base_image) + "-" + region_id + ".jpg",
                ),
                seamless_result,
            )


if __name__ == "__main__":
    main()
