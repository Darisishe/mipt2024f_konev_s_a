import os
import json
import numpy as np


class ViaImgMetadataParser:
    def __init__(self, json_file, dir):
        self.json_data = json.load(open(json_file, "r"))
        self.img_metadata = self.json_data["_via_img_metadata"]
        self.dir = dir
        self.regions = self.get_regions()

    def get_regions(self):
        all_regions = []
        for _, img_data in self.img_metadata.items():
            regions = img_data["regions"]
            region_id = 0
            for region in regions:
                if region["shape_attributes"]["name"] == "polygon":
                    all_points_x = region["shape_attributes"]["all_points_x"]
                    all_points_y = region["shape_attributes"]["all_points_y"]
                    polygon_points = np.float32(list(zip(all_points_x, all_points_y)))
                    all_regions.append(
                        {
                            "polygon": polygon_points,
                            "image": os.path.join(self.dir, img_data["filename"]),
                            "type": region["region_attributes"]["type"],
                            "region_id": "reg" + str(region_id)
                        }
                    )
                region_id += 1
        return all_regions

    def __iter__(self):
        return ViaRegionsIterator(self.regions)


class ViaRegionsIterator:
    def __init__(self, regions):
        self.regions = regions
        self.ind = 0

    def __next__(self):
        if self.ind < len(self.regions):
            image_file = self.regions[self.ind]["image"]
            polygon_points = self.regions[self.ind]["polygon"]
            barcode_type = self.regions[self.ind]["type"]
            region_id = self.regions[self.ind]["region_id"]

            self.ind += 1
            return image_file, polygon_points, barcode_type, region_id
        else:
            raise StopIteration
