import logging
from sentinelhub import MimeType, CRS, BBox, DataCollection, bbox_to_dimensions
from sentinelhub import SHConfig, SentinelHubRequest

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def sentinel_get_image(bbox, size, data_collection, config):
    """
    This function returns a Sentinel Hub request object for Sentinel-2 imagery.
    """
    return SentinelHubRequest(
        bbox=bbox,
        time=('2017-12-01', '2017-12-31'),
        data_collection=data_collection,
        bands=['B04', 'B03', 'B02'],
        maxcc=0.2,
        mosaicking_order='mostRecent',
        config=config




def get_sentinel_image(bbox, timeline, data_collection=DataCollection.SENTINEL2_L2A, maxcc=0.8, mosaicking_order = 'mostRecent', resolution=10,
                       img_size=None, save_images=False, data_folder="sentinel-hub"):
    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];

        }
    """
    # Set resolution and region bb/size.
    region_bbox = BBox(bbox = bbox, crs = CRS.WGS84)
    region_size = bbox_to_dimensions(region_bbox, resolution = resolution)
    output_img_size = region_size if img_size == None else img_size
    print(f'Requesting images with {resolution}m resolution and region size of {output_img_size} pixels')
    # Build the request.
    request_true_color = SentinelHubRequest(
        data_folder = data_folder,
        evalscript = evalscript_true_color,
        input_data = [
            SentinelHubRequest.input_data(
                data_collection = data_collection,
                time_interval = timeline,
                mosaicking_order = mosaicking_order,
                # maxcc = maxcc,
            )
        ], 
        responses = [
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox = region_bbox,
        # resolution = 10,
        size = output_img_size,
        config = config,
    )

    # By construction, only one image at time is returned.
    true_color_imgs = request_true_color.get_data(save_data=save_images)
    image = Image.fromarray(true_color_imgs[0].astype('uint8')).convert('RGB')
    return image



# bbox = [58.488808,23.630371,58.573265,23.699550]
# time_interval = ['2023-07-05', '2023-09-25']
# images = request_images(coords_wgs84=bbox, timeline=time_interval)


def sentinel_get_area(bbox, timeline, data_collection, config):
    pass