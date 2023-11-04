import logging
from sentinelhub import MimeType, CRS, BBox, DataCollection, bbox_to_dimensions, SHConfig, SentinelHubRequest
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from tools import bbox_geometry_calculator, bbox_divide, xyz2bbox, bbox2xyz
import datetime, requests
from io import BytesIO
import numpy as np
from PIL import Image
import os
import math
import cv2
from tqdm import tqdm
 
#  Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# Your client credentials
client_id = 'e0b127dc-fbb5-4151-8946-96d8728003c7'
client_secret = '4Xy2[Bc#?&L!xbha(XB6*%Xbry,BU-cZA0Dd9n,W'

# Create a session
# client = BackendApplicationClient(client_id=client_id)
# oauth = OAuth2Session(client=client)

config = SHConfig(instance_id='',
         sh_client_id = client_id,
         sh_client_secret = client_secret,
         sh_base_url='https://services.sentinel-hub.com',
         sh_token_url='https://services.sentinel-hub.com/oauth/token',)

# Get token for the session
# token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',
#                           client_secret=client_secret)

# # All requests using this session will have an access token automatically added
# resp = oauth.get("https://services.sentinel-hub.com/oauth/tokeninfo")
# print(resp.content)

# save sentinel config
# config.instance_id = "my-instance-id"
# config.save("my-profile")


# define a function to download a single image from sentinel hub
# ---------------------------------------------------------------------------------
def sentinel_single_image(bbox, timeline, config, data_collection=DataCollection.SENTINEL2_L2A, maxcc=0.8, mosaicking_order = 'mostRecent', resolution=10,
                       img_size=None, return_numpy=False, verbose=False, save_image=False, save_dir="sentinel-hub", evalscript=None):
    
    if save_image and save_dir is None:
        raise ValueError("save_dir must be specified when save_image is True")
    
    # Verbose printout
    if verbose:
        w, h, area = bbox_geometry_calculator(bbox)
        print(f"Territory dimensions: {w:,.0f}m x {h:,.0f}m | Area: {(area * 1e-6):,.0f} km^2")
    
    if evalscript is None:
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
                    mosaicking: "TILE"
                };
            }

            function evaluatePixel(sample) {
                return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];

            }
        """
        evalscript = evalscript_true_color

    # Set resolution and region bb/size.
    region_bbox = BBox(bbox = bbox, crs = CRS.WGS84)
    if img_size == None:
        region_size = bbox_to_dimensions(region_bbox, resolution = resolution)
    else:
        region_size = img_size
    print(f'Requesting images with {resolution}m resolution and region size of {region_size} pixels') if verbose else None
    
    # Build the request.
    request_true_color = SentinelHubRequest(
        evalscript = evalscript,
        input_data = [
            SentinelHubRequest.input_data(
                data_collection = data_collection,
                time_interval = timeline,
                mosaicking_order = mosaicking_order,
                maxcc = maxcc,
            )
        ], 
        responses = [
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox = region_bbox,
        # resolution = 10,
        size = region_size,
        config = config,
    )

    # By construction, only one image at time is returned.
    image_numpy = request_true_color.get_data(save_data=False)[0]
    if save_image or (return_numpy == False):
        image = Image.fromarray(image_numpy.astype('uint8')).convert('RGB')
    
    if save_image:
        image.save(save_dir)
    
    return image_numpy if return_numpy else image



# define a function to download a high-resolution image of a vast area from sentinel hub
# ---------------------------------------------------------------------------------------
def sentinel_territory(bbox_coords, timeline, config, data_collection=DataCollection.SENTINEL2_L2A, mosaicking_order = 'mostRecent', maxcc=0.8, evalscript=None,
                      resolution=5, img_size=(2500,2500), lon_lat_step=(0.05, 0.05), in_memory=True, temp_dir=r"sentinel-tmp", save_concat_image=False,
                      concat_image_dir=r"sentinel-concat", concat_image_name="default", delete_temp=False, ignore_inference_size_error=False):
    # Verbose printout
    w, h, area = bbox_geometry_calculator(bbox_coords)
    if area > 1500e6:
        raise ValueError(f"Area of the territory is too big: {(area * 1e-6):,.0f} km^2! Territory's area shouldn't exceed 1500 km^2.")
    lon1_ref, lat1_ref, lon2_ref, lat2_ref = bbox_coords
    bboxes = bbox_divide(bbox_coords, lon_step=lon_lat_step[0], lat_step=lon_lat_step[1])
    total_no_bboxes = int (len(bboxes) * len(bboxes[0]))
    image_width =  int((lon2_ref - lon1_ref) * img_size[0] / lon_lat_step[0])
    image_height = int((lat2_ref - lat1_ref) * img_size[1] / lon_lat_step[1])
    image_pixels = image_width * image_height
    print(f"Territory dimensions: {w:,.0f}m x {h:,.0f}m | Area: {(area * 1e-6):,.0f} km^2")
    print(f"Concatenated image will be in size of {image_width} x {image_height} p (if all of its area is supported by sentinel-hub)")
    print("There are {} bboxes to download".format(total_no_bboxes))
    if ignore_inference_size_error == False:
        if image_pixels > 178e6:
            raise ValueError(f"Image size is too big for inference: {image_pixels:,.0f} pixels! Image size shouldn't exceed {178956970:,.0f} pixels because of inference limitations.\nTry reducing each dimension by {(math.sqrt(image_pixels/178000000)-1)*100:.1f}%.")
    
    # Create output folder
    if in_memory == False:
        if temp_dir == None:
            raise ValueError("temp_dir must be specified if in_memory is False")
        tmp_subfolder = f"{timeline[0]}_{timeline[1]}-{bbox_coords}-res{resolution}-{img_size}-{mosaicking_order}-{data_collection.catalog_id}-maxcc{maxcc}"
        output_dir = os.path.join(temp_dir, tmp_subfolder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Download images from sentinel-hub
    images = []
    print()
    pbar = tqdm(total=total_no_bboxes, desc="dl from sentinel-hub")
    image_bbox_map = ""
    for i, bbox_row in enumerate(bboxes):
        images_row = []
        for j, bbox in enumerate(bbox_row):
            img = sentinel_single_image(bbox, timeline, config, data_collection=data_collection, maxcc=maxcc, mosaicking_order = mosaicking_order,
                                        evalscript=evalscript ,resolution=resolution, img_size=img_size, return_numpy=True, verbose=False)
            if in_memory:
                images_row.append(img)
            else:
                tmp_img = Image.fromarray(img.astype('uint8')).convert('RGB')
                tmp_img.save(f"{output_dir}" + f"/{i}_{j}" + ".jpg")
                image_bbox_map += f"{i}_{j} = {bbox}\n"
            pbar.update(1)
            pbar.set_description_str(f"dl from sentinel-hub | bbox: {bbox}")
        images.append(images_row)
        
    if in_memory == False and (delete_temp==False):
        with open(os.path.join(output_dir, "image_bbox_map.txt"), "w") as f:
            f.write(image_bbox_map)
        
    print("All images are downloaded into {} - concatenating images is in progress...".format("memory" if in_memory else "disk"))
    if in_memory:     
        images_horizontally = []
        for images_row in images:
            im_v = cv2.hconcat(images_row)
            images_horizontally.append(im_v)
            
        concat_image = cv2.vconcat(images_horizontally[::-1])
        concat_image = Image.fromarray(concat_image.astype('uint8')).convert('RGB')
    else:
        n_rows, n_cols = len(bboxes), len(bboxes[0])
        images_horizontally = []
        for row in range(n_rows):
            row_images_path = [os.path.join(output_dir, f"{row}_{col}.jpg") for col in range(n_cols)]
            row_images = [np.array(Image.open(img_path)) for img_path in row_images_path]
            images_horizontally.append(cv2.hconcat(row_images))

        concat_image = cv2.vconcat(images_horizontally[::-1])
        concat_image = Image.fromarray(concat_image.astype('uint8')).convert('RGB')
    print("Done!")

    if delete_temp:
        for filename in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, filename)) 
        os.rmdir(output_dir)
        print("Temp")
    
    if save_concat_image:
        if concat_image_dir is None:
            raise ValueError("concat_image_dir must be specified if save_concat_image is True")
        elif os.path.exists(concat_image_dir) == False:
            os.mkdir(concat_image_dir)
        concat_image_name = f"{timeline[0]}_{timeline[1]}-{bbox_coords}-res{resolution}-{img_size}-{mosaicking_order}-{data_collection.catalog_id}-maxcc{maxcc}.jpg" if concat_image_name == "default" else concat_image_name
        concat_image.save(os.path.join(concat_image_dir, concat_image_name))
        print("concatenated image saved in {}".format(concat_image_dir))
    return concat_image



def sentinel_query(coords, start=None, end=None, n_days_before_date=None, date=None, save_img=False,
                     output_dir="", img_name=None, output_img=False, output_url=False):
    if n_days_before_date != None:
        if date == None:
            end = datetime.datetime.now()
            start = end - datetime.timedelta(days=n_days_before_date)
        elif type(date) == datetime.datetime:
            end = date
            start = end - datetime.timedelta(days=n_days_before_date)
        else:
            end_year, end_month, end_day = end.split('/')
            end = datetime.date(end_year, end_month, end_day)
            start = end - datetime.timedelta(days=n_days_before_date)

    if type(start)!= datetime.datetime:
        start_year, start_month, start_day = start.split('/')
        start = datetime.date(start_year, start_month, start_day)
    if type(end) != datetime.datetime:
        end = date
        start = end - datetime.timedelta(days=n_days_before_date)

    start_formatted = datetime.datetime.strftime(start, "%Y-%m-%dT%H:%M:%SZ")
    end_formatted = datetime.datetime.strftime(end, "%Y-%m-%dT%H:%M:%SZ")

    if len(coords) == 3:
        lonmin, latmin, lonmax, latmax = xyz2bbox(coords)
    elif len(coords) == 4:
        lonmin, latmin, lonmax, latmax = coords
    else:
        raise ValueError("coords must be either 3 or 4 elements - (x,y,z) or (lonmin, latmin, lonmax, latmax), in form of list or tuple")
    print(lonmin, latmin, lonmax, latmax)
    url = fr"http://services.sentinel-hub.com/v1/wms/cd280189-7c51-45a6-ab05-f96a76067710?service=WMS&request=GetMap&layers=1_TRUE_COLOR&styles=&format=image%2Fpng&transparent=true&version=1.1.1&showlogo=false&additionalParams=%5Bobject%20Object%5D&name=Sentinel-2&height=256&width=256&errorTileUrl=%2Fimage-browser%2Fstatic%2Fmedia%2FbrokenImage.ca65e8ca.png&pane=activeLayer&maxcc=20&time={start_formatted}/{end_formatted}&srs=EPSG%3A4326&bbox={lonmin},{latmin},{lonmax},{latmax}"
    
    response = requests.get(url)
    if output_img:
        img = Image.open(BytesIO(response.content))
    # Save the image
    if save_img:
        if img_name == None:
            img_name = f"[{lonmin:.4f},{latmin:.4f},{lonmax:.4f},{latmax:.4f}]-{start_formatted.split('T')[0]}_{end_formatted.split('T')[0]}).jpg"
        elif img_name.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")) == False:
            img_name = img_name + ".jpg"
        img_path = os.path.join(output_dir, img_name)
        with open(img_path, 'wb') as f:
            f.write(response.content)
    if output_img and output_url==False:
        return img
    elif output_img and output_url:
        return img, url
    return

