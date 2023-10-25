from math import radians, cos, sin, asin, sqrt
import math
import jdatetime
import datetime
from PIL import Image
import numpy as np

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
     
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
      
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000    # Radius of earth in meters

    distance = c * r     #distance in meters
    
    return distance


def bbox_geometry_calculator(bbox):
    #get coords of all four points of the bbox, width, height and area of the bbox
    lon1, lat1, lon2, lat2 = bbox
    
    # point_sw = (lon1, lat1)
    # point_ne =(lon2, lat2)
    # point_nw = (lon1, lat2)
    # point_se = (lon2, lat1)
    
    width =  haversine_distance(lon1, lat1, lon2, lat1)
    height = haversine_distance(lon1, lat1, lon1, lat2)
    area = width * height
    
    return width, height, area



def shamsi_date_time():
    # Get the current Gregorian date and time
    today = jdatetime.date.today()
    day, month, year = today.day, today.month, today.year
    clock = datetime.datetime.now().strftime("%H;%M")

    # Format the Jalali date and time
    formated_datetime = f"{year}_{month}_{day}-{clock}"

    return formated_datetime


def is_image(var):
    return (type(var) == np.ndarray) or (isinstance(var, Image.Image))
    

def verify_coords(coords, inference_mode):
    try:
        lon1, lat1, lon2, lat2 = coords
    except:
        if inference_mode in ["images_dict", "directory"]:
            raise ValueError(f"""bbox_coord_wgs84 should be a python dictionary containing keys equal to the images name and values equals to wgs84 coordinations in a list which is as follows:\n[West Longitude , South Latitude , East Longitude , North Latitude]""")
        else:
            raise ValueError(f"""bbox_coord_wgs84 can be a list, a tuple or a dictionary a key (anything) and value equals to wgs84 coordinations in a list or tuple rtpe which is as follows:\n[West Longitude , South Latitude , East Longitude , North Latitude]""")
    if (lon1 > lon2) or (lat1 > lat2):
        raise ValueError("""bbox_coord_wgs84 is supposed to be in the following format:\n[left, bottom, right, top]\nor in other words:\n[min Longitude , min Latitude , max Longitude , max Latitude]\nor in other words:\n[West Longitude , South Latitude , East Longitude , North Latitude]""")
    if any([(lon1 > 180), (lon2 > 180),
            (lon1 < -180), (lon2 < -180),
            (lat1 > 90), (lat2 > 90),
            (lat1 < -90), (lat2 < -90)]):
        raise ValueError("""Wrong coordinations! Latitude is between -90 and 90 and Longitude is between -180 and 180. Also, the following format is required:\n[left, bottom, right, top]\nor in other words:\n[min Longitude , min Latitude , max Longitude , max Latitude]\nor in other words:\n[West Longitude , South Latitude , East Longitude , North Latitude]""")
    coords_verified = True
    return coords_verified, lon1, lat1, lon2, lat2


def calculate_scale_down_factor(area, model_input_dim=768, a=0.2 , b=0.75, threshold=1.5):
    average_dim =  math.sqrt(area)
    dim_ratio = average_dim / model_input_dim
    if dim_ratio > threshold:
        scale_factor = (a * dim_ratio) + b
    else:
        scale_factor = 1
    scale_factor = max(scale_factor, 1)
    return scale_factor


def bbox_divide(bbox, lon_step=0.05, lat_step=0.05):
    lon_step=0.05
    lat_step=0.05

    m = str(lon_step)[::-1].find('.')
    lon1_ref, lat1_ref, lon2_ref, lat2_ref = bbox
    h_bbox = lat2_ref - lat1_ref
    w_bbox = lon2_ref - lon1_ref
    lon_no_steps = int(w_bbox//lon_step) + 1
    lat_no_steps = int(h_bbox//lat_step) + 1
    bboxes = []
    for h_partition in range(lat_no_steps):
        lat1 = lat1_ref + h_partition * lat_step
        lat1 = round(lat1, m)
        lat2 = lat1 + lat_step
        lat2 = round(lat2, m)
        bboxes_row = []
        for w_partition in range(lon_no_steps):
            lon1 = lon1_ref + w_partition * lon_step
            lon1 = round(lon1, m)
            lon2 = lon1 + lon_step
            lon2 = round(lon2, m)
            bboxes_row.append([lon1, lat1, lon2, lat2])
        bboxes.append(bboxes_row)
    return bboxes