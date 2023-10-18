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