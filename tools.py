from math import radians, cos, sin, asin, sqrt
import math
import jdatetime
import datetime
from PIL import Image
import numpy as np
import requests
import os
from io import BytesIO

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
    m = str(lon_step)[::-1].find('.')
    lon1_ref, lat1_ref, lon2_ref, lat2_ref = bbox
    h_bbox = lat2_ref - lat1_ref
    w_bbox = lon2_ref - lon1_ref
    lon_no_steps = int(w_bbox//lon_step) + 1
    lat_no_steps = int(h_bbox//lat_step) + 1
    bboxes = []
    for h_partition in range(lat_no_steps):
        lat1 = lat1_ref + h_partition * lat_step
        lat2 = lat1 + lat_step
        lat1, lat2 = map(lambda x: round(x, m), [lat1, lat2])
        bboxes_row = []
        for w_partition in range(lon_no_steps):
            lon1 = lon1_ref + w_partition * lon_step
            lon2 = lon1 + lon_step
            lon1, lon2 = map(lambda x: round(x, m), [lon1, lon2])
            bboxes_row.append([lon1, lat1, lon2, lat2])
        bboxes.append(bboxes_row)
    return bboxes


def xyz2bbox(x,y,z):   
    lonmin = x / math.pow(2.0, z) * 360.0 - 180
    lonmax = (x+1) / math.pow(2.0, z) * 360.0 - 180
    n1 = math.pi - (2.0 * math.pi * y) / math.pow(2.0, z)
    latmax = math.atan(math.sinh(n1)) * 180 / math.pi
    n2 = math.pi - (2.0 * math.pi * (y+1)) / math.pow(2.0, z)
    latmin = math.atan(math.sinh(n2)) * 180 / math.pi
    coords = (lonmin, latmin, lonmax, latmax)
    return coords


def sentinel_url_xyz(x, y, z, start=None, end=None, n_days_before_date=None, date=None, save_img=False,
                     output_dir="", img_name=None, output_img=True):
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

    lonmin, latmin, lonmax, latmax = xyz2bbox(x,y,z)
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
    if output_img:
        return img
    return


def sentinel_url_longlat(lonmin, latmin, lonmax, latmax, start=None, end=None, n_days_before_date=None, date=None,
                         save_img=False, output_dir="", img_name=None, output_img=True, output_url=False):
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
    if output_img and output_url:
        return img, url
    elif (output_img == True) and (output_url == False):
        return img
    elif (output_img == False) and (output_url == True):
        return url
    return