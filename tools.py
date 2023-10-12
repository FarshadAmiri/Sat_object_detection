import math
import jdatetime
import datetime
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
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