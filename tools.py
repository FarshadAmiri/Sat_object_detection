from math import radians, cos, sin, asin, sqrt

def distance(lt1, lt2, lg1, lg2):
     
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lg1 = radians(lg1)
    lg2 = radians(lg2)
    lt1 = radians(lt1)
    lt2 = radians(lt2)
      
    # Haversine formula 
    dlg = lg2 - lg1 
    dlt = lt2 - lt1
    a = sin(dlt / 2)**2 + cos(lt1) * cos(lt2) * sin(dlg / 2)**2
 
    c = 2 * asin(sqrt(a)) 
    
    # Radius of earth in meters
    r = 6371000

    # calculate the distance in meters
    distance = c * r
    
    return distance
     