import math
import numpy as np

# Define the points
#parent_x = [94, 90.84666304551956, 106.97411816978266, 125.08310470604326, 145.0369884065586, 162.92164220257663, 160.5212127404915, 140.52227345449435, 128.5137223042476, 109.44706819500902, 104.40165132531894, 92.62157907526067, 72.63524371012376, 55.22040830911175, 35.75191574779196, 29.292963283538317]
#parent_y = [132, 112.25015275878043, 100.42200027350701, 91.93291277074604, 90.57551626776383, 81.62346853573189, 61.76804261742045, 61.97402182753673, 45.9804387620724, 39.94200344498443, 59.29513674714113, 43.13252873409425, 42.39334088768601, 52.228151915644744, 47.64799936470976, 28.719662589797082]
# class TransformData:
def transform(x_y, image_shape):
    x_y = x_y


    width = image_shape[0]
    height = image_shape[1]
    scalar_factor_x = 10/width
    scalar_factor_y = 10/height
    parent_x = [(width/2 - u[1])*scalar_factor_x for u in x_y]
    parent_y = [(height/2 - u[0])*scalar_factor_y for u in x_y]

    xy_coordinates = [(w, h) for w, h in zip(parent_x, parent_y)]
    a = 0
    b = 1
    distance = 0
    for x in range(len(xy_coordinates)-1):
        covered = math.sqrt( ((xy_coordinates[a][0]-xy_coordinates[b][0])**2)+((xy_coordinates[a][1]-xy_coordinates[b][1])**2) )
        a+=1
        b+=1
        distance+=covered

    # calculating the path heading angle
    H = xy_coordinates[0][0]-xy_coordinates[1][0]
    V = xy_coordinates[0][1] - xy_coordinates[1][1]
    angle_deg = math.atan(V/H)*180/math.pi
    file_path = 'output.txt'
    array_of_tuples = np.array(xy_coordinates)

    # Specify the file path where you want to save the text file
    file_path = 'output.txt'

    # Save the NumPy array to a text file
    np.savetxt(file_path, array_of_tuples, fmt='%s')
    return [xy_coordinates, angle_deg, math.atan(V/H), distance]

