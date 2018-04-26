from PIL import Image
import cv2
import json
from google.protobuf.json_format import MessageToJson
import os
import math
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0,'/home/gengshay/code/carla-agent//PythonClient')
import pdb

from carla.client import make_carla_client, VehicleControl
import carla.sensor
from carla.settings import CarlaSettings
from carla.benchmarks.experiment import Experiment
from carla.autopilot.autopilot import Autopilot
from carla.autopilot.pilotconfiguration import ConfigAutopilot
from carla.planner.planner import Planner
from carla.planner.map import CarlaMap
from matplotlib.patches import Circle

autopilot = Autopilot(ConfigAutopilot('Town01'))
planner = Planner('Town01')
town_map = plt.imread("../carla-agent/PythonClient/carla/planner/Town01.png")
carla_map = CarlaMap('Town01', 16.53, 50)

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if os.path.isdir(path):
            pass
        else: raise

def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)

def poses_town01():
    """
    Each matrix is a new task. We have all the four tasks

    """

    def _poses_straight():
       return [[36, 40], [39, 35], [110, 114], [7, 3], [0, 4],
                [68, 50], [61, 59], [47, 64], [147, 90], [33, 87],
                [26, 19], [80, 76], [45, 49], [55, 44], [29, 107],
                [95, 104], [84, 34], [53, 67], [22, 17], [91, 148],
                [20, 107], [78, 70], [95, 102], [68, 44], [45, 69]]

    def _poses_one_curve():
        return [[138, 17], [47, 16], [26, 9], [42, 49], [140, 124],
                [85, 98], [65, 133], [137, 51], [76, 66], [46, 39],
                [40, 60], [0, 29], [4, 129], [121, 140], [2, 129],
                [78, 44], [68, 85], [41, 102], [95, 70], [68, 129],
                [84, 69], [47, 79], [110, 15], [130, 17], [0, 17]]

    return [_poses_straight(),
            _poses_one_curve()]

# town01
weathers = [1, 3, 6, 8, 4, 14]
poses_tasks = poses_town01()
vehicles_tasks = [20, 20]
pedestrians_tasks = [50, 50]

cam0 = carla.sensor.Camera('im0', PostProcessing='SceneFinal')
cam0.set_image_size(1152, 384)
cam0.set_position(x=0., y=-6, z=165)

cam1 = carla.sensor.Camera('im1', PostProcessing='SceneFinal')
cam1.set_image_size(1152, 384)
cam1.set_position(x=0., y=48, z=165)

depth_cam0 = carla.sensor.Camera('depth0', PostProcessing='Depth')
depth_cam0.set_image_size(1152, 384)
depth_cam0.set_position(x=0., y=-6, z=165)

depth_cam1 = carla.sensor.Camera('depth1', PostProcessing='Depth')
depth_cam1.set_image_size(1152, 384)
depth_cam1.set_position(x=0., y=48, z=165)

seg_cam = carla.sensor.Camera('seg', PostProcessing='SemanticSegmentation')
seg_cam.set_image_size(1152, 384)
seg_cam.set_position(x=0., y=-6, z=165)
experiments_vector = []

for weather in weathers:
    for iteration in range(len(poses_tasks)):
        poses = poses_tasks[iteration]
        vehicles = vehicles_tasks[iteration]
        pedestrians = pedestrians_tasks[iteration]

        conditions = CarlaSettings()
        conditions.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=vehicles,
            NumberOfPedestrians=pedestrians,
            WeatherId=weather,
            SeedVehicles=123456789,
            SeedPedestrians=123456789
        )
        
        # Add all the cameras that were set for this experiments
        conditions.add_sensor(cam0)
        conditions.add_sensor(cam1)
        conditions.add_sensor(depth_cam0)
        conditions.add_sensor(depth_cam1)
        conditions.add_sensor(seg_cam)
        #conditions.add_sensor(lidar)

        experiment = Experiment()
        experiment.set(
            Conditions=conditions,
            Poses=poses,
            Id=iteration,
            Repetitions=1
        )
        experiments_vector.append(experiment)


fig,ax = plt.subplots()

with make_carla_client('localhost', 2000) as client:
    for experiment in experiments_vector:
        positions = client.load_settings(
            experiment.conditions).player_start_spots

        for pose in experiment.poses:
            for rep in range(experiment.repetitions):
                seq_name ='ped-%d-veh-%d-wea-%d-pos-%d-%d-rep%d'%\
                             (experiment.conditions.NumberOfPedestrians,
                              experiment.conditions.NumberOfVehicles,
                              experiment.conditions.WeatherId,
                              pose[0],pose[1],rep)
                mkdir_p('./data/%s'%seq_name)
                mkdir_p('./data/%s/cam0'%seq_name)
                mkdir_p('./data/%s/cam1'%seq_name)
                mkdir_p('./data/%s/depth0'%seq_name)
                mkdir_p('./data/%s/depth1'%seq_name)
                mkdir_p('./data/%s/seg'%seq_name)
                mkdir_p('./data/%s/measures'%seq_name)
                mkdir_p('./data/%s/waypoints'%seq_name)

                start_point = pose[0]
                end_point = pose[1]
                target = positions[end_point]
                print('%d to %d'%(start_point,end_point))

                ax.clear()
                ax.imshow(town_map)
                pixel = carla_map.convert_to_pixel([positions[start_point].location.x,
                                    positions[start_point].location.y,
                                    positions[start_point].location.z])
                ax.add_patch(Circle((pixel[0], pixel[1]), 12, color='r', label='A point'))
                pixel = carla_map.convert_to_pixel([positions[end_point].location.x,
                                                    positions[end_point].location.y,
                                                    positions[end_point].location.z])
                ax.add_patch(Circle((pixel[0], pixel[1]), 12, color='b', label='A point'))

                client.start_episode(start_point)
               
                for i in range(50):
                    measurements, sensor_data = client.read_data()
                    client.send_control(measurements.player_measurements.autopilot_control)

                fr_max = sldist([measurements.player_measurements.transform.location.x,
                                   measurements.player_measurements.transform.location.y],
                                      [target.location.x, target.location.y]) / 50
                for i in range(int(fr_max)):
                    measurements, sensor_data = client.read_data()
                    curr_x = measurements.player_measurements.transform.location.x
                    curr_y = measurements.player_measurements.transform.location.y
                    distance = sldist([curr_x, curr_y],
                                      [target.location.x, target.location.y])
                    if distance < 200:break

                    control,wp_info = autopilot.run_step(measurements, sensor_data,target)
                    direction = planner.get_next_command(
                        (curr_x,curr_y,22),
                        (measurements.player_measurements.transform.orientation.x,
                         measurements.player_measurements.transform.orientation.y,
                         measurements.player_measurements.transform.orientation.z),
                        (target.location.x, target.location.y, 22),
                        (target.orientation.x, target.orientation.y, -0.001))

                    if i%10==0:
                        pixel = carla_map.convert_to_pixel([curr_x,curr_y,22])
                        ax.add_patch(Circle((pixel[0], pixel[1]), 12, color='g', label='A point'))
                        # waypoints
                        for loc in wp_info['waypoints'][:10]:
                            pixel = carla_map.convert_to_pixel([loc[0],loc[1],22])
                            ax.add_patch(Circle((pixel[0], pixel[1]), 12, color='g', label='A point'))
                        fig.canvas.draw()
                        plt.pause(.0001)

                    client.send_control(control)
               #     # record
               #     im0=Image.fromarray(sensor_data['im0'].data, mode='RGB')
               #     im1=Image.fromarray(sensor_data['im1'].data, mode='RGB')
               #     depth0 = (sensor_data['depth0'].data*256*256).astype(np.uint16)
               #     depth1 = (sensor_data['depth1'].data*256*256).astype(np.uint16)
               #     seg = Image.fromarray(sensor_data['seg'].data)

               #     im0.save('data/%s/cam0/%05d.png'%(seq_name,i))
               #     im1.save('data/%s/cam1/%05d.png'%(seq_name,i))
               #     cv2.imwrite('data/%s/depth0/%05d.png'%(seq_name,i),depth0)
               #     cv2.imwrite('data/%s/depth1/%05d.png'%(seq_name,i),depth1)
               #     seg.save('data/%s/seg/%05d.png'%(seq_name,i))

               #     with open("data/%s/waypoints/%05d.json"%(seq_name,i), 'w') as jsfile:
               #         json.dump(wp_info,jsfile)
               #     with open("data/%s/measures/%05d.json"%(seq_name,i), 'w') as jsfile:
               #         actual_json_text = MessageToJson(measurements)
               #         actual_json_text = actual_json_text[:-1] + ',"command": %d}'%int(direction)
               #         jsfile.write( actual_json_text )
               #     
               # plt.savefig('data/%s-map.png'%seq_name)
