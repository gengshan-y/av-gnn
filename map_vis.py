from moviepy.editor import ImageSequenceClip
from matplotlib import pyplot as plt
import numpy as np
import pdb

ld = []
for i in range(38):
    cam0 = plt.imread('/data/gengshay/carla-data/ped-50-veh-20-wea-1-pos-7-3-rep0/cam0/%05d.png'%i)
    depth0 = plt.imread('/data/gengshay/carla-data/ped-50-veh-20-wea-1-pos-7-3-rep0/depth0/%05d.png'%i)
    seg = plt.imread('/data/gengshay/carla-data/ped-50-veh-20-wea-1-pos-7-3-rep0/seg/%05d.png'%i)
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot('311')
    ax.axis('off')
    plt.imshow(cam0)

    ax = fig.add_subplot('312')
    ax.axis('off')
    plt.imshow(np.log(depth0),cmap='gnuplot2')

    ax = fig.add_subplot('313')
    ax.axis('off')
    plt.imshow(seg,cmap='tab20b')

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    

    fig, (a0, a1) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 1.5]}, figsize=(24,10))
    a0.axis('off')
    a0.imshow(data)

    mapi = plt.imread('/home/gengshay/code/av-gnn/data/ped-50-veh-20-wea-1-pos-7-3-rep0/map-%05d.png'%i)
    a1.axis('off')
    a1.imshow(mapi)

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    ld.append(data)

clip = ImageSequenceClip(ld,fps=5)
clip.write_videofile('vis.mp4',fps=5)
