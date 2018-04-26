from moviepy.editor import ImageSequenceClip
from matplotlib import pyplot as plt
import numpy as np

ld = []
for i in range(38):
    im = plt.imread('/home/gengshay/code/av-gnn/data/ped-50-veh-20-wea-1-pos-7-3-rep0/map-%05d.png'%i)*255
    im = im.astype(np.uint8)
    print(np.max(im))
    ld.append(im)

clip = ImageSequenceClip(ld,fps=5)
clip.write_videofile('map_vis.mp4',fps=5)
