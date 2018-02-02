# let's try sublime text in python!


# okay, we're going to try another different polar conversion t see if this works
import numpy as np
from scipy.ndimage.interpolation import geometric_transform

def topolar(img, order=1):
    """
    Transform img to its polar coordinate representation.

    order: int, default 1
        Specify the spline interpolation order. 
        High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal 
    # from a corner to the mid-point of img.

    print "in to polar"
    max_radius = 0.5*np.linalg.norm( img.shape )

    def transform(coords):
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)

        # Then map it to the interval [0, max_radius].
        #radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]

        i = 0.5*img.shape[0] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[1]
        return i,j

    print "beginning sklearn geometric transform"
    polar = geometric_transform(img, transform, order=order)
    print "finished geometric transform"

    rads = max_radius * np.linspace(0,1,img.shape[0])
    angs = np.linspace(0, 2*np.pi, img.shape[1])

    return polar, (rads, angs)

from skimage.data import chelsea

print "stating end function"

import matplotlib.pyplot as plt

img = chelsea()[...,0] / 255.
pol, (rads,angs) = topolar(img)

print "finished to polar"

fig,ax = plt.subplots(2,1,figsize=(6,8))

ax[0].imshow(img, cmap=plt.cm.gray, interpolation='bicubic')

ax[1].imshow(pol, cmap=plt.cm.gray, interpolation='bicubic')

ax[1].set_ylabel("Radius in pixels")
ax[1].set_yticks(range(0, img.shape[0]+1, 50))
ax[1].set_yticklabels(rads[::50].round().astype(int))

ax[1].set_xlabel("Angle in degrees")
ax[1].set_xticks(range(0, img.shape[1]+1, 50))
ax[1].set_xticklabels((angs[::50]*180/3.14159).round().astype(int))

plt.show()
