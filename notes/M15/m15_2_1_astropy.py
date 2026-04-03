# Astropy Example
# author: allee updated by sdm

from urllib.parse import urlencode          # encode for internet access
from urllib.request import urlretrieve      # retrieve file from the web

# Astropy package contents required...
from astropy import units as u              # so we can specify arc minutes
from astropy.coordinates import SkyCoord    # get coordinates of objects

# Maybe this works in Spyder?
# from an image processing package
#from IPython.display import Image           # display the image

# I will use this instead...
from PIL import Image                        # display the image

# Set up matplotlib and use a nicer set of plot parameters
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
plt.style.use(astropy_mpl_style)

# List of tuples consisting of (Object name, arcmins to use)
object_list = [ ('M42',     300), \
                ('IC 5146',  70), \
                ('M51',      64), \
                ('M64',      30), \
                ('M92',      50), \
                ('M101',     40), \
                ('M104',     40), \
                ('NGC 2859', 15), \
                ('NGC 3718', 20), \
                ('NGC 4656', 25)  ]

for objects in object_list:
    telescope_center = SkyCoord.from_name(objects[0])      # point the scope
    arcmins = objects[1]                                   # set field of view

    #print(telescope_center.ra, telescope_center.dec)      # print the above
    #print(telescope_center.ra.hour, telescope_center.dec)

    # tell the SDSS service how big of a cutout we want
    im_size = arcmins*u.arcmin                       # get an NxN arcmin square
    im_pixels = 4096

    # set up the query
    cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
    query_string = urlencode(dict(ra=telescope_center.ra.deg,
                                  dec=telescope_center.dec.deg,
                                  width=im_pixels, height=im_pixels,
                                  scale=im_size.to(u.arcsec).value/im_pixels))
    url = cutoutbaseurl + '?' + query_string

    # this downloads the image to your disk into the image_name file
    image_name = objects[0]+'.jpg'
    urlretrieve(url, image_name)

    # and now we can display it
    space = Image.open(image_name)
    space.show()
