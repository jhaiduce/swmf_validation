from spacepy.pybats import ImfInput
from cdaweb import get_cdf
from swmf_validation.metrics import interp_timeseries
from datetime import datetime
import dateutil
import numpy as np
import sys
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

if len(sys.argv)!=4:
    print ''
    print 'Usage: build_imfinput.py start_date end_date imffile'
    print 'start_date and end_date can be in any format readable by dateutil.parser.parse'
    sys.exit()

dtstart=dateutil.parser.parse(sys.argv[1])
dtend=dateutil.parser.parse(sys.argv[2])

imfdata=ImfInput(sys.argv[3],load=False)

# Fetch OMNI data
omnidata=get_cdf('sp_phys','OMNI_HRO_1MIN',dtstart,dtend,('BX_GSE','BY_GSM','BZ_GSM','Vx','Vy','Vz','Pressure','T','proton_density'))

# Populate a list of keys for both datasets
omni_keys=['T','proton_density']
imfinput_keys=['temp','rho']
for ax in 'x','y','z':
    imfinput_keys.append('b'+ax)
    if ax=='x':
        omni_keys.append('B'+ax.upper()+'_GSE')
    else:
        omni_keys.append('B'+ax.upper()+'_GSM')

# Convert OMNI velocities to GSM coordinates
u_gse=np.ma.vstack([
    np.ma.masked_greater(
        omnidata[omni_key],omnidata[omni_key].attrs['VALIDMAX']
    ) for omni_key in ['Vx','Vy','Vz']]).T
cvals=Coords(u_gse,'GSE','car',ticks=Ticktock(omnidata['Epoch']))
u_gsm=cvals.convert('GSM','car').data

for i,ax in enumerate(['x','y','z']):
    imfdata['u'+ax]=u_gsm[:,i]
imfdata['ux']=np.ma.array(imfdata['ux'],mask=u_gse[:,0].mask)
for comp in 'uy','uz':
    imfdata[comp]=np.ma.array(imfdata[comp],mask=np.logical_or(u_gse[:,1].mask,
                                                           u_gse[:,2].mask))

# Store the times
imfdata['time']=omnidata['Epoch']

# Store the rest of the varibles, masking any invalid values
for omni_key,imfinput_key in zip(omni_keys,imfinput_keys):
    imfdata[imfinput_key]=np.ma.masked_greater(omnidata[omni_key],omnidata[omni_key].attrs['VALIDMAX'])

# Fill in invalid values by linear interpolation
times=imfdata['time']
for key in imfdata.attrs['var']:
    # Find the invalid entries
    mask=imfdata[key].mask

    # Make sure the mask is useable
    try:
        len(mask)
    except TypeError:
        continue

    # Interpolate over the invalid values
    oldtimes=times[np.logical_not(mask)]
    newtimes=times[mask]
    imfdata[key][mask]=interp_timeseries(imfdata[key].compressed(),oldtimes,newtimes)

# Write the data to disk
imfdata.write()
