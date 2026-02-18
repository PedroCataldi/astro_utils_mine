import numpy as np

############### Profile r tub weightged ################
def profile_r_tab(radius,mass,thirdv,rmin=1e-3,rmax=20,nbin=50):
    df = pd.DataFrame({'r' : radius[radius.argsort()], 'mgal' : mass[radius.argsort()], 'msvtans' : thirdv[radius.argsort()]})
    rmin=np.log10(rmin)
    rmax=np.log10(rmax)
    bins = np.logspace(rmin,rmax, nbin+1)  
    data_cut = pd.cut(df.r,bins)    
    grp = df.groupby(by = data_cut) 
    rmean = np.asarray(grp.r.aggregate(np.nanmean))    
    ret = grp.aggregate(np.sum) 
    mT_test_bin = np.asarray(ret.msvtans)/np.asarray(ret.mgal)
    mgal_bin = np.asarray(ret.mgal)    
    
    
    return rmean, mT_test_bin 
########### Moving median ########################

def moving_median(x,y,z):
    step_size=(np.max(x)-np.min(x))/(len(x)/2)
    bin_centers  = np.arange(np.min(x),np.max(x)-0.5*step_size,step_size)+0.5*step_size
    bin_avg = np.zeros(len(bin_centers))

    for index in range(0,len(bin_centers)):
        bin_center = bin_centers[index]
        items_in_bin = y[(x>(bin_center-z*0.5) ) & (x<(bin_center+z*0.5))]
        bin_avg[index] = np.median(items_in_bin)

    return bin_centers,bin_avg

########### Moving 25 and 75 percentile ########################

def moving_percentile75(x,y,z):
    step_size=(np.max(x)-np.min(x))/(len(x)/2)
    bin_centers75  = np.arange(np.min(x),np.max(x)-0.5*step_size,step_size)+0.5*step_size
    bin_avg75 = np.zeros(len(bin_centers75))

    for index in range(0,len(bin_centers75)):
        bin_center75 = bin_centers75[index]
        items_in_bin = y[(x>(bin_center75-z*0.5) ) & (x<(bin_center75+z*0.5))]
        bin_avg75[index] = np.percentile(items_in_bin,75)

    return bin_centers75,bin_avg75


def moving_percentile25(x,y,z):
    step_size=(np.max(x)-np.min(x))/(len(x)/2)
    bin_centers25  = np.arange(np.min(x),np.max(x)-0.5*step_size,step_size)+0.5*step_size
    bin_avg25 = np.zeros(len(bin_centers25))

    for index in range(0,len(bin_centers25)):
        bin_center25 = bin_centers25[index]
        items_in_bin = y[(x>(bin_center25-z*0.5) ) & (x<(bin_center25+z*0.5))]
        bin_avg25[index] = np.percentile(items_in_bin,25)

    return bin_centers25,bin_avg25

### Compute J vector given the pos and vel ##############

def calc_J(m, x, y, z, vx, vy, vz):
    Jx = (m*(y*vz - z*vy))
    Jy = (m*(z*vx - x*vz))
    Jz = (m*(x*vy - y*vx))           
    return Jx, Jy, Jz    
### Find nearest index value from an yarray, given an xvalue ##############

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

### Given the cosmology, the integrated time ##############


def integrated_time(age, Omega0, OmegaLambda, time):
    
    Hub_1=3.24077e-18*3.0856e21/1e5

    hubble_a = Hub_1*np.sqrt(Omega0 / (time * time * time)
      + (1 - Omega0 - OmegaLambda) / (time * time) +OmegaLambda)
    
    time_hubb = time * hubble_a

    t1 = age
    t3 = time
    t2 = t1 + (t3 - t1) / 2
    
    f1 = 1 / (t1 * Hub_1 * np.sqrt(Omega0 / (t1 * t1 * t1) 
         + (1 - Omega0 - OmegaLambda) / (t1 * t1) + OmegaLambda))
    
    f2 = 1 / (t2 * Hub_1 * np.sqrt(Omega0 / (t2 * t2 * t2)
         + (1 - Omega0 - OmegaLambda) / (t2 * t2) + OmegaLambda))
   
    f3 = 1 / time_hubb
    
    deltat = (t3 - t1) / 2. * (f1 / 3. + 4. / 3. * f2 + f3 / 3.) 

    return deltat

############# Not duplicate legends ##############

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),frameon=False, fontsize=15)
    
############# # Get unique elements of A and B and the indices based on the uniqueness ##############

def unq_searchsorted(A,B):

    # Get unique elements of A and B and the indices based on the uniqueness
    unqA,idx1 = np.unique(A,return_inverse=True)
    unqB,idx2 = np.unique(B,return_inverse=True)

    # Create mask equivalent to np.in1d(A,B) and np.in1d(B,A) for unique elements
    mask1 = (np.searchsorted(unqB,unqA,'right') - np.searchsorted(unqB,unqA,'left'))==1
    mask2 = (np.searchsorted(unqA,unqB,'right') - np.searchsorted(unqA,unqB,'left'))==1

    # Map back to all non-unique indices to get equivalent of np.in1d(A,B), 
    # np.in1d(B,A) results for non-unique elements
    return mask1[idx1],mask2[idx2]

#################### Get an annotation line ############

def annotation_line( ax, xmin, xmax, y, text, ytext=0, linecolor='black', linewidth=1, fontsize=12 ):

    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '-', 'color':linecolor, 'linewidth':linewidth})
    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '<->', 'color':linecolor, 'linewidth':linewidth})

    xcenter = xmin + (xmax-xmin)/2
    if ytext==0:
        ytext = y + ( ax.get_ylim()[1] - ax.get_ylim()[0] ) / 20

    ax.annotate( text, xy=(xcenter,ytext), ha='center', va='center', fontsize=fontsize,color='w')

############### Colored line ##############3


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)