import site
import sys

sys.path.append('/home/jovyan')
site.addsitedir('/home/jovyan') 

import illustris_python as il
import illustris_sam as ilsam
import scipy.spatial as spt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const

# Constants in CGS units
gamma = 5.0 / 3.0
k_B = const.k_B.cgs.value         # Boltzmann constant [erg/K]
m_p = const.m_p.cgs.value         # Proton mass [g]
X_H = 0.76                        # Hydrogen mass fraction
#####################################

def compute_mu_custom(Ye):

    return 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * Ye)

def temp(InternalEnergy, ElectronAbundance):

    u = np.asarray(InternalEnergy)
    Ye = np.asarray(ElectronAbundance)

    mu = compute_mu_custom(Ye)

    # Convert u to erg/g: (km/s)^2 = 1e10 (cm/s)^2
    u_cgs = u * 1e10

    T = (gamma - 1.0) * mu * m_p * u_cgs / k_B
    return T

def delta_time_from_scale_factors(a1, a2, cosmo):

    z1 = (1.0 / a1) - 1
    z2 = (1.0 / a2) - 1

    t1 = cosmo.age(z1)
    t2 = cosmo.age(z2)

    return abs((t2 - t1).to(u.Gyr).value)*1e9


def periodicfix(pos, center, boxsize):

    d = pos - center
    d = (d + boxsize / 2) % boxsize - boxsize / 2  
    pos_corrected = center + d
    return pos_corrected, center

#####################################

#basepath
bp = '/home/jovyan/SimulationData/L35n2160TNG/output/'

#boxsize in ckpc/h
boxsize = 35000
#subhalo index
subhalo = int(sys.argv[1])
#snapshot number
snap = 21

#load header
header = il.groupcat.loadHeader(bp, snap)
#load current scale factor
a2 = header["Time"][()]
hh = header['HubbleParam'][()]
Omega0 = header['Omega0'][()]
OmegaLambda = header['OmegaLambda'][()]
Omega = Omega0+OmegaLambda
OmegaR = 0
cosmoTNG = FlatLambdaCDM(H0=hh*100, Om0=Omega0)
#load SubhaloPosition in ckpc/h
center=il.groupcat.loadSingle(bp,snap,subhaloID=subhalo)["SubhaloPos"]

#take 15 times stellar halfmassradius in proper pc as boxsize for dust calcualtion in SKIRT (like Rodriguez-Gomez)
size = np.array([(il.groupcat.loadSingle(bp,snap,subhaloID=subhalo)["SubhaloHalfmassRadType"][4])*a2*7.5*1000/hh])
twosize = 2*size
np.savetxt("shalo"+str(subhalo)+"_size.dat",size)
np.savetxt("shalo"+str(subhalo)+"_twosize.dat",twosize)

#define temperature threshold for dust formation
Tthreshold = 75000

#########################~~~~~~~~~~~~~~~~~OLDSTARS~~~~~~~~~~~~~~~~~~########################


fields = ["Coordinates","GFM_StellarFormationTime","GFM_InitialMass","GFM_Metallicity"]

#load stars
stars= il.snapshot.loadSubhalo(bp,snap,subhalo,4,fields)
#ignore all wind particles
if (stars["count"]!=0):
    starindex = np.where(stars["GFM_StellarFormationTime"] > 0)

#star positions in proper pc, centered on position of halo
if (stars["count"]!=0):
    fix = periodicfix(stars["Coordinates"][starindex],center,boxsize)
    starpos = (fix[0])*1000*a2
    centerpc = (fix[1])*1000*a2

    x = (starpos[:,0]-centerpc[0])/hh
    y = (starpos[:,1]-centerpc[1])/hh
    z = (starpos[:,2]-centerpc[2])/hh
else:
    x = []
    y = []
    z = []

#smoothing length in proper pc (distance to 32th closest neighbour) (like Rodriguez-Gomez)
if (stars["count"]!=0):
    #create array with locations
    loc = np.array([x,y,z]).T
    #setup KDTree
    tree = spt.cKDTree(loc,leafsize = 100)
    h = np.zeros(len(x))
    i = 0
    for point in loc:
        dist = tree.query(point, k= 32)[0][-1]
        h[i] = dist
        i += 1
else:
    h = []

#initial mass in solar masses
if (stars["count"]!=0):
    M = np.array((stars["GFM_InitialMass"][starindex]))*1e10/hh
else:
    M = []

#metallicity (not solar metallicity)
if (stars["count"]!=0):
    Z = stars["GFM_Metallicity"][starindex]
else:
    Z = []

#age of the stars in years
if (stars["count"]!=0):
    t = delta_time_from_scale_factors(stars["GFM_StellarFormationTime"][starindex],a2,cosmoTNG)
else:
    t = []


#only consider stars older than 10Myr for bruzual charlot spectra
xb = x[t > 1e7]
yb = y[t > 1e7]
zb = z[t > 1e7]
hb = h[t > 1e7]
Mb = M[t > 1e7]
Zb = Z[t > 1e7]
tb = t[t > 1e7]

#create 2D-array and write into data file
totalb = np.column_stack((xb,yb,zb,hb,Mb,Zb,tb))
np.savetxt("shalo"+str(subhalo)+"_stars.dat",totalb,delimiter = " ")

#######################################~~~~YOUNGSTARS~~~~~~~###################################

#only consider stars younger than 10Myr for mappings-iii spectra and that have scale factor at formation time of a < 1.0 (so effectively tm > 0)

xm = x[t <= 1e7]
ym = y[t <= 1e7]
zm = z[t <= 1e7]
hm = h[t <= 1e7]
Mm = M[t <= 1e7]
Zm = Z[t <= 1e7]
tm = t[t <= 1e7]

#calculate SFR (assuming SFR has been constant in the last 10Myr)
SFRm = Mm/1e7

#for compactness, take typical value of 5:
logC = np.full(len(SFRm),5)

#for ISM pressure, take typical value log(P/kB / cm^-3K) = 5, P = 1.38e-12 N/m^2
pISM = np.full(len(SFRm),1.38e-12)

#for PDR covering fraction, take f = 0.2
fPDR = np.full(len(SFRm),0.2)


#create 2D-array and write into data file
totalm = np.column_stack((xm,ym,zm,hm,SFRm,Zm,logC,pISM,fPDR))
np.savetxt("shalo"+str(subhalo)+"_mappings.dat",totalm,delimiter = " ")


#######################################~~~~~~~~~GAS (VORONOI)~~~~~~~~~~~####################################

fields = ["Coordinates","Density","GFM_Metallicity","InternalEnergy","ElectronAbundance","StarFormationRate"]

#load gas particles
gas= il.snapshot.loadSubhalo(bp,snap,subhalo,0,fields=fields)

#gas particle positions in proper pc
if (gas["count"]!=0):
    fix = periodicfix(gas["Coordinates"],center,boxsize)
    gaspos = (fix[0])*1000*a2
    centerpc = (fix[1])*1000*a2

    xg = (gaspos[:,0]-centerpc[0])/hh
    yg = (gaspos[:,1]-centerpc[1])/hh
    zg = (gaspos[:,2]-centerpc[2])/hh
else:
    xg = []
    yg = []
    zg = []

#gas cell density in solar masses per proper pc^3
if (gas["count"]!=0):
    rhog = (gas["Density"]*hh)/((1000*a2)**3)
else:
    rhog = []

#gas particle metallicity (not in solar units)
if (gas["count"]!=0):
    Zg = gas["GFM_Metallicity"]
else:
    Zg = []

#gas temperature in K
if (gas["count"]!=0):
    T = temp(gas["InternalEnergy"],gas["ElectronAbundance"])
else:
    T = []

#SFR of gas cell
if (gas["count"]!=0):
    SFRg = gas["StarFormationRate"]
else:
    SFRg = []

#set density to zero where Temperature is above threshold value and where there is no SFR
if (gas["count"]!=0):
    for i in range(len(xg)):
        if (T[i] > Tthreshold) and (SFRg[i] == 0):
            Zg[i] = 0

numcells = np.array([len(xg)])
np.savetxt("shalo"+str(subhalo)+"_numcells.dat",numcells, fmt="%i")

#create 2D-array and write into data file
totalg = np.column_stack((xg,yg,zg,rhog,Zg))
np.savetxt("shalo"+str(subhalo)+"_gas.dat",totalg,delimiter = " ")
