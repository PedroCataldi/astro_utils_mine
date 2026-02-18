import numpy as np
import pandas as pd




################# Shrinking sphere ##################
def shrinking_sphere(pos_star, vel_star, ms, f_shrink=0.975, min_frac=0.1, min_n=100):

    # Shift to initial guess
    pos = pos_star# - center

    # Total number and threshold
    n_init = len(ms)
    n_thresh = max(int(min_frac * n_init), min_n)

    # Start radius and mask
    r = np.sqrt(np.sum(pos**2, axis=1))
    rmax = np.max(r)

    # Iterative shrinking
    while len(ms) >= n_thresh:
        inside = r <= f_shrink * rmax
        if not np.any(inside):
            break
        pos = pos[inside]
        vel_star = vel_star[inside]
        ms = ms[inside]
        r = np.sqrt(np.sum(pos**2, axis=1))
        rmax = np.max(r)

    # Final mass-weighted centre
    msum = np.sum(ms)
    xcm, ycm, zcm = np.sum(pos * ms[:, None], axis=0) / msum
    vxcm, vycm, vzcm = np.sum(vel_star * ms[:, None], axis=0) / msum

    return xcm, ycm, zcm, vxcm, vycm, vzcm

def shrinking_sphere_old(pos_star,vel_star,ms,center):
    mtotal = np.sum(ms)
    na = len(ms)
    nlow = 0.1*len(ms)

    xpri = np.array(pos_star[:, 0])-center[0]
    ypri = np.array(pos_star[:, 1])-center[1]
    zpri = np.array(pos_star[:, 2])-center[2]
    
    vxpristar = np.array(pos_star[:, 0])
    vypristar = np.array(pos_star[:, 1])
    vzpristar = np.array(pos_star[:, 2])
    
    xsh = np.concatenate((pos_star[:, 0], []), axis=None)
    ysh = np.concatenate((pos_star[:, 1], []), axis=None)
    zsh = np.concatenate((pos_star[:, 2], []), axis=None)  
    vxsh = np.concatenate((vel_star[:, 0], []), axis=None)
    vysh = np.concatenate((vel_star[:, 1], []), axis=None)
    vzsh = np.concatenate((vel_star[:, 2], []), axis=None)               
    msh = np.concatenate((ms, []), axis=None)

    rpri = np.sqrt(np.square(xpri)+np.square(ypri)+np.square(zpri))
    rmax = np.amax(rpri)
    cont = 0
    no_it = 0
    while ((na>=nlow) & (na>=100)):
        a = np.where(rpri <= (0.975*rmax))
        na = len(a[0])
        if na > 0:
            xcut = xsh[a]
            ycut = ysh[a]
            zcut = zsh[a]
            vxcut = vxsh[a]
            vycut = vysh[a]
            vzcut = vzsh[a]     
            mcut = msh[a]
            mtotal = np.sum(mcut)
            xcm0 = np.sum(xcut*mcut)/mtotal
            ycm0 = np.sum(ycut*mcut)/mtotal
            zcm0 = np.sum(zcut*mcut)/mtotal
            vxcm0 = np.sum(vxcut*mcut)/mtotal
            vycm0 = np.sum(vycut*mcut)/mtotal
            vzcm0 = np.sum(vzcut*mcut)/mtotal       
            xpri = xcut-xcm0
            ypri = ycut-ycm0
            zpri = zcut-zcm0
            rpri = np.sqrt(np.square(xpri)+np.square(ypri)+np.square(zpri))
            rmax = np.amax(rpri)
            xsh=xcut
            ysh=ycut
            zsh=zcut
            msh=mcut
            cont=cont+1
        else:
            print("No puedo iterar")  
            no_it = no_it+1
    return xcm0, ycm0, zcm0, vxcm0, vycm0, vzcm0

############## Get Shape eigen values and vectors from Dubinski & Carlberg 1991  ##############

def shape_halo_vector(xpridark_ord, ypridark_ord, zpridark_ord, mass, bin, rmin, rmax):
    rbin=(np.log10(rmax)-np.log10(rmin))/bin
    rr=np.ones(bin)
    
    vala=np.zeros(bin)
    valb=np.zeros(bin)
    valc=np.zeros(bin)
   
    v_a = np.zeros((bin,3))           
    v_b = np.zeros((bin,3))           
    v_c = np.zeros((bin,3))           
   
    r3D = np.sqrt(np.square(xpridark_ord)+np.square(ypridark_ord)+np.square(zpridark_ord)) 
    r3D_ord=r3D[r3D.argsort()]    
    
    for j in range(nbin):
        abin=np.where((r3D_ord > (10**(np.log10(rmin)+j*rbin))) & (r3D_ord < (10**(np.log10(rmin)+(j+1)*rbin))))
        nabin=len(abin[0])
        if (nabin > 5):            
            yu = np.zeros((3,nabin))
            rrd2=np.zeros(nabin)                       
            bbaa=0
            ccaa=0
            yu[0,:]=xpridark_ord[abin]
            yu[1,:]=ypridark_ord[abin]                    
            yu[2,:]=zpridark_ord[abin]  
            rrd2[:] = (np.square(yu[0,:]) + np.square(yu[1,:]) + np.square(yu[2,:]))
            IIaxis = np.zeros((3,3))           
            for s in range(0,3):
                for k in range(0,3):
                    IIaxis[s,k] = np.sum(yu[s,:]*yu[k,:]*mass[abin]/rrd2[:])/np.sum(mass[abin])
            nmax=1000
            nin=0
            rrd2=np.zeros(nabin)
            while (nin <  nmax):
                Ae,v = np.linalg.eig(IIaxis) 
                Aer= Ae.real
                ind = Aer.argsort()
                
                vector_c= v[ind[0]]
                vector_b= v[ind[1]]
                vector_a= v[ind[2]]
                
                bbaa1=np.sqrt(Aer[ind[1]]/Aer[ind[2]])
                ccaa1=np.sqrt(Aer[ind[0]]/Aer[ind[2]])
                delb=bbaa-bbaa1
                delc=ccaa-ccaa1 
                if ((np.abs(delb) < 1e-3) and (np.abs(delc) < 1e-3)):
                    break
                rrd2[:] = np.square(yu[0,:])+np.square(yu[1,:]/bbaa1)+np.square(yu[2,:]/ccaa1)
                for s in range(0,3):
                    for k in range(0,3):
                        IIaxis[s,k] = np.sum(yu[s,:]*yu[k,:]*mass[abin]/rrd2[:])/np.sum(mass[abin])
                nin=nin+1
                if (nin == nmax):
                    print('do not converge')                 
                bbaa=bbaa1
                ccaa=ccaa1 
   
            vala[j]=np.sqrt(Aer[ind[2]])
            valb[j]=np.sqrt(Aer[ind[1]])
            valc[j]=np.sqrt(Aer[ind[0]])
        
            rr[j]=(10**(np.log10(rmin)+j*rbin) + 10**(np.log10(rmin)+(j+1)*rbin))/2.  
            v_a[j,:] = vector_a[:]   
            v_b[j,:] = vector_b[:] 
            v_c[j,:] = vector_c[:] 
                
    return rr, vala, valb,  valc,v_a, v_b, v_c

############## Get Shape ellipicty and prolateness from Dubinski & Carlberg 1991  ##############

def shape_halo_ep(xpridark_ord, ypridark_ord, zpridark_ord, bin, rmin, rmax):
    rbin=(np.log10(rmax)-np.log10(rmin))/bin
    rr=np.ones(bin)
    ratioba=np.zeros(bin)
    ratioca=np.zeros(bin)    
    for j in range(nbin):
        abin=np.where((r3D_ord < (10**(np.log10(rmin)+(j+1)*rbin))))
        nabin=len(abin[0])
        if (nabin > 5):            
            yu = np.zeros((3,nabin))
            rrd2=np.zeros(nabin)                       
            bbaa=0
            ccaa=0
            yu[0,:]=xpridark_ord[abin]
            yu[1,:]=ypridark_ord[abin]                    
            yu[2,:]=zpridark_ord[abin]  
            rrd2[:] = (np.square(yu[0,:]) + np.square(yu[1,:]) + np.square(yu[2,:]))
            IIaxis = np.zeros((3,3))           
            for s in range(0,3):
                for k in range(0,3):
                    IIaxis[s,k] = np.sum(yu[s,:]*yu[k,:]/rrd2[:])
            nmax=1000
            nin=0
            rrd2=np.zeros(nabin)
            while (nin <  nmax):
                Ae,v = np.linalg.eig(IIaxis) 
                Aer= Ae.real
                ind = Aer.argsort()
                bbaa1=np.sqrt(Aer[ind[1]]/Aer[ind[2]])
                ccaa1=np.sqrt(Aer[ind[0]]/Aer[ind[2]])
                delb=bbaa-bbaa1
                delc=ccaa-ccaa1 
                if ((np.abs(delb) < 1e-3) and (np.abs(delc) < 1e-3)):
                    break
                rrd2[:] = np.square(yu[0,:])+np.square(yu[1,:]/bbaa1)+np.square(yu[2,:]/ccaa1)
                for s in range(0,3):
                    for k in range(0,3):
                        IIaxis[s,k] = np.sum(yu[s,:]*yu[k,:]/rrd2[:])
                nin=nin+1
                if (nin == nmax):
                    print('do not converge')                 
                bbaa=bbaa1
                ccaa=ccaa1 
   
            ratioba[j]=bbaa
            ratioca[j]=ccaa
            rr[j]=(10**(np.log10(rmin)+j*rbin) + 10**(np.log10(rmin)+(j+1)*rbin))/2.
            
        L= 1 + np.square(ratioca) + np.square(ratioba) 
        ratio_eps=(1-np.square(ratioca))/2*L
        ratio_prol= (1-2*np.square(ratioba)+np.square(ratioca))/2*L
    return rr, ratio_eps, ratio_prol 

################ Velocity Dispersion ######################

def velocity_dispersion(rotxs, rotys, rotzs, rotvxs, rotvys, rotvzs, 
                        mgal, ropt, nopts, nbin=100, rmin = 0, rmax = 1.5):
#    Sigue el mismo algoritmo que la función de IDL, pero en vez de analizar
#    partícula por partícula usa el comando de pandas para agrupar en cajas. 
#    Se trabaja con el mismo dataframe en caso de ser posibles las restas.
#    rotrs = np.sqrt(np.square(rotxs)+np.square(rotys)) # r2D
# Se ordenan los vectores por orden creciente de rotrs (que se usará para las bins)"""
    ord_index = np.argsort(rotrs)
    rotrs = rotrs[ord_index]
    rotxs = rotxs[ord_index]
    rotys = rotys[ord_index]
    rotzs = rotzs[ord_index]
    rotvxs = rotvxs[ord_index]
    rotvys = rotvys[ord_index]
    rotvzs = rotvzs[ord_index]
    mgal = mgal[ord_index]
    
    # Encontrar los puntos entre el radio máximo y mínimo deseado"""
    rmin = rmin
    rmax = rmax*ropt
    arbin = np.where((rotrs > rmin)&(rotrs < rmax))
    rotrs = rotrs[arbin]
    rotxs = rotxs[arbin]
    rotys = rotys[arbin]
    rotzs = rotzs[arbin]
    rotvxs = rotvxs[arbin]
    rotvys = rotvys[arbin]
    rotvzs = rotvzs[arbin]
    mgal = mgal[arbin]    
    
    # Cálculo velocidad radial (sólo para estrellas en los límites)"""
    costita_disk = rotxs / np.sqrt(rotxs**2.+ rotys**2.)
    sentita_disk = rotys / np.sqrt(rotxs**2.+ rotys**2.)        
    vtans = rotvys * costita_disk - rotvxs * sentita_disk
    vrads = rotvxs * costita_disk + rotvys * sentita_disk
    vzetas = rotvzs
    # Definimos vector con las bins (uniforme)"""
    bins = np.linspace(rmin,rmax, nbin+1)   # Cajas equiespaciadas"""
    # Construimos data frame:"""
    #Pondera cada velocidad con la masa de su estrella, luego se divide por la 
    #masa total de la caja
    msvtans = vtans*mgal 
    msvrads = vrads*mgal
    msvzetas = vzetas*mgal
    df = pd.DataFrame({'r' : rotrs, 'mgal' : mgal, 
                       'msvtans' : msvtans, 'msvrads' : msvrads, 'msvzetas' : msvzetas})  
    # Dividimos los datos de r en las bins con el límite elegido:"""
    data_cut = pd.cut(df.r,bins)          
    # Agrupamos el resto de datos según ese corte"""   
    grp = df.groupby(by = data_cut) 
    
    # Calcular el radio de cada bin como la media:"""
    rmean = np.asarray(grp.r.aggregate(np.nanmean))    
    ret = grp.aggregate(np.sum) # Suma todos los componentes de las velocidades"""
                               # Se ignora la componente r porque la suma no tiene sentido"""
    vtan_bin = np.asarray(ret.msvtans)/np.asarray(ret.mgal)
    vrad_bin = np.asarray(ret.msvrads)/np.asarray(ret.mgal)
    vzeta_bin = np.asarray(ret.msvzetas)/np.asarray(ret.mgal)
    mgal_bin = np.asarray(ret.mgal)

    # Cálculo sigma:"""   
    grp_size = np.asarray(grp.size()) # Tamaño de cada grupo"""
    # A falta de un método mejor, creamos vector de índices en repeticiones"""
    index_vector = np.array([])
    index = int(0)
    for vec in range(0,len(grp_size)):
        ins_vec = np.linspace(index,index,grp_size[vec],dtype=int)
        index_vector = np.concatenate((index_vector, ins_vec), axis=None)        
        index = index+1
    index_vector = index_vector.astype(int)
    vtan_bin_rep = vtan_bin[index_vector]
    vrad_bin_rep = vrad_bin[index_vector]
    vzeta_bin_rep = vzeta_bin[index_vector]
    # Vamos a crear otro data frame para binear también para la dispresión
    # Antes operamos con los vectores"""
    sigma_t = mgal*(vtan_bin_rep-vtans)**2
    sigma_r = mgal*(vrad_bin_rep-vrads)**2
    sigma_zeta = mgal*(vzeta_bin_rep-vzetas)**2
    # Data frame:"""
    dfd = pd.DataFrame({'r' : rotrs, 'mgal' : mgal ,'sigma_t' : sigma_t, 
                        'sigma_r' : sigma_r, 'sigma_zeta' : sigma_zeta})  
    grp2 = dfd.groupby(by = data_cut) 
    ret2 = grp2.aggregate(np.sum) # Obtener el valor para cada grupo"""
    #sigma_s = np.sqrt((vrads-vrad_bin_rep)**2.+(vtans-vtan_bin_rep)**2.+(vzetas-vzeta_bin)**2.)
    #dispersion_s = sigma_s/vtans"""
    sigma_group = np.sqrt((np.asarray(ret2.sigma_t) + np.asarray(ret2.sigma_r) 
            + np.asarray(ret2.sigma_zeta))/np.asarray(ret2.mgal))
    sigma_group_1d = sigma_group/3

    anan = np.isnan(sigma_group)
    binnovacio = np.where(anan==False)    
    # Cálculo del lambda"""
    num = np.nansum(rmean[binnovacio]*np.absolute(vtan_bin[binnovacio]*mgal_bin[binnovacio]))
    den = np.nansum(rmean[binnovacio]*mgal_bin[binnovacio]*np.sqrt(vtan_bin[binnovacio]**2+sigma_group_1d[binnovacio]**2))
    
    lambda_tan = num/den
    return lambda_tan, rmean, vtan_bin, vrad_bin, vzeta_bin


##############  Compute of Density given the Mass radial profile #########

def calc_densidad(rfit,mfit,rmin=1e-2,rmax=15,nbin=200):
# Inicializar vectores de salida"""
    rho_sup = np.zeros(nbin)   # Densidad (toda la componente)"""
    rbin_sup = np.zeros(nbin)  # Radios en los que se calcula la densidad"""
    rmin=rmin
    rmax=rmax
    rbin=(rmax-rmin)/nbin      # Distancia entre divisiones"""
    mbin_tot = 0
    for j in range(0, nbin):
        area_bin = np.pi*(np.square(rmin+(j+1)*rbin)-np.square((rmin+j*rbin)))
        abin = np.where((rfit >= (rmin+j*rbin))&(rfit < (rmin+(j+1)*rbin)))
        nabin = len(abin[0])
        if nabin > 0:
            mbin = np.sum(mfit[abin])
            mbin_tot= mbin_tot + mbin       #masa total encerrada
            rho_sup[j] = mbin/area_bin 
        else:
            rho_sup[j] = 0

        rbin_sup[j] = ((rmin+j*rbin) + (rmin+(j+1)*rbin))/2 # Radio""
        
    return rbin_sup, rho_sup, mbin_tot


def calc_densidad3d(rfit,mfit,rmin=1e-2,rmax=15,nbin=200):
# Inicializar vectores de salida"""
    rho_sup = np.zeros(nbin)   # Densidad (toda la componente)"""
    rbin_sup = np.zeros(nbin)  # Radios en los que se calcula la densidad"""
    rmin=rmin
    rmax=rmax
    rbin=(rmax-rmin)/nbin      # Distancia entre divisiones"""
    mbin_tot = 0
    for j in range(0, nbin):
        vol_bin = (4/3)*np.pi*((rmin+(j+1)*rbin)**3-(rmin+j*rbin)**3)
        abin = np.where((rfit >= (rmin+j*rbin))&(rfit < (rmin+(j+1)*rbin)))
        nabin = len(abin[0])
        if nabin > 0:
            mbin = np.sum(mfit[abin])
            mbin_tot= mbin_tot + mbin       #masa total encerrada
            rho_sup[j] = mbin/vol_bin 
        else:
            rho_sup[j] = 0

        rbin_sup[j] = ((rmin+j*rbin) + (rmin+(j+1)*rbin))/2 # Radio""
    return rbin_sup, rho_sup, mbin_tot
        
################ Profile r mass #############

def profile_r_mass(radius,mass,thirdv,rmin=1e-2,rmax=15,nbin=200):

    df = pd.DataFrame({'r' : radius[radius.argsort()], 'mgal' : mass[radius.argsort()], 'msvtans' : thirdv[radius.argsort()]})
    rmin=np.log(rmin)
    rmax=np.log(rmax)
    bins = np.logspace(rmin,rmax, nbin+1)  
    data_cut = pd.cut(df.r,bins)    
    grp = df.groupby(by = data_cut) 
    rmean = np.asarray(grp.r.aggregate(np.nanmean))    
    ret = grp.aggregate(np.sum) 
    mT_test_bin = np.asarray(ret.msvtans)/np.asarray(ret.mgal)
    mgal_bin = np.asarray(ret.mgal)    
    
    return rmean, mT_test_bin 


##############  Compute Caustic #########

def defcaustic(x, y, ndata, xmin, xmax, delta):
    Nhist = int(np.floor((xmax-xmin)/delta)+1)
    xf = np.zeros(Nhist+1)  
    xf2 = np.zeros(Nhist+1)  
    yf = np.zeros(Nhist+1)  
   
    # No hace falta el bucle for que inicializa en idl porque aquí ya son 0"""
    npp = 3
    i = 0
    ynorm = np.copy(y)       
    
    for i in range (0, Nhist+1):
        xf[i] = xmin + i*delta
        
        abin = np.where((x>xf[i]) & (x<(xf[i]+delta))) 
        nabin = len(abin[0])
        
        if nabin > 10:
           # yf(i)=max(y(abin))"""
            orde = np.argsort(y[abin])
            
            if nabin > npp:
                npa = npp
            else:
                npa = nabin
            
            abin = abin[0]
            arg = orde[(nabin-npa):(nabin)]
            arg = np.asarray(abin[arg])
            
            yf[i] = np.sum(y[arg]/npa)  
                
            # Normalizo para definir el epsilon"""
            ynorm[abin] = y[abin]/yf[i]# Normalizo para definir el epsilon
       
    for i in range(0, Nhist):
        xf2[i] = xf[i] + (xf[i+1]-xf[i])/2
        
    aclean = np.where(yf != 0)
    yf3 = yf[aclean]
    xf3 = xf2[aclean]
    
    return xf3, yf3

############## Compute epsilon=E/Ecirc ##########

def epsilon(mgal, rotxs, rotys, rotzs, ropt, bins = 50, rmin = 0.5, rmax = 2):
    rmin = rmin
    rmax = rmax*ropt
    bins = bins
    r3D = np.sqrt(np.square(rotxs)+np.square(rotys)+np.square(rotzs)) 
    ratioba = np.ones(bins)*np.nan
    ratioca = np.ones(bins)*np.nan
    rbinlog = np.linspace(np.log10(rmin), np.log10(rmax), bins+1)
    #rbin = 10**rbinlog
    epsilon_ba = np.nan
    epsilon_ca = np.nan
    repsilon = np.nan
    a = np.nan 
    b = np.nan
    c = np.nan
    # Momento de inercia con fórmula real (Trayford tiene los signos mal)
    for j in range(0,bins):
        abin = np.where(r3D < 10**rbinlog[j+1])
        nabin = len(abin[0])
        
        if nabin>300:
            yu = np.zeros((3,nabin))
            yu[0,:]=rotxs[abin]
            yu[1,:]=rotys[abin]                    
            yu[2,:]=rotzs[abin]                    
            yu_mod = vec3_module(yu[0,:], yu[1,:], yu[2,:])
            
            IIaxis = np.zeros((3,3))
            for s in range(0,3):
                for k in range(0,3):
                    if s!=k:
                        IIaxis[s,k] = -np.sum(yu[s,:]*yu[k,:]/yu_mod)
                    else:
                        IIaxis[s,k] = np.sum((yu_mod**2-yu[s,:]**2)/yu_mod)
            w_raw,v = np.linalg.eig(IIaxis)  
            w = np.sort(w_raw)
            
            I1 = w[0]
            I2 = w[1]
            I3 = w[2]
            
            a = np.sqrt(I2+I3-I1)
            b = np.sqrt(I1+I3-I2)
            c = np.sqrt(I2+I1-I3)

            
            ratioba[j] = np.sqrt((I1+I3-I2)/(I2+I3-I1))
            ratioca[j] = np.sqrt((I1+I2-I3)/(I2+I3-I1))
            
            
            epsilon_ca = 1-ratioca[j]
            epsilon_ba = 1-ratioba[j]

            repsilon = 10**rbinlog[j]

    return epsilon_ba, epsilon_ca, repsilon, a, b, c    

############ Density Theoritcal Profiles ###########

################# Theorical density profile #################

def NFW(x,rs,dc):

    F = np.log(rho_crit*dc/(x/rs)/(1.+x/rs)**2.)
    return F

def einas(x,n,r_2,Mt,rvir):
    x_vir=2*n*(rvir/r_2)**(1/n)
    rho_2=Mt*np.exp(-2*n)*r_2**(-3.)*(2*n)**(3*n)/4./math.pi/n/math.gamma(3*n)/sc.gammainc(3*n,x_vir)                                                         
    rho = np.log(rho_2)-2*n*((x/r_2)**(1/n)-1)

    return rho

################# Get Encolsed Mass radil profile #################

def mass_profile(r, mass):
    idx = r.argsort()
    cumsum = mass[idx].cumsum()
    return r[idx], cumsum
    
################# Get Optical Radius #################

def get_optical_radius(data, frac=0.83):
    baryons_R = np.array([])
    baryons_mass = np.array([])
    for data_type in ['gas', 'stars']:
        R = data[data_type]["R"]
        mass = data[data_type]["mass"]
        baryons_R = np.append(baryons_R, R)
        baryons_mass = np.append(baryons_mass, mass)
    
    R, cumsum = mass_profile(baryons_R, baryons_mass)
    
    mask = cumsum <= frac * cumsum.max()
    
    return R[mask].max()
    
################# Get Optical Radius with Rmax #################

def get_optical_radius_cut_DeRossi(data, aexp=1.0,frac=0.83):
    
    baryons_R = np.array([])
    baryons_mass = np.array([])
    for data_type in ['gas', 'stars']:
        R = data[data_type]["R"]
        mass = data[data_type]["mass"]
        baryons_R = np.append(baryons_R, R)
        baryons_mass = np.append(baryons_mass, mass)
    ##################### galaxy disc ############
    if np.sum(baryons_mass) <= 0.123480*1e10: # Corte de Mari
        rcut = 50 * np.sqrt(np.sum(baryons_mass)/1e10/0.7) * aexp    
    else:
        rcut = 30  * aexp 
    ##############################################    
    index_r = np.where(baryons_R<rcut)
    
    R, cumsum = mass_profile(baryons_R[index_r], baryons_mass[index_r])        
    mask = cumsum <= frac * cumsum.max()
        
    return R[mask].max()


################# SMHM Semi-Empirical #################

def Moster2018(Maux, z):
    epsilonN = 0.15/3 +0.689*(z/(z+1))
    M1 = 11.339+ 0.692*(z/(z+1))
    beta = 3.344 -2.079*(z/(z+1))
    gamma = 0.966
    return 2*epsilonN*(((Maux/10**(M1))**-beta + (Maux/10**(M1))**gamma)**(-1.))

def Moster2018_4(Maux):
    epsilonN = 0.2
    M1 = 12.07
    beta = 1.36
    gamma = 0.6
    
    return 2*epsilonN*(((Maux/10**(M1))**-beta + (Maux/10**(M1))**gamma)**(-1.))

def Moster2018_8(Maux):
    epsilonN = 0.24
    M1 = 12.10
    beta = 1.30
    gamma = 0.6
    
    return 2*epsilonN*(((Maux/10**(M1))**-beta + (Maux/10**(M1))**gamma)**(-1.))

def sigma_SMHM(Maux,sigma0,M_sigma,alfa):
    
    sigma=sigma0+np.log10((Maux/M_sigma)**(-alfa)+1)
    
    return sigma
    
######### Sersic profile ################

def fit_Sersic_fun(r, Sd00, Rdd, nn):
    rho_fit = Sd00*np.exp(-((r/Rdd)**(1/nn)))
    return rho_fit