import numpy as np


########################### Radial Profiles over Gas, Stars, DM ###########################

def FL_all_properties(j,box,MyRedshift,run_interested,FlatLambdaCDM,Mstar_opt_arr,rhm_star_arr,rhm_coldgas_arr,radial_coldden_arr,
                           radial_gasSFR_arr,radial_T_arr,radial_Z_arr,radial_fcold_arr,radial_vr_arr,radial_macc_arr,radial_mout_arr,sup_stellar,sup_gassfr,sup_sfr,
                           sfr_arr,mstar_stack_arr,mgas_stack_arr,mdm_stack_arr,vol_size=5.,nbin=200,rmin=1e-2,rmax=15):
    #####################################
    run=run_interested
    print(run_interested)

    rmin=np.log(rmin)
    rmax=np.log(rmax)
    rbins = np.logspace(rmin,rmax, nbin+1)         

    ####################################
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    Ob0=0.05
    mp=1.67262192e-24
    factor = 1/mp   
    To=1e4

    ####################################
    print("z",MyRedshift[j])   

    MyExpFactor = 1 / (1 + MyRedshift[j])
    MyExpFactor = np.around(MyExpFactor, 3)

    mp=1.67262192e-24
    factor = 1/mp
    rho_mean_z = (cosmo.critical_density(MyRedshift[j]).value*Ob0*factor)#* density_units.to("Msun/pc3").value

    
    if (box=='Box40Mpc'):
        
        path ="/scratch/danielcv/FirstLight/Outputs_FirstLight_Massive/FL"+str(run)+"/"
        text_files = [f for f in os.listdir(path) if f.endswith("{:0.3f}".format(MyExpFactor)+'.dat')]
        cutout = text_files[0][len("FL"+str(run)+"_S"):len("FL"+str(run)+"_")+4]      
        
        snap_info = {"basedir": "",
            "simulation": "/scratch/danielcv/FirstLight/Outputs_FirstLight_Massive/FL"+str(run)+"/FL"+str(run),
            "cutout_size": int(cutout),
            "expansion_factor": MyExpFactor,} 

    else:
        path ="/media/FirstLight/FL"+str(run)+"/"
        text_files = [f for f in os.listdir(path) if f.endswith("{:0.3f}".format(MyExpFactor)+'.dat')]
        cutout = text_files[0][len("FL"+str(run)+"_S"):len("FL"+str(run)+"_")+4]  
        
        snap_info = {"basedir": "",
            "simulation": "/media/FirstLight/FL"+str(run)+"/FL"+str(run),
            "cutout_size": int(cutout),
            "expansion_factor": MyExpFactor,}   
    

    virial_radius = snap_info["cutout_size"] / 4
    
    try:
        data_types = ["stars", "gas","dark_matter"]

        data = {}
        for data_type in data_types:
            data[data_type] = load_data(data_type=data_type, **snap_info)

        xcm0,ycm0,zcm0,vxcm0,vycm0,vzcm0 = shrinking_sphere(data["stars"]["position"],data["stars"]["velocity"],data["stars"]["mass"])

        # compute radial coordinate from center
        for data_type in data_types:
            #print(data_type, data[data_type].keys())
            pos = data[data_type]["position"]
            data[data_type]["R"] = np.linalg.norm(pos, axis=1)
            data[data_type]["R2d"] = np.linalg.norm(pos[:, [1, 2]], axis=1)
            data[data_type]["position"]=data[data_type]["position"]-[xcm0,ycm0,zcm0]
            data[data_type]["velocity"]=data[data_type]["velocity"]-[vxcm0,vycm0,vzcm0]
        # compute the mass for gas cells

        sel_gas = data["gas"]

        # not sure if this is correct
        density_units =  1.66e-24 * units.g * units.cm**-3
        density = sel_gas["density"] * density_units.to("Msun/pc3").value

        sel_gas["mass"] = sel_gas["cell_size"]**3 * density   

        data_type = "stars"
        position = data[data_type]["position"]
        velocity = data[data_type]["velocity"]
        mass = data[data_type]["mass"]
        R = data[data_type]["R"]
        rcut = 0.15*virial_radius


        # Half Mass Stellar radius
        rpristar_ord = R[R.argsort()]
        mstar_ord = mass[R.argsort()]
        acut = np.where(rpristar_ord <= rcut)
        ncut = len(acut[0])         
        if ncut > 0:
            rgalcut = rpristar_ord[acut]
            mgalcut = mstar_ord[acut]
            mcut = np.sum(mgalcut)

            massb50 = 0.50*mcut
            maux = 0
            t = 0
            while t<ncut:
                if maux<massb50:
                    maux = maux + mgalcut[t]
                    rhm_star=rgalcut[t]
                t = t+1         
        ################### inside rhm ##################


        aropt = np.where(data["stars"]["R"] <= rhm_star)    
        narop = len(aropt[0])
        if narop > 0:
            mstar_hm = np.sum(data["stars"]["mass"][aropt])
        else:
            mstar_hm = 0 

        aropt = np.where(data["gas"]["R"] <= rhm_star)    
        narop = len(aropt[0])
        if narop > 0:
            mgas_hm = np.sum(data["gas"]["mass"][aropt])
        else:
            mgas_hm = 0          

        aropt = np.where(data["dark_matter"]["R"] <= rhm_star)    
        narop = len(aropt[0])
        if narop > 0:
            mdm_hm = np.sum(data["dark_matter"]["mass"][aropt])
        else:
            mdm_hm = 0    

        ########### append ########
        #mstar_hm_arr.append(mstar_hm)
        #mgas_hm_arr.append(mgas_hm)
        #mdm_hm_arr.append(mdm_hm)                      

        # Ropt with all baryons
        rgal = np.array([])
        mgal = np.array([])
        for data_type in ['gas', 'stars']:
            R = data[data_type]["R"]
            mass = data[data_type]["mass"]
            rgal = np.append(rgal, R)
            mgal = np.append(mgal, mass)

        # Ropt
        rgal_ord = rgal[rgal.argsort()]
        mgal_ord = mgal[rgal.argsort()]

        acut = np.where(rgal_ord <= rcut)
        ncut = len(acut[0]) 
        if ncut > 0:
            rgalcut = rgal_ord[acut]
            mgalcut = mgal_ord[acut]
            mcut = np.sum(mgalcut)

            massb50 = 0.83*mcut
            maux = 0
            t = 0
            while t<ncut:
                if maux<massb50:
                    maux = maux + mgalcut[t]
                    ropt=rgalcut[t]

                t = t+1                


        ################### inside ropt ##################
        aropt = np.where(R <= ropt)    
        nsrop = len(aropt[0])
        if nsrop > 0:
            mstar_ropt = np.sum(mass[aropt])
        else:
            mstar_ropt = 0                 

        ################################
        # limit to 1.5 times the optical radius
        mask = R < 2.0 * rhm_star 

        position = position[mask]
        velocity = velocity[mask]
        mass = mass[mask].reshape(-1, 1)
        # compute angular momentum

        specific_angular_momentum = np.cross(position, velocity)
        angular_momentum = mass * specific_angular_momentum
        total_angular_momentum = angular_momentum.sum(axis=0)
        total_specific_angular_momentum = specific_angular_momentum.sum(axis=0)
        L_dir = total_specific_angular_momentum / np.linalg.norm(total_specific_angular_momentum)

        # compute rotation matrix to align reference frame with the angular momentum

        rot_matrix = get_rotation_matrix(total_specific_angular_momentum)
        factor_kms_MpcYr=1.02201e-12

        for data_type in data_types:

            rot_position = rot_matrix.dot(data[data_type]["position"].T).T
            rot_vel = rot_matrix.dot(data[data_type]["velocity"].T).T

            r, theta, phi, v_r, v_t, v_p= cartesian_to_spherical(rot_position[:,0], rot_position[:,1], rot_position[:,2], rot_vel[:,0], rot_vel[:,1], rot_vel[:,2])


            data[data_type]["v_radial"] = v_r
            data[data_type]["M_infall"] = data[data_type]["v_radial"]*data[data_type]["mass"]*factor_kms_MpcYr    

            R = data[data_type]["R"]
            mgal = data[data_type]["mass"]

            Rcum, cumsum = mass_profile(R, mgal)
            data[data_type]["Rcum"] = Rcum
            data[data_type]["Mass_Int"] = cumsum 
            
        
        ######### Matter stack ##########
        
        
        r_M_star= data["stars"]["Rcum"]
        M_star= data["stars"]["Mass_Int"]                  
        rmean, mstar_hm = profile_r_mass(r_M_star[r_M_star<15],np.ones(len(M_star[r_M_star<15])) ,M_star[r_M_star<15])
        mstar_stack_arr.append(mstar_hm)  
        

        r_M_dm=  data["dark_matter"]["Rcum"]
        M_dm= data["dark_matter"]["Mass_Int"]                  
        
        rmean, mdm_hm = profile_r_mass(r_M_dm[r_M_dm<15],np.ones(len(M_dm[r_M_dm<15])) ,M_dm[r_M_dm<15])
        mdm_stack_arr.append(mdm_hm)  
        

        r_M_gas=  data["gas"]["Rcum"]
        M_gas = data["gas"]["Mass_Int"]
       
        
        rmean, mgas_hm = profile_r_mass(r_M_gas[r_M_gas<15],np.ones(len(M_gas[r_M_gas<15])) ,M_gas[r_M_gas<15])
        mgas_stack_arr.append(mgas_hm)  

        ################################ norm ###################
        data_type = "gas"

        Temp_all = data[data_type]["temperature"]
        den_all = data[data_type]["density"]
        cond_g_dencold = ((Temp_all< 1e4)&(den_all>1.))

        mass_cold = data[data_type]["mass"][cond_g_dencold]
        R_cold = data[data_type]["R"][cond_g_dencold]

        # Half Gas-Cold radius
        rprigas_ord = R_cold[R_cold.argsort()]
        mgas_ord = mass_cold[R_cold.argsort()]
        acut = np.where(rprigas_ord <= rcut)
        ncut = len(acut[0])         
        if ncut > 0:
            rgalcut = rprigas_ord[acut]
            mgalcut = mgas_ord[acut]
            mcut = np.sum(mgalcut)

            massb50 = 0.50*mcut
            maux = 0
            t = 0
            while t<ncut:
                if maux<massb50:
                    maux = maux + mgalcut[t]
                    rhm_coldgas=rgalcut[t]
                t = t+1        
        else:
            rhm_coldgas=0.

        rhm_star_arr.append(rhm_star)
        rhm_coldgas_arr.append(rhm_coldgas)
        Mstar_opt_arr.append(mstar_ropt)

        ############## Gas ################
        data_type = "gas"                                                                           
        condition = (data[data_type]["R"]< 15.) 

        R = data[data_type]["R"][condition]
        mgal = data[data_type]["mass"][condition] 
        T = data[data_type]["temperature"][condition]  
        den= data[data_type]["density"][condition]
        Metalicity= data[data_type]['metals_mass_fraction_SNII'][condition]
        Vradial= data[data_type]["v_radial"][condition]
        Macc= data[data_type]["M_infall"][condition]    

        To=1e4

        #Density Cold Gas:

        mT_test = (den/rho_mean_z)*mgal 
        rmean, mT_test_bin = profile_r_mass(R[T<To] ,mgal[T<To] ,mT_test[T<To])
        radial_coldden_arr.append(mT_test_bin)

        #Density SFR Cold Gas:

        mT_test = (den/rho_mean_z)*mgal 
        rmean, mT_test_bin = profile_r_mass(R[(T<To)&(den>1)] ,mgal[(T<To)&(den>1)] ,mT_test[(T<To)&(den>1)])
        radial_gasSFR_arr.append(mT_test_bin)

        #Sup density SFR Cold Gas:

        mT_test = (den/rho_mean_z)*mgal 
        rmean, mT_test_bin = profile_r_mass(R[(T<To)&(den>1)] ,mgal[(T<To)&(den>1)] ,mT_test[(T<To)&(den>1)])
        sup_gassfr.append(mT_test_bin*rmean*2)        

        #Tempeture Cold Gas:

        mT_test = T*mgal 
        rmean, mT_test_bin = profile_r_mass(R[T<To] ,mgal[T<To] ,mT_test[T<To])
        radial_T_arr.append(mT_test_bin)

        #Metallicity Cold Gas 

        mT_test = Metalicity*mgal 
        rmean, mT_test_bin = profile_r_mass(R[T<To] ,mgal[T<To] ,mT_test[T<To])
        radial_Z_arr.append(mT_test_bin)    

        #Fraction of cold gas

        third = mgal
        mT_testCold=np.ones(len(third))
        mT_testCold[T<To] = third[T<To]
        fraction=mT_testCold
        rmean, mT_test_bin = profile_r_mass(R,mgal,fraction)
        radial_fcold_arr.append(mT_test_bin)        

        #Vrot Gas

        mT_test = Vradial*mgal  
        rmean, mT_test_bin = profile_r_mass(R[(T<To)&(mT_test<0)] ,mgal[(T<To)&(mT_test<0)] ,mT_test[(T<To)&(mT_test<0)])    
        radial_vr_arr.append(mT_test_bin)    


        #Macc Cold Gas:

        mT_test = Macc*mgal 
        rmean, mT_test_bin = profile_r_mass(R[(T<To)&(mT_test<0)],mgal[(T<To)&(mT_test<0)],mT_test[(T<To)&(mT_test<0)])
        mT_test_binMacc= mT_test_bin*(rmean**2)/(np.diff(rbins**3))*3e3

        radial_macc_arr.append(mT_test_binMacc)   
        
        #Mout Cold Gas:

        mT_test = Macc*mgal 
        rmean, mT_test_bin = profile_r_mass(R[(T<To)&(mT_test>0)],mgal[(T<To)&(mT_test>0)],mT_test[(T<To)&(mT_test>0)])
        mT_test_binMout= mT_test_bin*(rmean**2)/(np.diff(rbins**3))*3e3

        radial_mout_arr.append(mT_test_binMout)      
        ################### Stellar ####################
        data_type = "stars"                                                                           

        condition = (data[data_type]["R2d"]< 15.) 
        R = data[data_type]["R2d"][condition]   
        mgal = data[data_type]["mass"][condition] 

        rbin_sup, rho_sup, mbin_tot = calc_densidad(R[R.argsort()], mgal[R.argsort()])
        sup_stellar.append(rho_sup)        

        ################### Sup SFR ####################
        condition_SFR = (data[data_type]["R2d"]< 15.) & (data[data_type]["age"]*1e9 < 1e7) 

        R = data[data_type]["R2d"][condition_SFR]
        mgal = data[data_type]["mass"][condition_SFR]

        rbin_sup, rho_sup, mbin_tot = calc_densidad(R[R.argsort()], mgal[R.argsort()])
        sup_sfr.append(rho_sup)        

        ################### SFR ####################
        condition_SFR = (data[data_type]["R"]< 15.) & (data[data_type]["age"]*1e9 < 1e7) 

        R = data[data_type]["R"][condition_SFR]
        mT_test = data[data_type]["mass"][condition_SFR]
        R = data[data_type]["R"][condition_SFR]
        mgal = data[data_type]["mass"][condition_SFR]

        rbin_sup, rho_sup, mbin_tot = calc_densidad3d(R[R.argsort()], mgal[R.argsort()])        
        sfr_arr.append(rho_sup)   
         
        
    except FileNotFoundError:
        
        data_types = ["stars","dark_matter"]

        data = {}
        for data_type in data_types:
            data[data_type] = load_data(data_type=data_type, **snap_info)

        xcm0,ycm0,zcm0,vxcm0,vycm0,vzcm0 = shrinking_sphere(data["stars"]["position"],data["stars"]["velocity"],data["stars"]["mass"])

        # compute radial coordinate from center
        for data_type in data_types:
            #print(data_type, data[data_type].keys())
            pos = data[data_type]["position"]
            data[data_type]["R"] = np.linalg.norm(pos, axis=1)
            data[data_type]["R2d"] = np.linalg.norm(pos[:, [1, 2]], axis=1)
            data[data_type]["position"]=data[data_type]["position"]-[xcm0,ycm0,zcm0]
            data[data_type]["velocity"]=data[data_type]["velocity"]-[vxcm0,vycm0,vzcm0]


        data_type = "stars"
        position = data[data_type]["position"]
        velocity = data[data_type]["velocity"]
        mass = data[data_type]["mass"]
        R = data[data_type]["R"]
        rcut = 0.15*virial_radius


        # Half Mass Stellar radius
        rpristar_ord = R[R.argsort()]
        mstar_ord = mass[R.argsort()]
        acut = np.where(rpristar_ord <= rcut)
        ncut = len(acut[0])         
        if ncut > 0:
            rgalcut = rpristar_ord[acut]
            mgalcut = mstar_ord[acut]
            mcut = np.sum(mgalcut)

            massb50 = 0.50*mcut
            maux = 0
            t = 0
            while t<ncut:
                if maux<massb50:
                    maux = maux + mgalcut[t]
                    rhm_star=rgalcut[t]
                t = t+1         
        ################### inside rhm ##################


        aropt = np.where(data["stars"]["R"] <= rhm_star)    
        narop = len(aropt[0])
        if narop > 0:
            mstar_hm = np.sum(data["stars"]["mass"][aropt])
        else:
            mstar_hm = 0      

        aropt = np.where(data["dark_matter"]["R"] <= rhm_star)    
        narop = len(aropt[0])
        if narop > 0:
            mdm_hm = np.sum(data["dark_matter"]["mass"][aropt])
        else:
            mdm_hm = 0    

        ########### append ########
        #mstar_hm_arr.append(mstar_hm)
        #mgas_hm_arr.append(0)
        #mdm_hm_arr.append(mdm_hm)                      

        # Ropt with all baryons
        rgal = np.array([])
        mgal = np.array([])
        for data_type in ['stars']:
            R = data[data_type]["R"]
            mass = data[data_type]["mass"]
            rgal = np.append(rgal, R)
            mgal = np.append(mgal, mass)

        # Ropt
        rgal_ord = rgal[rgal.argsort()]
        mgal_ord = mgal[rgal.argsort()]

        acut = np.where(rgal_ord <= rcut)
        ncut = len(acut[0]) 
        if ncut > 0:
            rgalcut = rgal_ord[acut]
            mgalcut = mgal_ord[acut]
            mcut = np.sum(mgalcut)

            massb50 = 0.83*mcut
            maux = 0
            t = 0
            while t<ncut:
                if maux<massb50:
                    maux = maux + mgalcut[t]
                    ropt=rgalcut[t]

                t = t+1                
        ################### inside ropt ##################
        aropt = np.where(R <= ropt)    
        nsrop = len(aropt[0])
        if nsrop > 0:
            mstar_ropt = np.sum(mass[aropt])
        else:
            mstar_ropt = 0                 

        ################################
        # limit to 1.5 times the optical radius
        mask = R < 2.0 * rhm_star 

        position = position[mask]
        velocity = velocity[mask]
        mass = mass[mask].reshape(-1, 1)
        # compute angular momentum

        specific_angular_momentum = np.cross(position, velocity)
        angular_momentum = mass * specific_angular_momentum
        total_angular_momentum = angular_momentum.sum(axis=0)
        total_specific_angular_momentum = specific_angular_momentum.sum(axis=0)
        L_dir = total_specific_angular_momentum / np.linalg.norm(total_specific_angular_momentum)

        # compute rotation matrix to align reference frame with the angular momentum

        rot_matrix = get_rotation_matrix(total_specific_angular_momentum)
        factor_kms_MpcYr=1.02201e-12

        for data_type in data_types:

            rot_position = rot_matrix.dot(data[data_type]["position"].T).T
            rot_vel = rot_matrix.dot(data[data_type]["velocity"].T).T

            r, theta, phi, v_r, v_t, v_p= cartesian_to_spherical(rot_position[:,0], rot_position[:,1], rot_position[:,2], rot_vel[:,0], rot_vel[:,1], rot_vel[:,2])


            data[data_type]["v_radial"] = v_r
            data[data_type]["M_infall"] = data[data_type]["v_radial"]*data[data_type]["mass"]*factor_kms_MpcYr    

            R = data[data_type]["R"]
            mgal = data[data_type]["mass"]

            Rcum, cumsum = mass_profile(R, mgal)
            data[data_type]["Rcum"] = Rcum
            data[data_type]["Mass_Int"] = cumsum        

        rhm_star_arr.append(rhm_star)
        rhm_coldgas_arr.append(0)
        Mstar_opt_arr.append(mstar_ropt)

   

        
        ######### Matter stack ##########
        
        
        r_M_star= data["stars"]["Rcum"]
        M_star= data["stars"]["Mass_Int"]                  
        rmean, mstar_hm = profile_r_mass(r_M_star[r_M_star<15],np.ones(len(M_star[r_M_star<15])) ,M_star[r_M_star<15])
        mstar_stack_arr.append(mstar_hm)  

        r_M_dm= data["dark_matter"]["Rcum"]
        M_dm = data["dark_matter"]["Mass_Int"]                  

        rmean, mdm_hm = profile_r_mass(r_M_dm[r_M_dm<15],np.ones(len(M_dm[r_M_dm<15])) ,M_dm[r_M_dm<15])
        mdm_stack_arr.append(mdm_hm)  
        
        mgas_stack_arr.append(np.zeros(200))
        
        #Density Cold Gas:
        
        
        
        #################################
        radial_coldden_arr.append(np.full(200, 0))

        #Density SFR Cold Gas:
        radial_gasSFR_arr.append(np.full(200, 0))

        #Sup density SFR Cold Gas:

        sup_gassfr.append(np.full(200, 0))        

        #Tempeture Cold Gas:

        radial_T_arr.append(np.full(200, np.nan))

        #Metallicity Cold Gas 

        radial_Z_arr.append(np.full(200, 0))    

        #Fraction of cold gas

        radial_fcold_arr.append(np.full(200, 0))        

        #Vrot Gas

        radial_vr_arr.append(np.full(200, np.nan))    

        #Macc Cold Gas:

        radial_macc_arr.append(np.full(200, 0))     
        
        #Mout Cold Gas:

        radial_mout_arr.append(np.full(200, 0)) 
        ################### Stellar ####################
        data_type = "stars"                                                                           

        condition = (data[data_type]["R2d"]< 15.) 
        R = data[data_type]["R2d"][condition]   
        mgal = data[data_type]["mass"][condition] 

        rbin_sup, rho_sup, mbin_tot = calc_densidad(R[R.argsort()], mgal[R.argsort()])
        sup_stellar.append(rho_sup)        

        ################### Sup SFR ####################
        condition_SFR = (data[data_type]["R2d"]< 15.) & (data[data_type]["age"]*1e9 < 1e7) 

        R = data[data_type]["R2d"][condition_SFR]
        mgal = data[data_type]["mass"][condition_SFR]

        rbin_sup, rho_sup, mbin_tot = calc_densidad(R[R.argsort()], mgal[R.argsort()])
        sup_sfr.append(rho_sup)        

        ################### SFR ####################
        condition_SFR = (data[data_type]["R"]< 15.) & (data[data_type]["age"]*1e9 < 1e7) 

        R = data[data_type]["R"][condition_SFR]
        mgal = data[data_type]["mass"][condition_SFR]

        rbin_sup, rho_sup, mbin_tot = calc_densidad3d(R[R.argsort()], mgal[R.argsort()])        
        sfr_arr.append(rho_sup)  

    return Mstar_opt_arr,rhm_star_arr,rhm_coldgas_arr,radial_coldden_arr,radial_gasSFR_arr,radial_T_arr,radial_Z_arr,radial_fcold_arr,radial_vr_arr,radial_macc_arr,radial_mout_arr,sup_stellar,sup_gassfr,sup_sfr,sfr_arr,mstar_stack_arr,mgas_stack_arr,mdm_stack_arr





############ For cooling tables, Integrated Mass, Weighted Radial profiles ################


########################### start the loop over groups ###########################
def Tabs_coolin_all(j,box,rvir_select,MyRedshift,run_interested,FlatLambdaCDM):
    #####################################
    run=run_interested
    ####################################
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    Ob0=0.05
    mp=1.67262192e-24
    factor = 1/mp   
    To=1e4

    ####################################
    print("z",MyRedshift[j])   

    MyExpFactor = 1 / (1 + MyRedshift[j])
    MyExpFactor = np.around(MyExpFactor, 3)

    mp=1.67262192e-24
    factor = 1/mp
    rho_mean_z = (cosmo.critical_density(MyRedshift[j]).value*Ob0*factor)#* density_units.to("Msun/pc3").value

    
    if (box=='Box40Mpc'):
        
        path ="/rack/iSATA1/danielcv/FirstLight/Outputs_FirstLight_Massive/FL"+str(run)+"/"
        text_files = [f for f in os.listdir(path) if f.endswith("{:0.3f}".format(MyExpFactor)+'.dat')]
        cutout = text_files[0][len("FL"+str(run)+"_S"):len("FL"+str(run)+"_")+4]      
        
        snap_info = {"basedir": "",
            "simulation": "/rack/iSATA1/danielcv/FirstLight/Outputs_FirstLight_Massive/FL"+str(run)+"/FL"+str(run),
            "cutout_size": int(cutout),
            "expansion_factor": MyExpFactor,} 

    else:
        path ="/media/FirstLight/FL"+str(run)+"/"
        text_files = [f for f in os.listdir(path) if f.endswith("{:0.3f}".format(MyExpFactor)+'.dat')]
        cutout = text_files[0][len("FL"+str(run)+"_S"):len("FL"+str(run)+"_")+4]  
        
        snap_info = {"basedir": "",
            "simulation": "/media/FirstLight/FL"+str(run)+"/FL"+str(run),
            "cutout_size": int(cutout),
            "expansion_factor": MyExpFactor,}   
    

    virial_radius = snap_info["cutout_size"] / 4
    
    try:

        data_types = ["stars", "gas",'dark_matter']

        data = {}
        for data_type in data_types:
            data[data_type] = load_data(data_type=data_type, **snap_info)

        xcm0,ycm0,zcm0,vxcm0,vycm0,vzcm0 = shrinking_sphere(data["stars"]["position"],data["stars"]["velocity"],data["stars"]["mass"])

        data["stars"]["position"]=data["stars"]["position"]-[xcm0,ycm0,zcm0]
        data["stars"]["velocity"]=data["stars"]["velocity"]-[vxcm0,vycm0,vzcm0]
        data["gas"]["position"]=data["gas"]["position"]-[xcm0,ycm0,zcm0]
        data["gas"]["velocity"]=data["gas"]["velocity"]-[vxcm0,vycm0,vzcm0]
        data["dark_matter"]["position"]=data["dark_matter"]["position"]-[xcm0,ycm0,zcm0]
        data["dark_matter"]["velocity"]=data["dark_matter"]["velocity"]-[vxcm0,vycm0,vzcm0]

        # compute radial coordinate from center
        for data_type in data_types:
            #print(data_type, data[data_type].keys())
            pos = data[data_type]["position"]
            data[data_type]["R"] = np.linalg.norm(pos, axis=1)
            data[data_type]["R2d"] = np.linalg.norm(pos[:, [1, 2]], axis=1)

        # compute the mass for gas cells

        sel_gas = data["gas"]

        # not sure if this is correct
        density_units =  1.66e-24 * units.g * units.cm**-3
        density = sel_gas["density"] * density_units.to("Msun/pc3").value
        sel_gas["mass"] = sel_gas["cell_size"]**3 * density   


        Z_sun= 0.0129


        Den=data["gas"]["density"]
        Metal=(data["gas"]["metals_mass_fraction_SNII"]+data["gas"]["metals_mass_fraction_ZSNIa"])/Z_sun
        Temp=data["gas"]["temperature"]
        mgal_weight=data["gas"]["mass"]
        vol_weight=data["gas"]["cell_size"]**3
        R=data["gas"]["R"] 

        yarr=[Den,Temp,Metal]

        Redshift_arr=np.full(50,MyRedshift[j])
        Rvir=np.full(50,rvir_select[j])

        # normal median ############
        rmean, radial_gasden = profile_r_tab(R ,np.ones(len(mgal_weight)), yarr[0],rmin=1e-3,rmax=rvir_select[j])
        rmean, radial_T = profile_r_tab(R ,np.ones(len(mgal_weight)), yarr[1],rmin=1e-3,rmax=rvir_select[j])
        rmean, radial_Z = profile_r_tab(R ,np.ones(len(mgal_weight)), yarr[2],rmin=1e-3,rmax=rvir_select[j])

        # mass-weight median ############
        rmean, radial_gasden_mass = profile_r_tab(R ,mgal_weight, yarr[0]*mgal_weight,rmin=1e-3,rmax=rvir_select[j])
        rmean, radial_T_mass = profile_r_tab(R ,mgal_weight, yarr[1]*mgal_weight,rmin=1e-3,rmax=rvir_select[j])
        rmean, radial_Z_mass = profile_r_tab(R ,mgal_weight, yarr[2]*mgal_weight,rmin=1e-3,rmax=rvir_select[j])

        ## vol-weight median ############
        rmean, radial_gasden_vol = profile_r_tab(R ,vol_weight, yarr[0]*vol_weight,rmin=1e-3,rmax=rvir_select[j])
        rmean, radial_T_vol = profile_r_tab(R ,vol_weight, yarr[1]*vol_weight,rmin=1e-3,rmax=rvir_select[j])
        rmean, radial_Z_vol = profile_r_tab(R ,vol_weight, yarr[2]*vol_weight,rmin=1e-3,rmax=rvir_select[j])

        ############ dm binned ################
        mgal_weight=data["gas"]["mass"]
        vol_weight=data["gas"]["cell_size"]**3
        R=data["gas"]["R"] 

        r_total = np.concatenate((data['dark_matter']["R"] , data["stars"]["R"] , data["gas"]["R"] ), axis=None)
        m_total = np.concatenate((data['dark_matter']["mass"], data['stars']["mass"], data['gas']["mass"]), axis=None)

        r_total_ord = r_total[r_total.argsort()]  
        m_total_ord = m_total[r_total.argsort()]                  

        rtot_bin = np.logspace(np.log10(1e-3),np.log(rvir_select[j]),51)        
        mtot_bin, edges, binnu = binned_statistic(r_total_ord, m_total_ord, bins = rtot_bin, statistic = np.sum) 
        r_e =(edges[:-1]+edges[1:])/2

        data = {
            'T': radial_T[~np.isnan(rmean)],
            'Den': radial_gasden[~np.isnan(rmean)],
            'Z': radial_Z[~np.isnan(rmean)],
            'T_mass': radial_T_mass[~np.isnan(rmean)],
            'Den_mass': radial_gasden_mass[~np.isnan(rmean)],
            'Z_mass': radial_Z_mass[~np.isnan(rmean)],
            'T_vol': radial_T_vol[~np.isnan(rmean)],
            'Den_vol': radial_gasden_vol[~np.isnan(rmean)],
            'Z_vol': radial_Z_vol[~np.isnan(rmean)],
            'Mtot':mtot_bin[~np.isnan(rmean)],
            'radius':rmean[~np.isnan(rmean)],
            'redshift': Redshift_arr[~np.isnan(rmean)],    
            'rvir':Rvir[~np.isnan(rmean)]
        }

        # Crear un DataFrame
        df = pd.DataFrame(data)
        base_cool = "/home/cataldi/output/cooling/"

        # Guardar el DataFrame como un archivo CSV
        df.to_csv(base_cool+'Cooling_Tabs_z_'+str(np.round(MyRedshift[j],2))+'_h'+str(run_interested)+'.csv', index=False)   
    except FileNotFoundError:
        data_types = ["stars",'dark_matter']

        data = {}
        for data_type in data_types:
            data[data_type] = load_data(data_type=data_type, **snap_info)

        xcm0,ycm0,zcm0,vxcm0,vycm0,vzcm0 = shrinking_sphere(data["stars"]["position"],data["stars"]["velocity"],data["stars"]["mass"])

        data["stars"]["position"]=data["stars"]["position"]-[xcm0,ycm0,zcm0]
        data["stars"]["velocity"]=data["stars"]["velocity"]-[vxcm0,vycm0,vzcm0]
        data["dark_matter"]["position"]=data["dark_matter"]["position"]-[xcm0,ycm0,zcm0]
        data["dark_matter"]["velocity"]=data["dark_matter"]["velocity"]-[vxcm0,vycm0,vzcm0]

        # compute radial coordinate from center
        for data_type in data_types:
            #print(data_type, data[data_type].keys())
            pos = data[data_type]["position"]
            data[data_type]["R"] = np.linalg.norm(pos, axis=1)
            data[data_type]["R2d"] = np.linalg.norm(pos[:, [1, 2]], axis=1)


        ############ dm binned ################
        Redshift_arr=np.full(50,MyRedshift[j])
        Rvir=np.full(50,rvir_select[j])

        r_total = np.concatenate((data['dark_matter']["R"] , data["stars"]["R"]), axis=None)
        m_total = np.concatenate((data['dark_matter']["mass"], data['stars']["mass"]), axis=None)

        r_total_ord = r_total[r_total.argsort()]  
        m_total_ord = m_total[r_total.argsort()]                  

        rtot_bin = np.logspace(np.log(1e-3),np.log(rvir_select[j]),51)        
        mtot_bin, edges, binnu = binned_statistic(r_total_ord, m_total_ord, bins = rtot_bin, statistic = np.sum) 
        r_e =(edges[:-1]+edges[1:])/2


        data = {
            'T': np.full(50,np.nan),
            'Den': np.full(50,0.),
            'Z': np.full(50,0.),
            'T_mass': np.full(50,np.nan),
            'Den_mass': np.full(50,0.),
            'Z_mass': np.full(50,0.),
            'T_vol': np.full(50,0.),
            'Den_vol': np.full(50,0.),
            'Z_vol': np.full(50,0.),
            'Mtot':mtot_bin[~np.isnan(r_e)],
            'radius':r_e[~np.isnan(r_e)],
            'redshift': Redshift_arr[~np.isnan(r_e)],    
            'rvir':Rvir[~np.isnan(r_e)]
        }

        # Crear un DataFrame
        df = pd.DataFrame(data)
        base_cool = "/home/cataldi/output/cooling/"

        # Guardar el DataFrame como un archivo CSV
        df.to_csv(base_cool+'Cooling_Tabs_z_'+str(np.round(MyRedshift[j],2))+'_h'+str(run_interested)+'.csv', index=False)   


############## Generate Maps ###############################

def FL_all_dynamics_Maps(j,box,MyRedshift,run_interested,FlatLambdaCDM,Mstar_opt_arr,rhm_star_arr,rhm_coldgas_arr,maps_den_arr,maps_coldden_arr,maps_T_arr,maps_Z_arr,maps_fcold_arr,maps_vr_arr,maps_macc_arr,vol_size=5.):
    #####################################
    run=run_interested
    #box=boxarr
    
    print(run_interested)


    ####################################
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    Ob0=0.05
    mp=1.67262192e-24
    factor = 1/mp   
    To=1e4

    ####################################
    #for j in range(len(MyRedshift)):
    print("z",MyRedshift[j])   

    MyExpFactor = 1 / (1 + MyRedshift[j])
    MyExpFactor = np.around(MyExpFactor, 3)

    mp=1.67262192e-24
    factor = 1/mp
    rho_mean_z = (cosmo.critical_density(MyRedshift[j]).value*Ob0*factor)#* density_units.to("Msun/pc3").value

    
    if (box=='Box40Mpc'):
        
        path ="/scratch/danielcv/FirstLight/Outputs_FirstLight_Massive/FL"+str(run)+"/"
        text_files = [f for f in os.listdir(path) if f.endswith("{:0.3f}".format(MyExpFactor)+'.dat')]
        cutout = text_files[0][len("FL"+str(run)+"_S"):len("FL"+str(run)+"_")+4]      
        
        snap_info = {"basedir": "",
            "simulation": "/scratch/danielcv/FirstLight/Outputs_FirstLight_Massive/FL"+str(run)+"/FL"+str(run),
            "cutout_size": int(cutout),
            "expansion_factor": MyExpFactor,} 

    else:
        path ="/media/FirstLight/FL"+str(run)+"/"
        text_files = [f for f in os.listdir(path) if f.endswith("{:0.3f}".format(MyExpFactor)+'.dat')]
        cutout = text_files[0][len("FL"+str(run)+"_S"):len("FL"+str(run)+"_")+4]  
        
        snap_info = {"basedir": "",
            "simulation": "/media/FirstLight/FL"+str(run)+"/FL"+str(run),
            "cutout_size": int(cutout),
            "expansion_factor": MyExpFactor,}   
    

    virial_radius = snap_info["cutout_size"] / 4
    data_types = ["stars", "gas"]

    data = {}
    for data_type in data_types:
        data[data_type] = load_data(data_type=data_type, **snap_info)

    xcm0,ycm0,zcm0,vxcm0,vycm0,vzcm0 = shrinking_sphere(data["stars"]["position"],data["stars"]["velocity"],data["stars"]["mass"])

    data["stars"]["position"]=data["stars"]["position"]-[xcm0,ycm0,zcm0]
    data["stars"]["velocity"]=data["stars"]["velocity"]-[vxcm0,vycm0,vzcm0]
    data["gas"]["position"]=data["gas"]["position"]-[xcm0,ycm0,zcm0]
    data["gas"]["velocity"]=data["gas"]["velocity"]-[vxcm0,vycm0,vzcm0]


    # compute radial coordinate from center
    for data_type in data_types:
        #print(data_type, data[data_type].keys())
        pos = data[data_type]["position"]
        data[data_type]["R"] = np.linalg.norm(pos, axis=1)

    # compute the mass for gas cells

    sel_gas = data["gas"]

    # not sure if this is correct
    density_units =  1.66e-24 * units.g * units.cm**-3
    density = sel_gas["density"] * density_units.to("Msun/pc3").value

    sel_gas["mass"] = sel_gas["cell_size"]**3 * density   

    data_type = "stars"
    position = data[data_type]["position"]
    velocity = data[data_type]["velocity"]
    mass = data[data_type]["mass"]
    R = data[data_type]["R"]
    rcut = 0.15*virial_radius


    # Half Mass Stellar radius
    rpristar_ord = R[R.argsort()]
    mstar_ord = mass[R.argsort()]
    acut = np.where(rpristar_ord <= rcut)
    ncut = len(acut[0])         
    if ncut > 0:
        rgalcut = rpristar_ord[acut]
        mgalcut = mstar_ord[acut]
        mcut = np.sum(mgalcut)

        massb50 = 0.50*mcut
        maux = 0
        t = 0
        while t<ncut:
            if maux<massb50:
                maux = maux + mgalcut[t]
                rhm_star=rgalcut[t]
            t = t+1         
                      

    # Ropt with all baryons
    rgal = np.array([])
    mgal = np.array([])
    for data_type in ['gas', 'stars']:
        R = data[data_type]["R"]
        mass = data[data_type]["mass"]
        rgal = np.append(rgal, R)
        mgal = np.append(mgal, mass)
        
    # Ropt
    rgal_ord = rgal[rgal.argsort()]
    mgal_ord = mgal[rgal.argsort()]

    acut = np.where(rgal_ord <= rcut)
    ncut = len(acut[0]) 
    if ncut > 0:
        rgalcut = rgal_ord[acut]
        mgalcut = mgal_ord[acut]
        mcut = np.sum(mgalcut)

        massb50 = 0.83*mcut
        maux = 0
        t = 0
        while t<ncut:
            if maux<massb50:
                maux = maux + mgalcut[t]
                ropt=rgalcut[t]

            t = t+1                
                
        
    ################### inside ropt ##################
    aropt = np.where(R <= ropt)    
    nsrop = len(aropt[0])
    if nsrop > 0:
        mstar_ropt = np.sum(mass[aropt])
    else:
        mstar_ropt = 0                 

    ################################
    # limit to 1.5 times the optical radius
    mask = R < 2.0 * rhm_star 

    position = position[mask]
    velocity = velocity[mask]
    mass = mass[mask].reshape(-1, 1)
    # compute angular momentum

    specific_angular_momentum = np.cross(position, velocity)
    angular_momentum = mass * specific_angular_momentum
    total_angular_momentum = angular_momentum.sum(axis=0)
    total_specific_angular_momentum = specific_angular_momentum.sum(axis=0)
    L_dir = total_specific_angular_momentum / np.linalg.norm(total_specific_angular_momentum)

    # compute rotation matrix to align reference frame with the angular momentum

    rot_matrix = get_rotation_matrix(total_specific_angular_momentum)
    factor_kms_MpcYr=1.02201e-12

    for data_type in data_types:

        rot_position = rot_matrix.dot(data[data_type]["position"].T).T
        rot_vel = rot_matrix.dot(data[data_type]["velocity"].T).T

        r, theta, phi, v_r, v_t, v_p= cartesian_to_spherical(rot_position[:,0], rot_position[:,1], rot_position[:,2], rot_vel[:,0], rot_vel[:,1], rot_vel[:,2])


        data[data_type]["v_radial"] = v_r
        data[data_type]["M_infall"] = data[data_type]["v_radial"]*data[data_type]["mass"]*factor_kms_MpcYr    

        R = data[data_type]["R"]
        mgal = data[data_type]["mass"]

        Rcum, cumsum = mass_profile(R, mgal)
        data[data_type]["Rcum"] = Rcum
        data[data_type]["Mass_Int"] = cumsum        

    ################################ norm ###################
    data_type = "gas"
            
    Temp_all = data[data_type]["temperature"]
    den_all = data[data_type]["density"]
    cond_g_dencold = ((Temp_all< 1e4)&(den_all>1.))
    
    mass_cold = data[data_type]["mass"][cond_g_dencold]
    R_cold = data[data_type]["R"][cond_g_dencold]
    
    # Half Gas-Cold radius
    rprigas_ord = R_cold[R_cold.argsort()]
    mgas_ord = mass_cold[R_cold.argsort()]
    acut = np.where(rprigas_ord <= rcut)
    ncut = len(acut[0])         
    if ncut > 0:
        rgalcut = rprigas_ord[acut]
        mgalcut = mgas_ord[acut]
        mcut = np.sum(mgalcut)

        massb50 = 0.50*mcut
        maux = 0
        t = 0
        while t<ncut:
            if maux<massb50:
                maux = maux + mgalcut[t]
                rhm_coldgas=rgalcut[t]
            t = t+1         
    
    rhm_star_arr.append(rhm_star)
    rhm_coldgas_arr.append(rhm_coldgas)
    Mstar_opt_arr.append(mstar_ropt)
    
    rot_position = rot_matrix.dot(data[data_type]["position"].T).T

    condition = np.where((np.abs(rot_position[:, 0]) < vol_size) & ((np.abs(rot_position[:, 1]) < vol_size) & (np.abs(rot_position[:, 2]) < vol_size)))[0]

    data_type_s = "stars"
    rot_position_s = rot_matrix.dot(data[data_type_s]["position"].T).T

    condition_s = np.where((np.abs(rot_position_s) < vol_size) & ((np.abs(rot_position_s) < vol_size) & (np.abs(rot_position_s) < vol_size)))[0]


    #third_value = [data[data_type]["density"][condition]/rho_mean_z]
    third_value = [data[data_type]["density"][condition]/rho_mean_z,data[data_type]["temperature"][condition],data[data_type]['metals_mass_fraction_SNII'][condition],
                   data[data_type]["v_radial"][condition],data[data_type]["M_infall"][condition]*1e6]
        
    mass = data[data_type]["mass"][condition]
    first_second_value = [rot_position[:, 0][condition],rot_position[:, 1][condition]]
    first_third_value = [rot_position[:, 0][condition],rot_position[:, 2][condition]]
    
    Temp = data[data_type]["temperature"][condition]
    cond_g_cold = (Temp< 1e4)
    cond_g_hot = (Temp>1e4)

    ######## rhm_star ###############

    x = first_third_value[0]
    y = first_third_value[1] 
    ######## Density Gas #############

    weight = third_value[0]*mass

    hist = np.histogram2d(x, y, weights=weight, bins=128)
    histM = np.histogram2d(x, y, weights=mass, bins=128)
    img = np.log10(hist[0].T/histM[0].T)   
    maps_den_arr.append(img)
    
    ######## Density SFR Gas #############
    x = first_third_value[0]
    y = first_third_value[1] 
    cond_g_dencold = ((Temp< 1e4)&(third_value[0]*rho_mean_z>1.))

    weight = third_value[0]*mass

    hist = np.histogram2d(x[cond_g_dencold], y[cond_g_dencold], weights=weight[cond_g_dencold], bins=128)
    histM = np.histogram2d(x[cond_g_dencold], y[cond_g_dencold], weights=mass[cond_g_dencold], bins=128)
    img = np.log10(hist[0].T/histM[0].T)   
    maps_coldden_arr.append(img)    
    
    
    ######## Tempeture Gas #############
    weight = third_value[1]*mass

    hist = np.histogram2d(x, y, weights=weight, bins=128)
    histM = np.histogram2d(x, y, weights=mass, bins=128)
    img = np.log10(hist[0].T/histM[0].T)    
    maps_T_arr.append(img)
        
    ######## Metallicity Gas #############

    weight = third_value[2]*mass

    hist = np.histogram2d(x, y, weights=weight, bins=128)
    histM = np.histogram2d(x, y, weights=mass, bins=128)
    img = np.log10(hist[0].T/histM[0].T)    
    maps_Z_arr.append(img)    
    
    ######## Fcold Gas #############

    weight = third_value[1]*mass

    histM = np.histogram2d(x, y, weights=mass, bins=256)
    extent = [histM[1][0], histM[1][-1], histM[2][0], histM[2][-1]]
    hist = np.histogram2d(x[cond_g_cold], y[cond_g_cold], weights=mass[cond_g_cold], bins=(histM[1],histM[2]))
    img = hist[0].T/histM[0].T   
    maps_fcold_arr.append(img)        
    
    ######## Vrot Gas #############

    weight = third_value[3]*mass
    
    hist = np.histogram2d(x, y, weights=weight, bins=256)
    histM = np.histogram2d(x, y, weights=mass, bins=256)
    img = hist[0].T/histM[0].T
    
    maps_vr_arr.append(img)    
    
    ######## Macc vrot Gas #############

    weight = third_value[4]*mass
    
    histDen,xedges, yedges = np.histogram2d(x, y, bins=256, density=True)
    volume_bin = (np.diff(xedges).reshape(-1, 1) * np.diff(yedges).reshape(1, -1))*vol_size
    hist = np.histogram2d(x, y, weights=weight, bins=256)
    histM = np.histogram2d(x, y, weights=mass, bins=256)            
    img = hist[0].T/(histM[0].T*volume_bin)
    
    maps_macc_arr.append(img)   
    
    


    return Mstar_opt_arr,rhm_star_arr,rhm_coldgas_arr,maps_den_arr,maps_coldden_arr,maps_T_arr,maps_Z_arr,maps_fcold_arr,maps_vr_arr,maps_macc_arr








