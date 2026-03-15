from .parameters import Params
import numpy as np
from .constants import *
from math import floor
from math import exp
from math import sin
from math import cos
from math import log

def sanity_checks(params: Params):
    if params.p_dir_isotropic_ctrl + params.p_dir_uniform_ctrl != 1:
        raise ValueError(
            'Only one initial positron movement direction scenario is allowed!')
    if params.phistorytoplot >= params.track_p_history_count:
        raise ValueError(
            'This positron cant plotted its history isnt tracked!')

def positronprop_initialize(params: Params):

    #total positroncount:
    positroncount = (params.Al26_src1_ctrl*params.Al_positroncount + \
                     params.SagA_src_ctrl*params.SagA_positroncount)
                     #add here...

    ###Set up MW cartesian coords------------------------------------

    xcornercount = params.xexpintervals*2 + 1 #grid cell corner count
    ycornercount = params.yexpintervals*2 + 1
    zcornercount = params.zexpintervals*2 + 1
    cellcount = (xcornercount-1)*(ycornercount-1)*(zcornercount-1)

    xexpfactor = np.linspace(params.xexpfactor_min,params.xexpfactor_max,params.xexpintervals)
    yexpfactor = np.linspace(params.yexpfactor_min,params.yexpfactor_max,params.yexpintervals)
    zexpfactor = np.linspace(params.zexpfactor_min,params.zexpfactor_max,params.zexpintervals)
    cellcorner_pc_x = np.empty(xcornercount)
    cellcorner_pc_y = np.empty(ycornercount)
    cellcorner_pc_z = np.empty(zcornercount)
    sidelength_pc_x = np.empty(xcornercount-1) #prepare cell sidelength arrays
    sidelength_pc_y = np.empty(ycornercount-1)
    sidelength_pc_z = np.empty(zcornercount-1)
    cellcenter_pc_x = np.empty(xcornercount-1) #prepare cell center coord arrays
    cellcenter_pc_y = np.empty(ycornercount-1)
    cellcenter_pc_z = np.empty(zcornercount-1)
    vol_array = np.empty((xcornercount-1,ycornercount-1,zcornercount-1)) #prepare volume-building arrays
    vol_array_pc = np.empty((xcornercount-1,ycornercount-1,zcornercount-1)) 
    vol_array_m = np.empty((xcornercount-1,ycornercount-1,zcornercount-1)) 
    cylradmag_pc = np.empty((xcornercount-1,ycornercount-1)) #cyl rad from MW center, pc
    cylradmag_m = np.empty((xcornercount-1,ycornercount-1))
    sphradmag_pc = np.empty((xcornercount-1,ycornercount-1,zcornercount-1)) #sph rad from MW center, pc
    sphradmag_m = np.empty((xcornercount-1,ycornercount-1,zcornercount-1)) #...in m

    centercount = np.zeros(dim)
    centercount[0] = xcornercount - 1
    centercount[1] = ycornercount - 1
    centercount[2] = zcornercount - 1

    ###Sanity checks made possible by centercount defns:

    if params.SagA_x_customspawncoordindex >= centercount[0]:
        raise ValueError('SagA e+ Would spawn outside the simulation region in x!')
    if params.SagA_y_customspawncoordindex >= centercount[1]:
        raise ValueError('SagA e+ Would spawn outside the simulation region in y!')
    if params.SagA_z_customspawncoordindex >= centercount[2]:
        raise ValueError('SagA e+ Would spawn outside the simulation region in z!')

    #grid cell corner coords
    def build_cellcorner(expintervals,cellcorner,expbase,\
                         expfactor,translation,cornercount):
        for i in range(0,expintervals): #python halts the for loop before the "halt" index computes
            cellcorner[i] = -expbase**(expfactor[expintervals - (i+1)]) + translation
            #^fill bottom half of coords
            cellcorner[expintervals + i + 1] = expbase**(expfactor[i]) + translation
            #skip the middle w/ "i+1" fill top half of z's
        cellcorner[round(floor(cornercount/2))] = 0.0 + translation
        #manually fill middle z value
        return cellcorner

    #grid cell sidelengths
    def build_sidelength(cellcorner,sidelength,cornercount):
        for i in range(0,cornercount-1):
            #-1 for #volumes = #boundaries - 1
            sidelength[i] = cellcorner[i+1]-cellcorner[i]
        return sidelength

    #grid cell center coords
    def build_cellcenter(cellcorner,cellcenter,sidelength,cornercount):
        for i in range(0,cornercount-1):
            cellcenter[i] = cellcorner[i]+sidelength[i]/2.0
        return cellcenter

    #grid cell volumes
    def cartesian_vol_build(sidelength_x,sidelength_y,sidelength_z,vol):
        for i in range(0,xcornercount-1):
            for j in range(0,ycornercount-1):
                for k in range(0,zcornercount-1):
                    #-1 for #vol = #boundaries - 1
                    vol[i][j][k] = sidelength_x[i]*sidelength_y[j]*sidelength_z[k]
        return vol

    #cylindrical radius
    def build_cylradmag(x,y):
        cylradmag = np.sqrt(x**2.0 + y**2.0)
        return cylradmag

    #spherical radius
    def build_sphradmag(x,y,z):
        return np.sqrt(x**2.0 + y**2.0 + z**2.0)

    #Build cell corner coord arrays:
    cellcorner_pc_x = build_cellcorner(params.xexpintervals,cellcorner_pc_x,\
                                       params.xexpbase,xexpfactor,params.coord_translation,xcornercount)
    cellcorner_pc_y = build_cellcorner(params.yexpintervals,cellcorner_pc_y,\
                                       params.yexpbase,yexpfactor,params.coord_translation,ycornercount)
    cellcorner_pc_z = build_cellcorner(params.zexpintervals,cellcorner_pc_z,\
                                       params.zexpbase,zexpfactor,params.coord_translation,zcornercount)
    cellcorner_m_x = m_in_pc*cellcorner_pc_x
    cellcorner_m_y = m_in_pc*cellcorner_pc_y
    cellcorner_m_z = m_in_pc*cellcorner_pc_z

    #build cell sidelength arrays:
    #Sidelength[i] = Length between x[i+1] and x[i] same for y and z. Same for vol.
    sidelength_pc_x = build_sidelength(cellcorner_pc_x,sidelength_pc_x,xcornercount)
    sidelength_pc_y = build_sidelength(cellcorner_pc_y,sidelength_pc_y,ycornercount)
    sidelength_pc_z = build_sidelength(cellcorner_pc_z,sidelength_pc_z,zcornercount)
    sidelength_m_x = sidelength_pc_x*m_in_pc
    sidelength_m_y = sidelength_pc_y*m_in_pc
    sidelength_m_z = sidelength_pc_z*m_in_pc

    #build cell center coord arrays:
    cellcenter_pc_x = build_cellcenter(cellcorner_pc_x,cellcenter_pc_x,\
                                       sidelength_pc_x,xcornercount)
    cellcenter_pc_y = build_cellcenter(cellcorner_pc_y,cellcenter_pc_y,\
                                       sidelength_pc_y,ycornercount)
    cellcenter_pc_z = build_cellcenter(cellcorner_pc_z,cellcenter_pc_z,\
                                       sidelength_pc_z,zcornercount)
    cellcenter_m_x = m_in_pc*cellcenter_pc_x
    cellcenter_m_y = m_in_pc*cellcenter_pc_y
    cellcenter_m_z = m_in_pc*cellcenter_pc_z

    #build cell volumes array pc^3
    vol_array_pc = cartesian_vol_build(sidelength_pc_x,sidelength_pc_y,sidelength_pc_z,vol_array)
    vol_array_m = cartesian_vol_build(sidelength_m_x,sidelength_m_y,\
                                      sidelength_m_z,vol_array)

    #build cylindrical radial magnitude from MW center for each cellcenter
    for i in range(xcornercount-1):
        for j in range(ycornercount-1):
            cylradmag_pc[i][j] = build_cylradmag(cellcenter_pc_x[i],cellcenter_pc_y[j])
            cylradmag_m[i][j] = build_cylradmag(cellcenter_m_x[i],cellcenter_m_y[j])

    #compute distance of each cellcenter from MW center
    for i in range(xcornercount-1):
        for j in range(ycornercount-1):
            for k in range(zcornercount-1):
                sphradmag_pc[i][j][k] = build_sphradmag(cellcenter_pc_x[i],\
                                                        cellcenter_pc_y[j],cellcenter_pc_z[k])
                sphradmag_m[i][j][k] = build_sphradmag(cellcenter_m_x[i],\
                                                       cellcenter_m_y[j],cellcenter_m_z[k])
    # return (cellcorner_pc_x,cellcorner_pc_y,cellcorner_pc_z,
    #     cellcorner_m_x,cellcorner_m_y,cellcorner_m_z,
    #     sidelength_pc_x,sidelength_pc_y,sidelength_pc_z,
    #     sidelength_m_x,sidelength_m_y,sidelength_m_z,
    #     cellcenter_pc_x,cellcenter_pc_y,cellcenter_pc_z,
    #     cellcenter_m_x,cellcenter_m_y,cellcenter_m_z,
    #     vol_array_pc,vol_array_m,
    #     cylradmag_pc,cylradmag_m,
    #     sphradmag_m,sphradmag_pc) #fix this when breaking stuff up

###B field------------------------------------

    Bdip = np.empty((xcornercount,ycornercount,zcornercount,dim))
    #add here...

    Btot = np.empty((xcornercount,ycornercount,zcornercount,dim))
    Btot_norm = np.empty((xcornercount,ycornercount,zcornercount,dim))
    Btot_abs = np.empty((xcornercount,ycornercount,zcornercount))
    Btot_dir = np.empty((xcornercount,ycornercount,zcornercount,dim))

    #uniform field:
    B_uniform = np.zeros(dim) #initialize array for uniform external magnetic pseudovector field...
    #...in n dimensions

    if params.B_uniform_ctrl == 1:
        B_uniform[0]=params.b_uniform_x
        #x component of uniform external pseudovector field, T
        B_uniform[1]=params.b_uniform_y
        B_uniform[2]=params.b_uniform_z
    #if B_uniform_ctrl == 0, the uniform B field will remain an array of zero's.

    #dipole field:
    if params.B_dipole_ctrl == 1:
        #dipole B magnitude:
        def build_magdipmom(x,y,z,dipx,dipy,dipz):
            return np.sqrt(dipx*x**2.0 + dipy*y**2.0 + dipz*z**2.0)

        #build dipole B field:
        def build_B_dipole():
            for i in range(xcornercount-1):
                for j in range(ycornercount-1):
                    for k in range(zcornercount-1):
                        mag_dipmom = build_magdipmom(\
                                                     cellcenter_m_x[i],cellcenter_m_y[j],\
                                                     cellcenter_m_z[k],params.b_dipole_x,params.b_dipole_y,params.b_dipole_z)
                        #dipole moment magnitude
                        Bdip[i][j][k][0] = ((3.0*cellcenter_m_x[i]*mag_dipmom)/\
                                            (cylradmag_m[i][j]**5.0))-(params.b_dipole_x/(cylradmag_m[i][j]**3.0))
                        Bdip[i][j][k][1] = ((3.0*cellcenter_m_y[j]*mag_dipmom)/\
                                            (cylradmag_m[i][j]**5.0))-(params.b_dipole_y/(cylradmag_m[i][j]**3.0))
                        Bdip[i][j][k][2] = ((3.0*cellcenter_m_z[k]*mag_dipmom)/\
                                            (cylradmag_m[i][j]**5.0))-(params.b_dipole_z/(cylradmag_m[i][j]**3.0))
                        #xyz components of dipole pseudovector field at coord ijk, Teslas
            return Bdip
        Bdip = build_B_dipole()

    #other contribution shapes:    



    #X-shaped field:



    #Static random B: (simulates large-spacetime turbulent modes)



    #Dynamic random B: (simulates small-spacetime turbulent modes)

    #this will force Btot to update every prop step.

    #add here...

    #total B field------------------

    def build_B_total():
        for i in range(xcornercount-1):
            for j in range(ycornercount-1):
                for k in range(zcornercount-1):
                    for m in range(dim):
                        Btot[i][j][k][m] = Bdip[i][j][k][m] + B_uniform[m] #+... add here...
                        #total B pseudovector field, T
                    Btot_abs[i][j][k] = np.sqrt(Btot[i][j][k][0]**2.0 + Btot[i][j][k][1]**2.0 + \
                                                Btot[i][j][k][2]**2.0 )
                    for m in range(dim):
                        Btot_dir[i][j][k][m] = Btot[i][j][k][m]/Btot_abs[i][j][k]
                        #normalize Btot to get unit pseudovector components
        return Btot,Btot_abs,Btot_dir
    Btot,Btot_abs,Btot_dir = build_B_total()

###MW ISM------------------------------------

    ISM_temp = np.empty((xcornercount-1,ycornercount-1,zcornercount-1))
    ISM_Hndens = np.empty((xcornercount-1,ycornercount-1,zcornercount-1))
    ISM_ionfrac = np.empty((xcornercount-1,ycornercount-1,zcornercount-1))
    #empty instead of zero because these values are mutually exclusive, rather than additive.

    if params.ISMcond_test1_ctrl == 1:
        #if first ISM config option is chosen:
        def build_ISM_1(temp,Hndens,ionfrac):
            for i in range(xcornercount-1):
                for j in range(ycornercount-1):
                    for k in range(zcornercount-1):
                        if sphradmag_pc[i][j][k] < 100: #CMZ radius, pc
                            temp[i][j][k] = params.MM_Temp
                            Hndens[i][j][k] = params.MM_Hndens
                            ionfrac[i][j][k] = params.MM_IonizFrac
                        elif sphradmag_pc[i][j][k] < 2000: #Outer bulge rad, pc
                            temp[i][j][k] = params.WNM_Temp
                            Hndens[i][j][k] = params.WNM_Hndens
                            ionfrac[i][j][k] = params.WNM_IonizFrac
                        elif np.abs(cellcenter_pc_z[k]) < 100: #Disk height, pc
                            if cylradmag_pc[i][j] < 13400: #MW radius, pc
                                temp[i][j][k] = params.MM_Temp
                                Hndens[i][j][k] = params.MM_Hndens
                                ionfrac[i][j][k] = params.MM_IonizFrac
                            else:
                                temp[i][j][k] = params.WIM_Temp
                                Hndens[i][j][k] = params.WIM_Hndens
                                ionfrac[i][j][k] = params.WIM_IonizFrac
                        else: #Halo
                            temp[i][j][k] = params.WIM_Temp
                            Hndens[i][j][k] = params.WIM_Hndens
                            ionfrac[i][j][k] = params.WIM_IonizFrac
            return temp,Hndens,ionfrac
        ISM_temp,ISM_Hndens,ISM_ionfrac = build_ISM_1(ISM_temp,ISM_Hndens,ISM_ionfrac)

    #if second ISM config option is chosen:

    #add here...

###MW 26Al mass dist------------------------------------

    Alrho_array = np.zeros((xcornercount-1,ycornercount-1,zcornercount-1)) #prepare 26Al mass dens array
    Almass_array = np.zeros((xcornercount-1,ycornercount-1,zcornercount-1)) #prepare 26Al mass array
    #other densities and masses...

    #if 26Al is selected as an active e+ source:
    if params.Al26_src1_ctrl == 1:
        #26Al mass dens spatial dist functional form:
        def build_Al_Mdens_1(r,z,rho_origin):
            return rho_origin*np.exp(-np.abs(z)/params.z0)*np.exp(-r/params.r0) #galactic 26Al mass density function

        #build 26 Al mass density array:
        def build_rho_Al(x,y,z,Alrho_array): #define function to build 26Al mass density profile
            for i in range(xcornercount-1): #define how to populate each 26Almass dens array value
                for j in range(ycornercount-1):
                    if cylradmag_pc[i][j] > params.bulgediskboundary_pc:
                        for k in range(zcornercount-1):
                           Alrho_array[i][j][k] = build_Al_Mdens_1(cylradmag_pc[i][j],z[k],params.Alrho0)
                    else:
                        for k in range(zcornercount-1):
                            Alrho_array[i][j][k] = 0.0
            return Alrho_array
        Alrho_array = build_rho_Al(cellcenter_pc_x,cellcenter_pc_y,cellcenter_pc_z,Alrho_array)
        #construct 26Al mass density array
        #This mass array is exponential in r outside the bulge and zero inside.
        Almass_array = np.multiply(Alrho_array,vol_array_pc) #Hadamard product, aka element-wise mult.
        #26Al mass in each grid cell

    #other 26Al mass distributions: add here...

    #other diffuse src mass distributions: add here...

    Altotalmass = np.sum(Almass_array) #milky way 26Al total mass, kg. needs to = 3 solar masses.
    Almasslist = np.reshape(Almass_array,cellcount) #flatten to list
    Almassfraclist = Almasslist/Altotalmass #list of mass fraction in each cell

    #other e+ src mass dist totals and fractions: add here...

###26Al e+ coords, weighted by Almass dist------------------

    runningindex_coord = 0 #Use this to control which p_coord entries are inherited from which srcs
    runningindex_coordindex = 0

    #def src-types
    Al_p_coordindex = np.empty((params.Al_positroncount,dim)) #prepare 26Al e+' initial xyz coordinate indexes
    Al_p_coord = np.empty((params.Al_positroncount,dim)) #Al26 e+ coords array
    SagA_p_coordindex = np.empty((params.SagA_positroncount,dim))
    SagA_p_coord = np.empty((params.SagA_positroncount,dim)) #prepare Sagittarius A e+ coords
    #add here...

    #def overall arrays
    p_coordindex = np.empty((positroncount,dim)) #prepare e+' initial xyz coordinate indexes
    p_coord = np.empty((positroncount,dim)) #e+ coords array

    #list-making function...
    def createList(start,stop):
        return list(range(start,stop))

    # Initialize rz volume chunk index list w/ dummy vars r1 and r2
    start1,stop1 = 0,cellcount 
    xyzindexlist = createList(start1,stop1) #make list of volume cell indexes

    #initial e+ x,y,z grid indexes (compute which coord boxes the e+'s' start in):
    def build_p_coordindexes(positroncount,p_coordindex,\
                            p_xyz_initialcoordindex):
        for i in range(positroncount): #for each positron...
            p_coordindex[i][0] = floor(p_xyz_initialcoordindex[i]/\
                                          ((zcornercount-1)*(ycornercount-1)))
            p_coordindex[i][1] = floor(p_xyz_initialcoordindex[i]/(zcornercount-1)) -\
                                    p_coordindex[i][0]*(ycornercount-1)
            p_coordindex[i][2] = p_xyz_initialcoordindex[i] -\
                                    p_coordindex[i][0]*(ycornercount-1)*(zcornercount-1) -\
                                    p_coordindex[i][1]*(zcornercount-1)
        #^weaponize knowledge of x,y,z coord array shape to get which row then col each index points to.
        #array shape: x sets of y sets of z. see Galactic 26Al mass dist section.
        return p_coordindex

    #now get the actual r,z coords from the above indexes..
    #take coords at indexed boxes and randomize within each box.
    def build_p_coords(positroncount,p_coordindex,p_coord):
        for i in range(positroncount): #for i'th positron, get initial coordinates, m
            p_coord[i][0] = cellcenter_m_x[round(p_coordindex[i][0])] \
                - sidelength_m_x[round(p_coordindex[i][0])]/2.0 \
                + np.random.random_sample(size=1)*(sidelength_m_x[round(p_coordindex[i][0])])
            p_coord[i][1] = cellcenter_m_y[round(p_coordindex[i][1])] \
                - sidelength_m_y[round(p_coordindex[i][1])]/2.0 \
                + np.random.random_sample(size=1)*(sidelength_m_y[round(p_coordindex[i][1])])
            p_coord[i][2] = cellcenter_m_z[round(p_coordindex[i][2])] \
                - sidelength_m_z[round(p_coordindex[i][2])]/2.0 \
                + np.random.random_sample(size=1)*(sidelength_m_z[round(p_coordindex[i][2])])
                #^start at cell center where the positron is, subtract cell sidelength/2,
                    #add random amt of sidelength back
                #^for some reason the indexes are floats. gotta convert to ints, hence the round().
                #the random displacement is uniform on each cell value interval.
        return p_coord

    if params.Al26_src1_ctrl == 1:
        #randomly select which cell each 26Al positron spawns in:
        Al_p_xyz_initialcoordindex = np.random.choice(xyzindexlist,params.Al_positroncount,p=Almassfraclist)
        #np.rand.choice(rand var values,count,probabilities)
        #^the values of p matter those of rzindexlist don't.
        #set positrons' grid indexes and coords:
        Al_p_coordindex = build_p_coordindexes(params.Al_positroncount,Al_p_coordindex,\
                                               Al_p_xyz_initialcoordindex)
        Al_p_coord = build_p_coords(params.Al_positroncount,Al_p_coordindex,Al_p_coord)
        
        #pass to propagator input arrays:
        p_coord[runningindex_coord:round(runningindex_coord + params.Al_positroncount)][:] = Al_p_coord
        #add to overall e+ coord index
        runningindex_coord = runningindex_coord + params.Al_positroncount
        #tell next srcs to populate later p_coord...
        #...array values
        p_coordindex[runningindex_coordindex:round\
                     (runningindex_coordindex + params.Al_positroncount)][:] = Al_p_coordindex
        runningindex_coordindex = runningindex_coordindex + params.Al_positroncount

    if params.SagA_MWcenter == 1: #If it's been set to the MW center,
        SagA_spawn_xcoordindex = round(centercount[0]/2.0)
        SagA_spawn_ycoordindex = round(centercount[1]/2.0)
        SagA_spawn_zcoordindex = round(centercount[2]/2.0)

    if params.SagA_arbitrary == 1: #If it's been set to an arbitrary location,
        SagA_spawn_xcoordindex = params.SagA_x_customspawncoordindex
        SagA_spawn_ycoordindex = params.SagA_y_customspawncoordindex
        SagA_spawn_zcoordindex = params.SagA_z_customspawncoordindex
        
        #all e+ spawn at the same location:
        SagA_p_coord[:,0] = np.full(params.SagA_positroncount,cellcenter_m_x[round(SagA_spawn_xcoordindex)])
        SagA_p_coord[:,1] = np.full(params.SagA_positroncount,cellcenter_m_y[round(SagA_spawn_ycoordindex)])
        SagA_p_coord[:,2] = np.full(params.SagA_positroncount,cellcenter_m_z[round(SagA_spawn_zcoordindex)])
        SagA_p_coordindex[:,0] = np.full(params.SagA_positroncount,round(SagA_spawn_xcoordindex))
        SagA_p_coordindex[:,1] = np.full(params.SagA_positroncount,round(SagA_spawn_ycoordindex))
        SagA_p_coordindex[:,2] = np.full(params.SagA_positroncount,round(SagA_spawn_zcoordindex))
        
        #pass to propagator input arrays:
        p_coord[runningindex_coord:round(runningindex_coord + params.SagA_positroncount)][:] = SagA_p_coord
        runningindex_coord = runningindex_coord + params.SagA_positroncount
        p_coordindex[runningindex_coordindex:round\
                     (runningindex_coordindex + params.SagA_positroncount)][:] = SagA_p_coordindex
        runningindex_coordindex = runningindex_coordindex + params.SagA_positroncount
        
    #add more e+ sources to p_coord...

###Positrons' initial isotropic rand movement directions------------------------------------

    p_dir = np.empty((positroncount,dim)) #e+ direction unit vector

    #we assume e+ are emitted with isotropic rand movement directions from all sources.

    ###Isotropic random positron movement directions------------------

    if params.p_dir_isotropic_ctrl == 1:
        alt = np.linspace(altmin,altmax,params.altintervals) #elevation angles
        #for isotropic velocity dist, need to make sure we don't get clusters at the poles:
        latweight = np.empty(params.altintervals)
        normedlatweight = np.empty(params.altintervals)
        p_dir_alt = np.empty(positroncount) #e+ initial direction elevations
        p_dir_azi = np.empty(positroncount) #e+ initial direction azimuths
        p_dir = np.empty((positroncount,dim)) #e+ direction unit vector

        #for isotropic velocity dist, need to make sure we don't get clusters at the poles:
        for i in range(params.altintervals):
            latweight[i] = np.cos(alt[i])
        normedlatweight = latweight/np.sum(latweight) #make probabilities sum to 1
        p_dir_alt = np.random.choice(alt,positroncount,p=normedlatweight) \
            + np.random.random_sample(size = positroncount)
        #rand e+ initial altitudes: first pick altitude degree w/ lat weight,
        #then rand uniformly within that degree

        p_dir_azi = np.random.random_sample(size = positroncount)*2.0*pi
        #e+ direction prob dist is isotropic in azimuth

        for i in range(positroncount):
            p_dir[i][0] = np.sin(p_dir_alt[i])*np.cos(p_dir_azi[i])
            p_dir[i][1] = np.sin(p_dir_alt[i])*np.sin(p_dir_azi[i])
            p_dir[i][2] = np.cos(p_dir_alt[i])
        #get initial direction unit vector components, cartesian

    ###Give all e+ the same direction------------------

    if params.p_dir_uniform_ctrl == 1:
        p_dir_norm = np.sqrt(params.p_dir_uniform_mag_x*2.0 
            + params.p_dir_uniform_mag_y*2.0 
            + params.p_dir_uniform_mag_z*2.0)
        for i in range(positroncount):
            p_dir[i][0] = params.p_dir_uniform_mag_x/p_dir_norm
            p_dir[i][1] = params.p_dir_uniform_mag_y/p_dir_norm
            p_dir[i][2] = params.p_dir_uniform_mag_z/p_dir_norm

###26Al Positrons' initial KEs------------------------------------

    runningindex_KE = 0 #for overall propagator input array construction
    runningindex_gamma = 0
    runningindex_beta = 0
    runningindex_absvel = 0
    runningindex_gammabin = 0
    runningindex_gammabinindex = 0

    #overall propagator inputs
    p_gammabin_index = np.empty(positroncount)
    p_gammabin = np.empty(positroncount)
    p_gamma = np.empty(positroncount)
    p_beta = np.empty(positroncount)
    p_absvel = np.empty(positroncount)
    p_KE = np.empty(positroncount)

    #gamma will be specifically constructed for each e+ src-type.
    def p_beta_absvel_KE_build(gamma):
        beta = np.sqrt(1.0 - 1.0/(gamma**2.0))
        absvel = c*beta
        KE = gamma*mp_eV*cp
        return beta,absvel,KE

    #TODO: add different src-types' gammamaxes, gamma int sizes, gammavals, betavals
    #somehow track which situation we're in when computing propagation

    Al_gammamax = (Al_Emax/mp_eV) + 1.0 #Maximum allowed gamma value for 26Al beta e+
    #corresponds with epsilon_0 in Milne's KE dist: "gamma | no KE goes to ve-"
    Al_gamma_interval_size = (Al_gammamax - 1.0)/(params.gamma_bin_count - 1.0)
    #uniform gamma bin intervals: bincount - 1 because the top bin value is the max allowed value
    Al_gammavals = np.linspace(1.0,Al_gammamax,params.gamma_bin_count)
        #gamma_bin_count entries long, not that many intervals
        #central values of each bin
    Al_betavals = np.sqrt(1.0-1.0/(Al_gammavals**2.0))
        #for ioniz exc en loss rate lookup table

    #generalize KE intervals somehow beyond 26Al...

    #add here...

    if params.Al26_src1_ctrl == 1:
        
        Al_p_gammabin = np.empty(params.Al_positroncount)
        Al_p_gamma = np.empty(params.Al_positroncount)
        Al_p_beta = np.empty(params.Al_positroncount)
        Al_p_absvel = np.empty(params.Al_positroncount)
        Al_p_KE = np.empty(params.Al_positroncount)
        Al_N_gamma = np.zeros(params.gamma_bin_count)
        Al_p_gammabin_indexes = np.linspace(0.0,params.gamma_bin_count-1,params.gamma_bin_count)
        #make ordered list of gamma bin barrier indexes
        
        #Al KE spectrum structure------------------
        
        Al_const1 = (2.0*pi/hbar)*(g**2.0)*(abs(M_if)**2.0)*(((4.0*pi)**2.0)/(h**6.0))*(mp**5.0)*(c**4.0)
        Al_const2 = -2.0*pi*Al_Z*alpha

        def columndepth(p_gamma): #ksi = column depth
            return Al_const2/((1.0-(p_gamma**(-2.0)))**(1.0/2.0))

        def KE_Coulomb_Correc(p_gamma): #Coulomb correction to KE spectrum
            return 2.0*pi*columndepth(p_gamma)/(1.0-exp(-columndepth(p_gamma)))

        #Define Al beta decay positron KE spectrum function:
        def transition_prob(p_gamma): #the above but positive
            return Al_const1*p_gamma*((Al_gammamax-p_gamma)**2.0\
                                     )*((p_gamma**2.0 - 1.0)**(1.0/2.0))*KE_Coulomb_Correc(p_gamma)

        #We want to avoid querying the KE dist fn too often when calculating N(gamma) for each e+.
        #Instead, let's directly compute a smaller set of N(gamma_int)s and make those into bins.
        #N(gamma) is an unnormalized function amplitude.
        for i in range(1,params.gamma_bin_count):
            Al_N_gamma[i] = transition_prob(Al_gammavals[i]) #distribution amplitude
        Al_N_gamma[0] = 0.0 #manually repair where gamma_pdf value = 0
        Al_N_gamma_pdf = Al_N_gamma/np.sum(Al_N_gamma) #turn N(gamma) vals into list of probabilities
        #normalizes N, so we no longer care about KE distrib. fn amplitude.
        
        #26Al e+ propagator input variables------------------

        Al_p_gammabin_index = np.random.choice(Al_p_gammabin_indexes,params.Al_positroncount,p=Al_N_gamma_pdf)
        #...choice(possible values,count,prob dist)
        #Roll which energy bin index each 26Al e+ falls into Randomly draw from the index list.

        for i in range(params.Al_positroncount):
            Al_p_gammabin[i] = Al_gammavals[round(Al_p_gammabin_index[i])]
        #Which energy bin each e+ falls into Randomly draw from the bin barrier index list:
        Al_p_gamma = Al_p_gammabin + (\
                                      np.random.sample(size=params.Al_positroncount) - 0.5\
                                     )*Al_gamma_interval_size
        #Each e+'s Lorentz factor
        #Add symmetric random interval fraction to each e+'s KE bin barrier value.
        #This is fine to do because the max and min values will never be populated,
        #...per our Al_N_gamma_pdf's values at gamma = 0 and = gammamax,...
        #...and so our e+ KEs will fall outside the allowed KE range with numerically zero probability.
        #The above assumes uniform probability in each bin.
        #Gamma here is the beta decay KE NOT sent to the neutrino.

        Al_p_beta,Al_p_absvel,Al_p_KE = p_beta_absvel_KE_build(Al_p_gamma)
        #e+ vel in terms of beta (v/c)
        #|v| for each e+, m/s
        #Rest-mass frame(s') total energy for each e+, J
        #We assume each e+ is independent, so we can boost our POV of each e+ ...
        #...to its respective rest-mass frame w/o changing the physics description of our situation.
        
        #add to overall arrays and update overall array structure tempvars------------------
        
        p_gammabin_index[runningindex_gammabinindex:round\
                         (runningindex_gammabinindex + params.Al_positroncount)][:] = Al_p_gammabin_index
        runningindex_gammabinindex = runningindex_gammabinindex + params.Al_positroncount
        
        p_gammabin[runningindex_gammabin:round\
                   (runningindex_gammabin + params.Al_positroncount)][:] = Al_p_gammabin
        runningindex_gammabin = runningindex_gammabin + params.Al_positroncount
        
        p_gamma[runningindex_gamma:round(runningindex_gamma + params.Al_positroncount)][:] = Al_p_gamma
        runningindex_gamma = runningindex_gamma + params.Al_positroncount
        
        p_beta[runningindex_beta:round(runningindex_beta + params.Al_positroncount)][:] = Al_p_beta
        runningindex_beta = runningindex_beta + params.Al_positroncount
        
        p_absvel[runningindex_absvel:round(runningindex_absvel + params.Al_positroncount)][:] = Al_p_absvel
        runningindex_absvel = runningindex_absvel + params.Al_positroncount
        
        p_KE[runningindex_KE:round(runningindex_KE + params.Al_positroncount)][:] = Al_p_KE
        runningindex_KE = runningindex_KE + params.Al_positroncount

    if params.SagA_src_ctrl == 1:

        SagA_p_gammabin = np.empty(params.SagA_positroncount)
        SagA_p_gamma = np.empty(params.SagA_positroncount)
        SagA_p_beta = np.empty(params.SagA_positroncount)
        SagA_p_absvel = np.empty(params.SagA_positroncount)
        SagA_p_KE = np.empty(params.SagA_positroncount)

        #dummy bin index value to get downstream code to behave. CHANGE THIS to a power law.
        SagA_p_gammabin_index = np.full(params.SagA_positroncount,60)
        
        for i in range(params.SagA_positroncount):
            SagA_p_gammabin[i] = Al_gammavals[round(SagA_p_gammabin_index[i])]
        SagA_p_gamma = SagA_p_gammabin# + (\
    #                                   np.random.sample(size=SagA_positroncount) - 0.5\
    #                                  )*Al_gamma_interval_size
        #Each e+'s Lorentz factor uniformly dist within single Al_gamma interval,...
        #...or if rest of line is disabled, is equal to the same value.
            
        SagA_p_beta,SagA_p_absvel,SagA_p_KE = p_beta_absvel_KE_build(SagA_p_gamma)

        #add to overall arrays and update overall array structure tempvars
        p_gammabin_index[runningindex_gammabinindex:round\
                         (runningindex_gammabinindex + params.SagA_positroncount)][:] = SagA_p_gammabin_index
        runningindex_gammabinindex = runningindex_gammabinindex + params.SagA_positroncount
        
        p_gammabin[runningindex_gammabin:round\
                   (runningindex_gammabin + params.SagA_positroncount)][:] = SagA_p_gammabin
        runningindex_gammabin = runningindex_gammabin + params.SagA_positroncount
        
        p_gamma[runningindex_gamma:round(runningindex_gamma + params.SagA_positroncount)][:] = SagA_p_gamma
        runningindex_gamma = runningindex_gamma + params.SagA_positroncount
        
        p_beta[runningindex_beta:round(runningindex_beta + params.SagA_positroncount)][:] = SagA_p_beta
        runningindex_beta = runningindex_beta + params.SagA_positroncount
        
        p_absvel[runningindex_absvel:round(runningindex_absvel + params.SagA_positroncount)][:] = SagA_p_absvel
        runningindex_absvel = runningindex_absvel + params.SagA_positroncount
        
        p_KE[runningindex_KE:round(runningindex_KE + params.SagA_positroncount)][:] = SagA_p_KE
        runningindex_KE = runningindex_KE + params.SagA_positroncount

###Build other initial positron quantities------------------

    p_pitchangle = np.empty(positroncount) #prepare angle between e+ direction and B pseudovector
    p_cyclotronrad = np.empty(positroncount) #prepare cyclotron radius, m
    p_pitch = np.empty(positroncount) #prepare pitch, m
    p_dispperpitch = np.empty(positroncount) #prepare displacement per helix loop, m
    p_cyclotronperiod = np.empty(positroncount) #prepare cyclotron period
    p_LorentzForce = np.empty((positroncount,dim)) #prepare Lorentz force
    p_time = np.zeros(positroncount) #set initial time of each positron
    EnLossPerColDens_IonizExc = np.empty(params.gamma_bin_count-1) #exclude zeroth gammaval
    p_KELossPerPathlength_IonizExc = np.empty(positroncount)
    p_KELossPerPathlength_Plasma = np.empty(positroncount)


    def build_pitchangle(p_xdir,p_ydir,p_zdir,p_xindex,p_yindex,p_zindex): #pitchangle function
        return np.arccos((p_xdir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][0] \
                          + p_ydir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][1] \
                          + p_zdir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][2])\
                         /(np.sqrt(p_xdir**2.0 + p_ydir**2.0 + p_zdir**2.0)\
                           * (np.sqrt(Btot[round(p_xindex)][round(p_yindex)]\
                                      [round(p_zindex)][0]**2.0 \
                                      + Btot[round(p_xindex)][round(p_yindex)]\
                                      [round(p_zindex)][1]**2.0 \
                                      + Btot[round(p_xindex)][round(p_yindex)]\
                                      [round(p_zindex)][2]**2.0))))
        #^ = arccos((a dot b)/(|a||b|)) for a = initial direction and b = Btot direction
        #^in grid box where e+ exists radians

    def build_cyclotronrad(p_gamma,p_absvel,p_pitchangle,p_xindex,p_yindex,p_zindex):
        return p_gamma*(mp*p_absvel*np.sin(p_pitchangle)/\
                        (cp*Btot_abs[round(p_xindex)][round(p_yindex)][round(p_zindex)]))

    def build_pitch(cyclotronrad,pitchangle): #displacement per helix loop || B, m
        return 2.0*pi*cyclotronrad/np.tan(pitchangle)

    def build_dispperpitch(cyclotronrad,pitch): #physical pathlength, m
        return np.sqrt((2.0*pi*cyclotronrad)**2.0 + pitch**2.0)

    def build_cyclotronperiod(p_cyclotronrad,p_absvel,p_pitchangle): #seconds
        return 2.0*pi*p_cyclotronrad/(p_absvel*np.sin(p_pitchangle))

    def build_Lorentz(p_absvel,p_xdir,p_ydir,p_zdir,p_xindex,p_yindex,p_zindex):
        LorentzForce = np.empty(dim) #prepare local dummy array
        LorentzForce[0] = cp*(p_absvel*p_ydir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][2] \
                              - p_absvel*p_zdir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][1])
        LorentzForce[1] = cp*(p_absvel*p_zdir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][0] \
                              - p_absvel*p_xdir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][2])
        LorentzForce[2] = cp*(p_absvel*p_xdir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][1] \
                              - p_absvel*p_ydir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][0])
        # =q(v x B): yz-zy, 
        return LorentzForce

    #procedure------------------

    for i in range(positroncount): #relativistic cyclotron frequency and radius Lorentz force
        p_pitchangle[i] = build_pitchangle(p_dir[i][0],p_dir[i][1],p_dir[i][2],\
                                           p_coordindex[i][0],p_coordindex[i][1],\
                                           p_coordindex[i][2])
        p_cyclotronrad[i] = build_cyclotronrad(p_gamma[i],p_absvel[i],p_pitchangle[i],\
                                               p_coordindex[i][0],p_coordindex[i][1],\
                                               p_coordindex[i][2])
        p_pitch[i] = build_pitch(p_cyclotronrad[i],p_pitchangle[i])
        p_dispperpitch[i] = build_dispperpitch(p_cyclotronrad[i],p_pitch[i])
        p_cyclotronperiod[i] = build_cyclotronperiod(p_cyclotronrad[i],p_absvel[i],\
                                                        p_pitchangle[i])
        p_LorentzForce[i,:] = build_Lorentz(p_absvel[i],
                                               p_dir[i][0],p_dir[i][1],p_dir[i][2],
                                               p_coordindex[i][0],p_coordindex[i][1],\
                                               p_coordindex[i][2])
        #compute initial pitch angle, cyclotron radius, pitch, displacement per pitch, \
        #period, Lorentz force for all 26Al e+ SI
        #use Lorentz force to compute exact motion as set of affine (shearing) symplectic fns

###En loss rates------------------------------------

    I_avg = 9.1*params.Z_avg*(1.0 + 1.9/(params.Z_avg**(2/3)))
    #Apprx avg ionization potential for all species, eV
    EnLossRateConst1 = (4.0*pi*mp_eV*cp*(positronQrad**2.0)/(params.atomicweight*kgPerAMU))
    EnLossRateConst2 = I_avg/(mp_eV*cp) #*cp puts you in Joules instead of eV
    EnLossRateConst3 = (1.0/2.0)*log(2.0)
    EnLossRateConst4 = 23.0/2.0

    def build_EnLossPerColDens_IonizExc(beta,gamma):
        return ((EnLossRateConst1*params.Z_bound_H/(beta**2))*(log(np.sqrt(gamma-1.0)*gamma*beta/\
                                                          EnLossRateConst2) + EnLossRateConst3 \
                                                      - ((beta**2.0)/12.0)*\
                                                      (EnLossRateConst4 + 7.0/(gamma + 1.0) \
                                                       + 5.0/((gamma + 1.0)**2.0) \
                                                       + 2.0/((gamma + 1.0)**3.0))))
        #^Fn for energy loss per unit col dens (J/col dens = J/kg*m^2), ionization and atomic excitation

    def build_EnlossPerPathlength_Plasma(ISM_ionfrac,ISM_ndens,gamma,beta,n_e):
        return (ISM_ndens*(EnLossRateConst1*ISM_ionfrac/(beta**2.0))*\
                (log((gamma*beta**2.0)/(56.0*np.sqrt(n_e)*eVperJ/(mp_eV*cp))) + \
                log(2.0) - (1.0/2.0)*beta**2.0))
        #^energy loss per unit interaction depth, plasma wave excitations
        #^units of J/area dens (kg/m^2), then mult by gas mass dens (H for now) to get J/m

    #procedure------------------

    n_e = params.Z_bound_H*ISM_Hndens #assume n = nH for now, change to ISM_ndens later e- #dens array

    for i in range(1,params.gamma_bin_count-1):
        EnLossPerColDens_IonizExc[i] = build_EnLossPerColDens_IonizExc(Al_betavals[i],Al_gammavals[i])
        #compute energy loss per unit col dens (J/col dens = J/kg*m^2), ionization and atomic excitation
        #LOOKUP TABLE, as fn of energy valid on linearly partitioned energy intervals.
        #we assume the ioniz+exc en loss rate doesn't change much for small changes in e+ KE.

    #compute initial total en loss per unit interaction depth, J/m:
    for i in range(positroncount):
        p_KELossPerPathlength_IonizExc[i] = (protonmass_kg*ISM_Hndens\
                                                [round(p_coordindex[i][0])]\
                                                [round(p_coordindex[i][1])]\
                                                [round(p_coordindex[i][2])]*\
                                                EnLossPerColDens_IonizExc\
                                                [round(p_gammabin_index[i])])
        # = atomicmass*ndens*enloss/coldens (where each e+ is and at each e+'s energy)
        # = mdens*enloss/coldens = (kg/m^3)*J/(kg/m^2) = J/m
        #aka, compute initial ionization+excitation energy loss per int length for all e+, J/m.
        #only H for now.
        p_KELossPerPathlength_Plasma[i] = (protonmass_kg*build_EnlossPerPathlength_Plasma\
                                              (ISM_ionfrac[round(p_coordindex[i][0])]\
                                               [round(p_coordindex[i][1])]\
                                               [round(p_coordindex[i][2])],\
                                               ISM_Hndens[round(p_coordindex[i][0])]\
                                               [round(p_coordindex[i][1])]\
                                               [round(p_coordindex[i][2])],\
                                               p_gamma[i],p_beta[i],\
                                               n_e[round(p_coordindex[i][0])]\
                                               [round(p_coordindex[i][1])]\
                                               [round(p_coordindex[i][2])]))
        #compute initial plasma wave energy loss per int length for all e+, J/m
    p_KELossPerPathlength = p_KELossPerPathlength_IonizExc + p_KELossPerPathlength_Plasma

###Decay heuristics------------------------------------

    FreeAnnihCrossSecConst1 = pi*(positronQrad**2.0)
    FreeAnnihCrossSecConst2 = FreeAnnihCrossSecConst1*2.0*pi
    #constants for free annihilation cross sections
    #cuts down on number of operations per function call.

    FreeAnnihCrossSec_HEn = np.zeros(params.gamma_bin_count)
    FreeAnnihCrossSec_LEn = np.zeros(params.gamma_bin_count)

    def build_FreeAnnihCrossSec_HEn(gamma,gamma_bin_count):
        prevgamma0 = gamma[0] #store previous value of 0'th gamma, 1, in tempvar
        gamma[0] = 1.001 #prepare sensible cross-section for lowest gamma this won't matter.
        for i in range(gamma_bin_count):
            FreeAnnihCrossSec_HEn[i] = (FreeAnnihCrossSecConst1*(1.0/gamma[i])*\
                                        (((gamma[i]**2.0 + 4.0*gamma[i] + 1.0)/(gamma[i]**2.0 - 1.0))*\
                                         log(gamma[i] + np.sqrt(gamma[i]**2.0 - 1.0)) - \
                                         ((gamma[i] + 3.0)/np.sqrt(gamma[i]**2.0 - 1.0))))
            #2.62 milne, cross sec high en free annih 17 prantzos.
        gamma[0] = prevgamma0 #repair original data. very ugly/lazy/stupid way to avoid a NaN in...
        #...exactly one place.
        return FreeAnnihCrossSec_HEn

    def build_FreeAnnihCrossSec_LEn(beta,gamma_bin_count):
        prevbeta0 = beta[0] #store prev 0th beta value, 0, in tempvar
        beta[0] = 0.001 #prepare sensible cross-section for lowest beta this will matter.
        for i in range(gamma_bin_count):
            FreeAnnihCrossSec_LEn[i] = (FreeAnnihCrossSecConst2*(alpha/(beta[i]**2.0))\
                                        /(1.0 - exp(-2.0*pi*alpha/beta[i])))
            #2.63, ^low en, free annih gould '89
        beta[0] = prevbeta0 #repair original data. very ugly/lazy/stupid way to do this.
        return FreeAnnihCrossSec_LEn

    def build_FreeAnnihProb(p_KE_start,p_KE_stop,e_ndens,p_vel,FreeAnnihCrossSec,enlossrate):
        decayprob = 1.0 # - 
        return

    #add here...

    #tabulate free high-energy annih cross sec at each gammabin value not all will be used.
    FreeAnnihCrossSec_HEn = build_FreeAnnihCrossSec_HEn(Al_gammavals,params.gamma_bin_count)
    #plot these against KE
    #todo: generalize to multiple e+ sources

    #tabulate free low-energy annih cross sec at each gammabin value not all will be used.
    FreeAnnihCrossSec_LEn = build_FreeAnnihCrossSec_LEn(Al_betavals,params.gamma_bin_count)

###Propagator------------------------------------


    return (positroncount,
        xcornercount,ycornercount,zcornercount,
        cellcount,
        #xexpfactor,yexpfactor,zexpfactor,
        #cellcorner_pc_x,cellcorner_pc_y,cellcorner_pc_z,
        cellcorner_m_x,cellcorner_m_y,cellcorner_m_z,
        #sidelength_pc_x,sidelength_pc_y,sidelength_pc_z,
        sidelength_m_x,sidelength_m_y,sidelength_m_z,
        cellcenter_m_x,cellcenter_m_y,cellcenter_m_z,
        #cellcenter_pc_x,cellcenter_pc_y,cellcenter_pc_z,
        vol_array_m,#vol_array_pc,
        cylradmag_m,sphradmag_m,#cylradmag_pc,sphradmag_pc,
        centercount,
        Btot,Btot_norm,Btot_abs,Btot_dir,#B_uniform,Bdip,
        ISM_temp,ISM_Hndens,ISM_ionfrac,
        Alrho_array,Almass_array,Altotalmass,Almasslist,Almassfraclist,
        p_coordindex,p_coord,p_dir,
        p_gammabin_index,p_gammabin,p_gamma,p_beta,p_absvel,p_KE,
        Al_gammamax,Al_gamma_interval_size,Al_gammavals,Al_betavals,
        p_pitchangle,p_cyclotronrad,p_pitch,p_dispperpitch,p_cyclotronperiod,
        p_LorentzForce,p_time,EnLossRateConst1,
        EnLossPerColDens_IonizExc,p_KELossPerPathlength_IonizExc,p_KELossPerPathlength_Plasma,
        n_e,p_KELossPerPathlength,
        FreeAnnihCrossSec_HEn,FreeAnnihCrossSec_LEn)

def get_trajectories(params: Params):
    #call simulation code here and pass vars to graphics engine    

    #Run sanity checks on input ctrls------------------------------------
    sanity_checks(params)

    #Initialize simulation

    (positroncount,
        xcornercount,ycornercount,zcornercount,
        cellcount,
        #xexpfactor,yexpfactor,zexpfactor,
        #cellcorner_pc_x,cellcorner_pc_y,cellcorner_pc_z,
        cellcorner_m_x,cellcorner_m_y,cellcorner_m_z,
        #sidelength_pc_x,sidelength_pc_y,sidelength_pc_z,
        sidelength_m_x,sidelength_m_y,sidelength_m_z,
        cellcenter_m_x,cellcenter_m_y,cellcenter_m_z,
        #cellcenter_pc_x,cellcenter_pc_y,cellcenter_pc_z,
        vol_array_m,#vol_array_pc,
        cylradmag_m,sphradmag_m,#cylradmag_pc,sphradmag_pc,
        centercount,
        Btot,Btot_norm,Btot_abs,Btot_dir,#B_uniform,Bdip,
        ISM_temp,ISM_Hndens,ISM_ionfrac,
        Alrho_array,Almass_array,Altotalmass,Almasslist,Almassfraclist,
        p_coordindex,p_coord,p_dir,
        p_gammabin_index,p_gammabin,p_gamma,p_beta,p_absvel,p_KE,
        Al_gammamax,Al_gamma_interval_size,Al_gammavals,Al_betavals,
        p_pitchangle,p_cyclotronrad,p_pitch,p_dispperpitch,p_cyclotronperiod,
        p_LorentzForce,p_time,EnLossRateConst1,
        EnLossPerColDens_IonizExc,p_KELossPerPathlength_IonizExc,p_KELossPerPathlength_Plasma,
        n_e,p_KELossPerPathlength,
        FreeAnnihCrossSec_HEn,FreeAnnihCrossSec_LEn) = positronprop_initialize(params)


    print('starting propagator')

    Al26_MaxKELossPerStep = params.MaxKELossFracPerStep*(Al_gammamax - 1.0)*mp_eV*cp #J
    #subtract 1.0 because when gamma = 1, KE = 0.
    #this must NOT exceed corresponding gamma interval size!
    #only valid for Al atm. FIX THIS.

    #Propagation tempvars setup:

    UpdateIonizExcEnLoss = 0
    #if this = 1, then update ioniz+exc en loss rate by querying its lookup table.

    PhaseConst = 0.0 #setup tempvar for helix path phasechange

    #arrays------------------

    PhysUpdateTrig_high_now = np.ones((positroncount,dim))
    PhysUpdateTrig_low_now = np.ones((positroncount,dim))
    #All e+ start out with valbounds that, when crossed,...
    #...trigger a physics update in the current prop step.

    PhysUpdateTrig_high_next = np.ones((positroncount,dim))
    PhysUpdateTrig_low_next = np.ones((positroncount,dim))
    #Temp var to untangle if-tree logic
    #^^^These only hit the IonizExc enloss rate but are necessary

    validitygrid = np.ones((positroncount,dim))
    #validity boxes start out cellcorners-as-edges.

    #Set initial validity box coord values for each e+, m:
    validitybox_low = np.empty((positroncount,dim))
    validitybox_high = np.empty((positroncount,dim))

    p_in_sim_indicator = np.ones(positroncount)
    # =1 if e+ is inside simulation region =0 otherwise
    #array that tells you if e+ is inside the simulation boundary or not
    p_leavesim_count = 0 #track how many e+ have left the simulation

    p_KE_new = np.zeros(positroncount) #KE after prop step, J
    p_gamma_new = np.zeros(positroncount) #gamma after prop step
    p_pathlengthguess = np.zeros(positroncount) #pathlength per step
    p_mag_dispguess = np.zeros(positroncount) #displacement magnitude per step
    p_coordguess = np.empty((positroncount,dim)) #e+ coords after update, before boundarycheck
    valboundcrossindicator = np.zeros((positroncount,dim))
    #Says whether e+ crosses validity box boundary. 0 = no 1 = yes, "above" -1 = yes, "below".

    pathfracoutsidevalbox = np.zeros((positroncount,dim))
    #Fraction of the guessed propagation step that falls outside the current validitybox in each dim

    ValBoundToUpdate = np.zeros(dim)
    #This tells the validity boundary updater which specific boundary to update.
    #Boolean 0 = "don't update val bound". 1 means "update val bound".
    #if the code isn't broken, this should always be the first boundary encountered.
    #during a propagatiion step guess.

    p_decay_indicator = np.zeros(positroncount)
    # = 0: e+ has not decayed = 1: e+ has decayed.
    #later on, make fine correction to e+ position with an update-able log of positron phase.
    #maximum error in e+ coord from not doing this is 2*cyclotronrad per step. small.
    #however, this error may accumulate multiplicatively in the number of validity bound crossings.
    p_decay_count = 0 #tracks how many e+ have decayed

    valboundcrossenergy = np.zeros(positroncount)
    #stores the initial energy then the most recent energy at which a validity bound was crossed...
    #... and at which new ISM conditions were encountered.

    enlossbtvalboundcross = np.zeros(positroncount)
    #record energy lost between the two most recent validity bound crossings.

    step = np.ones(positroncount)
    #prop step counter we always start on step 1, not zero.

    #output arrays:
    traj = {
        "p_KE_record": np.array([]),
        "p_absvel_record:": np.array([]),
        "p_KELossPerPathlength_record:": np.array([]),
        "p_cyclotronrad_record:": np.array([]),
        "p_coord_record:": np.array([]),
        "enlossbtvalboundcross_record:": np.array([]),
        "valboundcrossenergy_record:": np.array([]),
        "p_gammabin_index_record:": np.array([]),
        "p_gamma_record:": np.array([]),
        "p_beta_record:": np.array([]),
        "p_beta_record": np.array([]),
        "p_KELossPerPathlength_Plasma_record": np.array([]),
        "HelixDPhase_record": np.array([]),
        "p_dir_record": np.array([]),
        "p_pitchangle_record": np.array([]),
        "p_pitch_record": np.array([]),
        "p_dispperpitch_record": np.array([]),
        "p_time_record": np.array([]),
        "validitybox_high_record": np.array([]),
        "validitybox_low_record": np.array([]),
        "validitygrid_record": np.array([]),
        "PhysUpdateTrigHighNow_record": np.array([]),
        "PhysUpdateTrigLowNow_record": np.array([]),
    }

    p_KE_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_absvel_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_KELossPerPathlength_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_cyclotronrad_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_coord_record = np.zeros((params.track_p_history_count,dim,params.maxframecount))
    enlossbtvalboundcross_record = np.zeros((params.track_p_history_count,params.maxframecount))
    valboundcrossenergy_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_gammabin_index_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_gamma_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_beta_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_KELossPerPathlength_Plasma_record = np.zeros((params.track_p_history_count,params.maxframecount))
    HelixDPhase_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_dir_record = np.zeros((params.track_p_history_count,dim,params.maxframecount))
    p_pitchangle_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_pitch_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_dispperpitch_record = np.zeros((params.track_p_history_count,params.maxframecount))
    p_time_record = np.zeros((params.track_p_history_count,params.maxframecount))
    validitybox_high_record = np.zeros((params.track_p_history_count,dim,params.maxframecount))
    validitybox_low_record = np.zeros((params.track_p_history_count,dim,params.maxframecount))
    validitygrid_record = np.zeros((params.track_p_history_count,dim,params.maxframecount))
    PhysUpdateTrigHighNow_record = np.zeros((params.track_p_history_count,dim,params.maxframecount))
    PhysUpdateTrigLowNow_record = np.zeros((params.track_p_history_count,dim,params.maxframecount))
    p_KELossPerPathlength_IE_record = np.zeros((params.track_p_history_count,params.maxframecount))
    #add here...  pathlength, pathlength per step, interaction depth per step

    ###Misc propagator initialization------------------

    #record initial conditions:
    for i in range(min(positroncount,params.track_p_history_count)):
        p_KE_record[i][0] = p_KE[i]
        p_absvel_record[i][0] = p_absvel[i]
        p_KELossPerPathlength_record[i][0] = p_KELossPerPathlength[i]
        p_cyclotronrad_record[i][0] = p_cyclotronrad[i]
        enlossbtvalboundcross_record[i][0] = enlossbtvalboundcross[i]
        valboundcrossenergy_record[i][0] = valboundcrossenergy[i]
        p_gammabin_index_record[i][0] = p_gammabin_index[i]
        p_gamma_record[i][0] = p_gamma[i]
        p_beta_record[i][0] = p_beta[i]
        p_KELossPerPathlength_Plasma_record[i][0] = p_KELossPerPathlength_Plasma[i]
        p_pitchangle_record[i][0] = p_pitchangle[i]
        p_pitch_record[i][0] = p_pitch[i]
        p_dispperpitch_record[i][0] = p_dispperpitch[i]
        p_time_record[i][0] = p_time[i]
        for m in range(dim):
            p_coord_record[i][m][0] = p_coord[i][m]
            p_dir_record[i][m][0] = p_dir[i][m]
            validitybox_high_record[i][m][0] = validitybox_high[i][m]
            validitybox_low_record[i][m][0] = validitybox_low[i][m]
            validitygrid_record[i][m][0] = validitygrid[i][m]
            PhysUpdateTrigHighNow_record[i][m][0] = PhysUpdateTrig_high_now[i][m]
            PhysUpdateTrigLowNow_record[i][m][0] = PhysUpdateTrig_low_now[i][m]
    #add here...  pathlength, pathlength per step, interaction depth per step

    #Set initial validity box coord values for each e+, m:
    for i in range(positroncount):
        validitybox_low[i][0] = cellcorner_m_x[round(p_coordindex[i][0])]
        validitybox_high[i][0] = cellcorner_m_x[round(p_coordindex[i][0] + 1.0)]
        validitybox_low[i][1] = cellcorner_m_y[round(p_coordindex[i][1])]
        validitybox_high[i][1] = cellcorner_m_y[round(p_coordindex[i][1] + 1.0)]
        validitybox_low[i][2] = cellcorner_m_z[round(p_coordindex[i][2])]
        validitybox_high[i][2] = cellcorner_m_z[round(p_coordindex[i][2] + 1.0)]
    #each box: indexes (0,0,1) in ascending order of (low_cellcorner,cellcenter,high_cellcorner).

#repeat some definitions from initialization step:

    def build_EnlossPerPathlength_Plasma(ISM_ionfrac,ISM_ndens,gamma,beta,n_e):
        return (ISM_ndens*(EnLossRateConst1*ISM_ionfrac/(beta**2.0))*\
                (log((gamma*beta**2.0)/(56.0*np.sqrt(n_e)*eVperJ/(mp_eV*cp))) + \
                log(2.0) - (1.0/2.0)*beta**2.0))
        #^energy loss per unit interaction depth, plasma wave excitations
        #^units of J/area dens (kg/m^2), then mult by gas mass dens (H for now) to get J/m

    def build_pitchangle(p_xdir,p_ydir,p_zdir,p_xindex,p_yindex,p_zindex): #pitchangle function
        return np.arccos((p_xdir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][0] \
                          + p_ydir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][1] \
                          + p_zdir*Btot[round(p_xindex)][round(p_yindex)][round(p_zindex)][2])\
                         /(np.sqrt(p_xdir**2.0 + p_ydir**2.0 + p_zdir**2.0)\
                           * (np.sqrt(Btot[round(p_xindex)][round(p_yindex)]\
                                      [round(p_zindex)][0]**2.0 \
                                      + Btot[round(p_xindex)][round(p_yindex)]\
                                      [round(p_zindex)][1]**2.0 \
                                      + Btot[round(p_xindex)][round(p_yindex)]\
                                      [round(p_zindex)][2]**2.0))))
        #^ = arccos((a dot b)/(|a||b|)) for a = initial direction and b = Btot direction
        #^in grid box where e+ exists radians

    def build_cyclotronrad(p_gamma,p_absvel,p_pitchangle,p_xindex,p_yindex,p_zindex):
        return p_gamma*(mp*p_absvel*np.sin(p_pitchangle)/\
                        (cp*Btot_abs[round(p_xindex)][round(p_yindex)][round(p_zindex)]))

    def build_pitch(cyclotronrad,pitchangle): #displacement per helix loop || B, m
        return 2.0*pi*cyclotronrad/np.tan(pitchangle)

    def build_dispperpitch(cyclotronrad,pitch): #physical pathlength, m
        return np.sqrt((2.0*pi*cyclotronrad)**2.0 + pitch**2.0)

###PROPAGATE------------------------------------

    for i in range(positroncount): #for all e+,
        while p_in_sim_indicator[i] == 1:
            #if i'th e+ is currently in simulation region,
            if p_decay_indicator[i] == 1:
                continue
                #if this e+ has already decayed, go to the next e+.
                
    #build naive straightline propagation step along guiding center:
            p_pathlengthguess[i] = np.abs(Al26_MaxKELossPerStep/p_KELossPerPathlength[i]) #m
            #compute propagation pathlength of displacement step guess, given current en loss rate
            
            #compute magnitude of displacement step guess:
            if p_pitchangle[i] == 0.0: #if e+ trajectory is exactly the guiding center,
                p_mag_dispguess[i] = p_pathlengthguess[i] #m
                #displacement guess is simply the pathlength guess
            else: #if e+ trajectory =/= exactly the guiding center,
                p_mag_dispguess[i] = p_pathlengthguess[i]*p_pitch[i]/p_dispperpitch[i] #m
                #displacement guess is pathlength guess scaled by ratio bt pitch and...
                #... displacement per pitch
            
    #         print("e+ ",i," step ",step[i]," pathlength in prop step = ",p_pathlengthguess[i])
    #         #debug
    #         print("e+ ",i," step ",step[i]," magnitude of displacement guess = ",p_mag_dispguess[i])
            #debug
        
            for m in range(dim):
                p_coordguess[i][m] = (p_coord[i][m] + p_mag_dispguess[i]*\
                                         Btot_dir[round(p_coordindex[i][0]),\
                                                  round(p_coordindex[i][1]),\
                                                  round(p_coordindex[i][2]),m])
                #set possible new e+ coords, given old coords, step guess, B at e+ starting point
                
    #         print("e+ ",i," step ",step[i]," guess of new coords = ",p_coordguess[i][:]) #debug
    #         print("e+ ",i," step ",step[i]," current validitybox lower bound = ",validitybox_low[i][:])
    #         #debug
    #         print("e+ ",i," step ",step[i]," current validitybox upper bound = ",validitybox_high[i][:])
    #         #debug
    #         print("e+ ",i," step ",step[i]," old coordindex = ",p_coordindex[i][:]) #debug

    #validitybox update procedure:
            #first, we must determine which validity boundary, if any, needs to be updated:
            for m in range(dim):
                if p_coordguess[i][m] > validitybox_high[i][m]:
                #if e+ will move RIGHT OF/IN FRONT OF/ABOVE current validity box,
                    valboundcrossindicator[i][m] = 1
                    #note that this will happen
                elif p_coordguess[i][m] < validitybox_low[i][m]:
                    #if e+ will move LEFT OF/BEHIND/BELOW current validity box,
                    valboundcrossindicator[i][m] = -1
                    #note that this will happen
                else: #if e+ will occupy the same validity box after prop step,
                    valboundcrossindicator[i][m] = 0
                    #reflect as much.
            #now use knowledge of whether e+ would cross any validity boundaries...
                if valboundcrossindicator[i][m] == 1:
                    #if an upper validity boundary was crossed by the guessed propagation step,
                    pathfracoutsidevalbox[i][m] = (np.abs(p_coordguess[i][m] - \
                                                          validitybox_high[i][m])/\
                                                   np.abs(p_coordguess[i][m] - p_coord[i][m]))
                    #fraction of guessed propagation step outside current validity box...
                    #... in mth dimension is...
                    #... |Endpoint - upper validitybound|/|Endpoint - startpoint|.
                if valboundcrossindicator[i][m] == -1:
                    #if a lower validity boundary was crossed by the guessed propagation step,
                    pathfracoutsidevalbox[i][m] = (np.abs(p_coordguess[i][m] - \
                                                          validitybox_low[i][m])/\
                                                   np.abs(p_coordguess[i][m] - p_coord[i][m]))
                    #fraction of guessed propagation step outside current validity box...
                    #... in mth dimension...
                    #... = |Endpoint - lower validitybound|/|Endpoint - startpoint|.
                    
    #         print("e+ ",i," step ",step[i]," e+ crosses this bound: ",valboundcrossindicator[i][:])
            #debug
        
            list_pathfracoutsidevalbox = list(pathfracoutsidevalbox[i][:])
            #tempvar: make list that max() can act on
            
    #         print("e+ ",i," step ",step[i]," list_pathfracoutsidevalbox = ",list_pathfracoutsidevalbox)
            #debug
        
            maxfracoutsidevalbox = max(list_pathfracoutsidevalbox)
            #tempvar: max fraction of e+ pathlength in any dimension outside current validity box
            
    #         print("e+ ",i," step ",step[i]," max prop step fraction outside validity box = ",\
    #               maxfracoutsidevalbox) #debug

            index_maxfracoutvalbox = list_pathfracoutsidevalbox.index(maxfracoutsidevalbox)
            #tempvar: index of dimension with maximum pathlength fraction outside current validity box

    #         print("e+ ",i," step ",step[i]," dimension where e+ valbound cross = ",\
    #               index_maxfracoutvalbox) #debug
        
            ValBoundToUpdate[index_maxfracoutvalbox] = 1
            #Tell valbound updater to act on validity bound dim where...
            #...pathlength frac outside current valbox = maximum.
            
    #once we know which, if any, validity boundary needs to be updated,
    #Update e+ physics coord indexes accordingly:
            if maxfracoutsidevalbox > 0.0:
                #if e+ will leave the current validity box this prop step,
                for m in range(dim):
                    if ValBoundToUpdate[m] == 1:
                        #if e+ will leave valbox in this dimension,
                        if valboundcrossindicator[i][m] == 1:
                            #if the e+ will exit current validitybox RIGHT/FORWARD/UP,
                            if p_coordindex[i][m] < centercount[m] - 1:
                                #if there are cells to the right/front/above current validity cell,
                                if validitygrid[i][m] == 0:
                                    #if current validity grid = cellcenters-as-edges,
                                    p_coordindex[i][m] = p_coordindex[i][m] + 1
                                    #increase e+ coord index for physics update AND validity bounds...
                                    #...this prop step
                                #then there are 3 cases:
                                if PhysUpdateTrig_high_now[i][m] == 1:
                                    #if we will update e+ physics this prop step,
                                    if PhysUpdateTrig_low_now[i][m] == 1:
                                        #if opposite valbound = active,
                                        #('active' = "if this physics validity bound...
                                        #...is crossed this prop step,...
                                        #...trigger e+ physics update.")
                                        PhysUpdateTrig_high_next[i][m] = 0
                                        #deactivate current valbound for next time e+ crosses it
                                        UpdateIonizExcEnLoss = 1
                                        #tell physics updater to update ioniz+exc en loss rate
                                    if PhysUpdateTrig_low_now[i][m] == 0:
                                        #if opposite validity boundary = inactive,
                                        PhysUpdateTrig_low_next[i][m] = 1
                                        #activate it for next time the e+ leaves upper valbox in this dim
                                if PhysUpdateTrig_high_now[i][m] == 0:
                                    #if we won't update e+ physics this prop step,
                                    PhysUpdateTrig_high_next[i][m] = 1
                                    #then update e+ phys next time this vbound is crossed
                            if p_coordindex[i][m] == centercount[m] - 1:
                                #if there are no cells to the right/front/top,
                                p_in_sim_indicator[i] = 0
                                #tell the propagator this e+ has left the simulation
                                #todo: log where e+ exits the simulation.
                                p_leavesim_count = p_leavesim_count + 1 #track how many e+ left sim
                        if valboundcrossindicator[i][m] == -1:
                            #if the e+ would exit current validitybox LEFT/BACK/DOWN,
                            if p_coordindex[i][m] > 0:
                                #if there are cells left/behind/below,
                                if validitygrid[i][m] == 1:
                                    #if current validity grid = cellcorners-as-edges,
                                    p_coordindex[i][m] = p_coordindex[i][m] - 1
                                    #decrease e+ coord index for physics update AND validity bounds...
                                    #...this prop step
                                #then there are 3 cases:
                                if PhysUpdateTrig_low_now[i][m] == 1:
                                    #if we will update e+ physics this prop step,
                                    if PhysUpdateTrig_high_now[i][m] == 1:
                                        #if opposite validity boundary = active,
                                        PhysUpdateTrig_low_next[i][m] = 0
                                        #deactivate current valbound for next time e+ crosses it
                                        UpdateIonizExcEnLoss = 1
                                        #tell physics updater to update ioniz+exc en loss rate
                                    if PhysUpdateTrig_high_now[i][m] == 0:
                                        #if opposite validity boundary = inactive,
                                        PhysUpdateTrig_high_next[i][m] = 1
                                        #activate it for next time the e+ leaves lower valbox in this dim
                                if PhysUpdateTrig_low_now[i][m] == 0:
                                    #if we won't update e+ physics this prop step,
                                    PhysUpdateTrig_low_next[i][m] = 1
                                    #then update physics next time this validitybound is crossed
                            if p_coordindex[i][m] == 0:
                                #if there are no cells to the left/back/bottom,
                                p_in_sim_indicator[i] = 0
                                #tell the propagator this e+ has left the simulation
                                p_leavesim_count = p_leavesim_count + 1
                                #track how many e+ left sim
                if p_in_sim_indicator[i] == 0:
                    #if e+ left the simulation region this prop step,
                    #reset all tempvars
                    ValBoundToUpdate = np.zeros(dim)
                    #reset valbound update-commanding tempvar for next e+
                    UpdateIonizExcEnLoss = 0
                    #tell physics updater to not update ioniz+exc en loss rate by default
                    print("e+",i," left simulation at step",step[i])
                    #announce when e+ leaves sim
                    continue
                    #halt update procedure and go to next e+.
                    
    # #        Now that we know whether e+ will leave the simulation this prop step,
    # #        and if not then what coord box it'll end up in,
    # #        update e+ physics:
                #compute actual energy loss in the step:
                p_KE_new[i] = p_KE[i] - (1.0-maxfracoutsidevalbox)*Al26_MaxKELossPerStep
                #assume energy loss is linear in propagation distance
                #fix to quadratic later.
                if params.track_p_KE == 1 and i < params.track_p_history_count:
                    #if we track this variable and if we track this positron
                    #(i starts at 0 so if track limit = 1, only "0th" gets recorded and...
                    #...history-tracking count limit is always satisfied.)
                    p_KE_record[i][round(step[i])] = p_KE_new[i]
                    #write to output
                
                enlossbtvalboundcross[i] = valboundcrossenergy[i] - p_KE_new[i]
                #measure energy lost between most recent two valbound crossings
                #use this to roll for in-flight annihilation...
                if params.track_enlossbtvalboundcross == 1 and i < params.track_p_history_count:
                    enlossbtvalboundcross_record[i][round(step[i])] = enlossbtvalboundcross[i]
                    #write to output
                
                valboundcrossenergy[i] = p_KE_new[i]
                #overwrite the energy at which the validitybound was crossed
                if params.track_valboundcrossenergy == 1 and i < params.track_p_history_count:
                    valboundcrossenergy_record[i][round(step[i])] = valboundcrossenergy[i]
                    #write to output
                    
                p_coord[i][:] = (p_coord[i][:] + (1.0 - maxfracoutsidevalbox)*\
                                    p_mag_dispguess[i]*\
                                    Btot_dir[round(p_coordindex[i][0]),\
                                             round(p_coordindex[i][1]),\
                                             round(p_coordindex[i][2]),:])
    #             print("e+ ",i," step ",step[i]," new e+ coords = ",p_coord[i][:]) #debug
                #update e+ coord to nearest validity boundary collision point
                #we'll overshoot in coordguess a bit if the e+ decayed this step.
                if params.track_p_coord == 1 and i < params.track_p_history_count:
                    for m in range(dim):
                        p_coord_record[i][m][round(step[i])] = p_coord[i][m]
                        #write to output
                
                if p_pitchangle[i] != 0.0: #If e+ isn't moving exactly along guiding center,
                    PhaseConst = ((1.0-maxfracoutsidevalbox)*p_mag_dispguess[i]\
                                  /np.abs(p_pitch[i]))
                    #set up constant for change in helix phase, given pitch along CURRENT prop path
                            
            else: #if e+ will stay in the same validity box after this prop step,
                p_KE_new[i] = p_KE[i] - Al26_MaxKELossPerStep
                #subtract the maximum KE loss per step
                if params.track_p_KE == 1 and i < params.track_p_history_count:
                    p_KE_record[i][round(step[i])] = p_KE_new[i]
                    #write to output
                
                p_coord[i][:] = p_coordguess[i][:]
                #Our new coord guess is correct in this case, so overwrite the old one
                #we'll overshoot in coordguess a bit if the e+ decayed this step.
    #             print("e+ ",i," step ",step[i]," new e+ coords = ",p_coord[i][:]) #debug
                if params.track_p_coord == 1 and i < params.track_p_history_count:
                    for m in range(dim):
                        p_coord_record[i][m][round(step[i])] = p_coord[i][m]
                        #write to output
                
                if p_pitchangle[i] != 0.0: #If e+ isn't moving exactly along guiding center,
                    PhaseConst = p_mag_dispguess[i]/p_pitch[i]
                    #set up constant for change in helix phase, dep. on current pitch and dispguess
                            
            p_gamma_new[i] = p_KE_new[i]/(mp_eV*cp) #update gamma
    #         print("e+ ",i," KE after prop step ",step[i]," = ",p_KE[i]/cp) #debug eV
    #         print("e+ ",i," gamma after prop step ",step[i]," = ",p_gamma[i]) #debug
            if params.track_p_gamma == 1 and i < params.track_p_history_count:
                p_gamma_record[i][round(step[i])] = p_gamma_new[i]
                #write to output

            if p_gamma_new[i] < p_gammabin[i] - 0.5*Al_gamma_interval_size:
                #if e+ KE dips below current KE interval this prop step,
                if p_gammabin_index[i] == 1:
                    #if we're currently in the lowest fully nonzero KE bin,
                    if p_KE_new[i] < (mp_eV + 100.0)*cp:
                        #if updated e+ KE < 100 eV greater than the e+ rest-frame energy,
                        p_KE[i] = (mp_eV + 100.0)*cp
                        #set KE of this e+ to 100 eV, which passes as input to low-energy decay...
                        #...heuristics.
                        if params.track_p_KE == 1 and i < params.track_p_history_count:
                            p_KE_record[i][round(step[i])] = p_KE[i]
                            #write to output[i][frame]
                        
                        p_gammabin_index[i] = 0
                        #set gamma bin to the one centered at KE = 0
                        if params.track_p_gammabin_index == 1 and i < params.track_p_history_count:
                            p_gammabin_index_record[i][round(step[i])] = p_gammabin_index[i]
                            #write to output[i][f]
                        
                        p_in_sim_indicator[i] = 0
                        p_decay_indicator[i] = 1
                        #tell the propagator that this e+ has decayed out of the simulation.
                        #otherwise, keep the same ionizexc en loss rate and gammabin index.
                        p_decay_count = p_decay_count + 1 #track how many e+ have decayed
                        
                        #reset all tempvars:
                        
                        ValBoundToUpdate = np.zeros(dim)
                        #reset valbound update-commanding tempvar for next e+
                        UpdateIonizExcEnLoss = 0
                        #tell physics updater to not update ioniz+exc en loss rate by default
                        print("e+ ",i,"decayed out at step ",step[i])
                        #announce what step e+ decayed out of the simulation
                        
                        continue
                        #halt update procedure and go to next e+.
                        
                else: # if we're not currently in the lowest fully nonzero KE bin,
                    p_gammabin_index[i] = p_gammabin_index[i] - 1
                    #decrease the energy interval we're in, for ioniz+exc en loss rate
                    if params.track_p_gammabin_index == 1 and i < params.track_p_history_count:
                        p_gammabin_index_record[i][round(step[i])] = p_gammabin_index[i]
                        #write to output
                    
                    UpdateIonizExcEnLoss = 1
                    #tell physics updater to update ioniz+exc en loss rate.        
            
            p_KE[i] = p_KE_new[i]
            p_gamma[i] = p_gamma_new[i]
            #if we didn't stop propagating the e+ this step, overwrite its KE and gamma
    #         print("e+ ",i," step ",step[i]," ionizexc enloss rate indicator: ",\
    #               UpdateIonizExcEnLoss) #debug

            if UpdateIonizExcEnLoss == 1: #if we should update the ioniz+exc en loss rate...
                p_KELossPerPathlength_IonizExc[i] = (protonmass_kg*ISM_Hndens\
                                                        [round(p_coordindex[i][0])]\
                                                        [round(p_coordindex[i][1])]\
                                                        [round(p_coordindex[i][2])]*\
                                                        EnLossPerColDens_IonizExc\
                                                        [round(p_gammabin_index[i])])
    #             print("e+ ",i," step ",step[i]," p_KELossPerPathlength_IonizExc = ",\
    #                   p_KELossPerPathlength_IonizExc[i]) #debug
                #update ioniz+exc energy loss rate, J/m
        
                #add here... write to output
        
            p_beta[i] = np.sqrt(1.0-(1.0/(p_gamma[i]**2.0))) #update beta
            if params.track_p_beta == 1 and i < params.track_p_history_count:
                p_beta_record[i][round(step[i])] = p_beta[i]
                #write to output[i][f]
            
            p_absvel[i] = c*p_beta[i] #update abs velocity
            if params.track_p_absvel == 1 and i < params.track_p_history_count:
                p_absvel_record[i][round(step[i])] = p_absvel[i]
                #write to output
            
            p_KELossPerPathlength_Plasma[i] = (protonmass_kg*\
                                                  build_EnlossPerPathlength_Plasma\
                                                  (ISM_ionfrac[round(p_coordindex[i][0])]\
                                                   [round(p_coordindex[i][1])]\
                                                   [round(p_coordindex[i][2])],\
                                                   ISM_Hndens[round(p_coordindex[i][0])]\
                                                   [round(p_coordindex[i][1])]\
                                                   [round(p_coordindex[i][2])],\
                                                   p_gamma[i],p_beta[i],\
                                                   n_e[round(p_coordindex[i][0])]\
                                                   [round(p_coordindex[i][1])]\
                                                   [round(p_coordindex[i][2])]))
    #         print("e+ ",i," step ",step[i]," p_KELossPerPathlength_Plasma = ",\
    #               p_KELossPerPathlength_Plasma[i])
            if params.track_p_KELossPerPathlength_Plasma == 1 and i < params.track_p_history_count:
                p_KELossPerPathlength_Plasma_record[i][round(step[i])] = p_KELossPerPathlength_Plasma[i]
                #write to output
        
            #debug
            #update plasma wave energy loss rate, J/m                    
            p_KELossPerPathlength[i] = (p_KELossPerPathlength_IonizExc[i] + \
                                           p_KELossPerPathlength_Plasma[i])
    #         print("e+ ",i," step ",step[i]," p_KELossPerPathlength = ",\
    #               p_KELossPerPathlength[i]) #debug
            #update total energy loss rate, J/m
            if params.track_p_KELossPerPathlength == 1 and i < params.track_p_history_count:
                p_KELossPerPathlength_record[i][round(step[i])] = p_KELossPerPathlength[i]
                #write to output
            
            HelixDPhase = 2.0*pi*(PhaseConst - floor(PhaseConst))
            #compute change in e+ orientation angle in plane orthog to B, radians
            if params.track_HelixDPhase == 1 and i < params.track_p_history_count:
                HelixDPhase_record[i][round(step[i])] = HelixDPhase
                #write to output[i][f]
            
            p_dir[i][:] = (p_dir[i][:]*np.cos(HelixDPhase) + \
                              Btot_dir[round(p_coordindex[i][0]),\
                                       round(p_coordindex[i][1]),\
                                       round(p_coordindex[i][2]),:]*\
                             np.dot(p_dir[i][:],Btot_dir\
                                    [round(p_coordindex[i][0]),\
                                     round(p_coordindex[i][1]),\
                                     round(p_coordindex[i][2]),:])*\
                              (1.0 - np.cos(HelixDPhase)) - \
                             np.cross(p_dir[i][:],Btot_dir\
                                      [round(p_coordindex[i][0]),\
                                       round(p_coordindex[i][1]),\
                                       round(p_coordindex[i][2]),:]*\
                                      np.sin(HelixDPhase)))
            #update e+ orientation unit vector:...
            #...dir*cos(helixphase) + B(dir . B)(1-cos(helixphase)) - (dir x Bsin(helixphase))
            if params.track_p_dir == 1 and i < params.track_p_history_count:
                for m in range(dim):
                    p_dir_record[i][m][round(step[i])] = p_dir[i][m]
                #write to output[i][:][f]
            
            p_pitchangle[i] = build_pitchangle(p_dir[i][0],\
                                                  p_dir[i][1],p_dir[i][2],\
                                                  p_coordindex[i][0],\
                                                  p_coordindex[i][1],\
                                                  p_coordindex[i][2])
            #update pitch angle
            if params.track_p_pitchangle == 1 and i < params.track_p_history_count:
                p_pitchangle_record[i][round(step[i])] = p_pitchangle[i]
                #write to output[i][f]
            
            p_cyclotronrad[i] = build_cyclotronrad(p_gamma[i],p_absvel[i],\
                                                      p_pitchangle[i],\
                                                      p_coordindex[i][0],\
                                                      p_coordindex[i][1],\
                                                      p_coordindex[i][2])
            #update cyclotron radius
            if params.track_p_cyclotronrad == 1 and i < params.track_p_history_count:
                p_cyclotronrad_record[i][round(step[i])] = p_cyclotronrad[i]
                #write to output
            
            if p_pitchangle[i] != 0.0: #If e+ isn't moving exactly along guiding center,
                p_pitch[i] = build_pitch(p_cyclotronrad[i],p_pitchangle[i])
                #update pitch
                if params.track_p_pitch == 1 and i < params.track_p_history_count:
                    p_pitch_record[i][round(step[i])] = p_pitch[i]
                    #write to output[i][f]
            
                p_dispperpitch[i] = build_dispperpitch(p_cyclotronrad[i],p_pitch[i])
                #update displacement per pitch
                if params.track_p_dispperpitch == 1 and i < params.track_p_history_count:
                    p_dispperpitch_record[i][round(step[i])] = p_dispperpitch[i]
                    #write to output
            
            if maxfracoutsidevalbox > 0.0:
                p_time[i] = p_time[i] + (np.abs((1.0 - maxfracoutsidevalbox)*\
                                               p_mag_dispguess[i]/p_absvel[i]))
            else:
                p_time[i] = p_time[i] + np.abs(p_mag_dispguess[i]/p_absvel[i])
                #add to total time elapsed during e+ prop.
            if params.track_p_time == 1 and i < params.track_p_history_count:
                p_time_record[i][round(step[i])] = p_time[i]
                #write to output
            
            #call free annih cross section here, now that p_KE, gammabin_index, and...
            #...valboundcrosstime are updated:

    #With e+ physics updated, we can now update validity cell bounds for next prop step:
            if maxfracoutsidevalbox > 0.0:
                #if e+ left the current validity box this prop step,
                for m in range(dim):
                    if (PhysUpdateTrig_high_next[i][m] == 0 or \
                        PhysUpdateTrig_low_next[i][m] == 0):
                    #if either valbound will update to become inactive next prop step,
                        validitygrid[i][m] = 0 #update valgrid to cellcenters-as-edges.
                        #grid becomes cellcenters-as-edges when either indicator = 0 (inactive).
                    else: #if both valbounds will update to become active next prop step,
                        validitygrid[i][m] = 1 #update valgrid to cellcorners-as-edges.
                        #grid becomes cellcorners-as-edges when both indicators = 1 (active).
                    if params.track_validitygrid == 1 and i < params.track_p_history_count:
                        validitygrid_record[i][m][round(step[i])] = validitygrid[i][m]
                        #write to output
    #             print("e+ ",i," step ",step[i]," validitygrid = ",validitygrid[i][:]) #debug
    #Now that we know which validity boundaries to update, update their coord values:
                if validitygrid[i][0] == 0: #if grid in x updated to cellcenters-as-edges for next prop,
                        validitybox_low[i][0] = cellcenter_m_x[round(p_coordindex[i][0])]
                        validitybox_high[i][0] = cellcenter_m_x[round(p_coordindex[i][0] + 1.0)]
                else: #if grid updated to cellcorners-as-edges just now,
                        validitybox_low[i][0] = cellcorner_m_x[round(p_coordindex[i][0])]
                        validitybox_high[i][0] = cellcorner_m_x[round(p_coordindex[i][0] + 1.0)]
                        #update validity boundary values accordingly.
                if validitygrid[i][1] == 0: #if grid in y updated to cellcenters-as-edges just now,
                        validitybox_low[i][1] = cellcenter_m_y[round(p_coordindex[i][1])]
                        validitybox_high[i][1] = cellcenter_m_y[round(p_coordindex[i][1] + 1.0)]
                else: #if grid updated to cellcorners-as-edges just now,
                        validitybox_low[i][1] = cellcorner_m_y[round(p_coordindex[i][1])]
                        validitybox_high[i][1] = cellcorner_m_y[round(p_coordindex[i][1] + 1.0)]
                        #update validity boundary values accordingly.
                if validitygrid[i][2] == 0: #if grid in z updated to cellcenters-as-edges for next prop,
                        validitybox_low[i][2] = cellcenter_m_z[round(p_coordindex[i][2])]
                        validitybox_high[i][2] = cellcenter_m_z[round(p_coordindex[i][2] + 1.0)]
                        #update validity boundary values accordingly.
                else: #if grid updated to cellcorners-as-edges just now,
                        validitybox_low[i][2] = cellcorner_m_z[round(p_coordindex[i][2])]
                        validitybox_high[i][2] = cellcorner_m_z[round(p_coordindex[i][2] + 1.0)]
                        #update validity boundary values accordingly.
                for m in range(dim):
                    PhysUpdateTrig_low_now[i][m] = PhysUpdateTrig_low_next[i][m]
                    PhysUpdateTrig_high_now[i][m] = PhysUpdateTrig_high_next[i][m]
                    #overwrite physics update trigger to pass...
                    #...decisions made this prop step to next prop step

            if params.track_validitybox == 1 and i < params.track_p_history_count:
                for m in range(dim):
                    validitybox_high_record[i][m][round(step[i])] = validitybox_high[i][m]
                    validitybox_low_record[i][m][round(step[i])] = validitybox_low[i][m]
                    #write new valbound coords to output, regardless of whether they updated or not

            if params.track_PhysUpdateTrigNow == 1 and i < params.track_p_history_count:
                for m in range(dim):
                    PhysUpdateTrigHighNow_record[i][m][round(step[i])] = PhysUpdateTrig_high_now[i][m]
                    PhysUpdateTrigLowNow_record[i][m][round(step[i])] = PhysUpdateTrig_low_now[i][m]

    #at end of prop step, do tempvar resets:

            step[i] = step[i] + 1.0
            if step[i] == params.maxframecount:
                break
            #stop the propagator if KE can't decay
            #tempvar count what step program gets to before throwing error...
            for m in range(dim):
                valboundcrossindicator[i][m] = 0
            #reset indicator for whether current e+ would cross a given validity boundary this step
            pathfracoutsidevalbox[i][:] = np.zeros(dim)
            #tell propagator that e+ is not outside its new validity box
            ValBoundToUpdate = np.zeros(dim)
            #reset valbound update-commanding tempvar for next e+
            UpdateIonizExcEnLoss = 0
            #tell physics updater to not update ioniz+exc en loss rate by default

        #push to graphics:

    if (params.track_p_KELossPerPathlength_Plasma == 1 and params.track_p_KELossPerPathlength == 1\
        and params.track_p_KELossPerPathlength_IE == 1):
        p_KELossPerPathlength_IE_record = (p_KELossPerPathlength_record - 
            p_KELossPerPathlength_Plasma_record)
        #write to output

    traj = { #add variable list here
        "p_KE": p_KE_record[params.phistorytoplot][0:round(step[params.phistorytoplot])],
        #define key for graphics to point to: record[e+][prop steps]
        "p_time": p_time_record[params.phistorytoplot][0:round(step[params.phistorytoplot])]
        }

    print(traj)

    yield traj,[]

        #loop to next step of current e+
    #loop to next e+

    print('ending propagator')
