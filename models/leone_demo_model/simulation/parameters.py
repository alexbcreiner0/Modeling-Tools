from dataclasses import dataclass, field
from numpy import array, ndarray, e

@dataclass
class Params:

    ###Boolean scenario control variables------------------

    #ISM configuration:---------
    ISMcond_test1_ctrl: bool = True
    #add here... more scenarios, disallow contradictions;

    #B field components:---------
    B_dipole_ctrl: bool = False
    B_uniform_ctrl: bool = True
    #add here...

    #positron src-types:---------
    Al26_src1_ctrl: bool = False #26Al src-type ctrl;
    SagA_src_ctrl: bool = True #Sagittarius A src-type (point src) ctrl;
    #add here...

    #positron initial movement dir---------
    p_dir_isotropic_ctrl: bool = True
    p_dir_uniform_ctrl: bool = False
    #add here...

    #positron counts:---------
    Al_positroncount: int = 50
    SagA_positroncount: int = 100
    #add here...

    ###Output-tracking ctrl variables------------------
    track_p_history_count: int = 5
    #Decide how many positrons to track in detail;

    phistorytoplot: int = 0
    #Which positron history do we plot?

    p_movement_dirs: str = 'isotropic'
    
    track_p_time: bool = True

    track_p_coord: bool = True
    track_p_gammabin_index: bool = True
    track_p_gamma: bool = True
    track_p_beta: bool = True
    track_p_absvel: bool = True
    track_p_KE: bool = True
    track_p_cyclotronrad: bool = True
    track_HelixDPhase: bool = True
    track_p_dir: bool = True
    track_p_pitchangle: bool = True
    track_p_pitch: bool = True
    track_p_dispperpitch: bool = True

    track_valboundcrossenergy: bool = True
    track_enlossbtvalboundcross: bool = True
    track_p_KELossPerPathlength: bool = True
    track_p_KELossPerPathlength_Plasma: bool = True
    track_p_KELossPerPathlength_IE: bool = True
    track_validitybox: bool = True
    track_validitygrid: bool = True
    track_PhysUpdateTrigNow: bool = False
    #add here...  pathlength, pathlength per step, interaction depth per step;

    ###coord grid------------------

    coord_translation = 0.01 #pc
    #how far we offset grid in xyz to avoid zero-valued coords
    #for physics or coordinate transformation computations.

    xexpbase: float = 1.5 #base to raise by exp to build x coords;
    xexpintervals: int = 52 #x coord exponent interval count; MUST BE EVEN INT;
    xexpfactor_min: float = -0.5 #set up range of exponent space. AVOID changing.
    xexpfactor_max: float = 25.0 #^";

    yexpbase: float = 1.5 #base to raise by exp to build y coords; axisymm when equal to xexpbase;
    yexpintervals: int = 52 #x coord exponent interval count; MUST BE EVEN INT;
    yexpfactor_min: float = -0.5 #set up range of exponent space;
    yexpfactor_max: float = 25.0 #^";

    zexpbase: float = e
    zexpintervals: int = 22 #z coord exponent interval count; MUST BE EVEN INT;
    zexpfactor_min: float = -0.5 #set up range of exponent space;
    zexpfactor_max: float = 10.0 #^";

    ###B field------------------

    b_uniform_x: float = 0.002 #tesla
    b_uniform_y: float = 0.001
    b_uniform_z: float = 0.0

    b_dipole_x: float = 0.0
    b_dipole_y: float = 0.0
    b_dipole_z: float = 1.0*10**53
    #sets |B| to O(10^-9 T near Earth, at coord [98,52,29])
    #for dipole field; automate this!
    #align dipole moment along z
    #Dipole currently passes through galactic origin.

    #X-shaped field:---------



    #static random field:---------



    #dynamic random field:---------



    #add here...



    ###MW ISM------------------

    #first option of ISM dist:---------

    #Temps, Kelvin
    MM_Temp: float = 15.0 #10-20 K
    CNM_Temp: float = 50.0 #20-100 K
    WNM_Temp: float = 5000.0 #1e3-1e4 K
    WIM_Temp: float = 8000.0 #always 8k K
    HIM_Temp: float = 10.0**6 #always 1e6 K

    #Hydrogen number densities, m^-3
    #nH = n for now!!!
    MM_Hndens: float = 10.0**10 #1e2-1e6 cm^-3
    CNM_Hndens: float = 5.0*10**7 #20-100 ''
    WNM_Hndens: float = 1.0*10**6 #0.2-2 ''
    WIM_Hndens: float = 2.0*10**5 #0.1-0.3 ''
    HIM_Hndens: float = 6.0*10**3 #0.003-0.01 ''

    #Ionization fraction
    MM_IonizFrac: float = 5.0*10**-5 #<= 10^-4
    CNM_IonizFrac: float = 7.0*10**-4 #4*10^-4 to 10^-3
    WNM_IonizFrac: float = 0.02 #0.007-0.05
    WIM_IonizFrac: float = 0.75 #0.6-0.9
    HIM_IonizFrac: float = 1.0

    #loop over possibilities here in experiments

    #second option of ISM dist:---------

    #add here...

    ###MW 26Al mass dist------------------

    #parameters for double-exponential dist outside the MW central bulge:

    bulgediskboundary_pc: float = 2000.0; #innermost radius of 26Al density, pc
    Alrho0: float = 2.204*10**-30; #26Al density at galactic center, kg/pc^3.
    z0: float = 100.0; #characteristic altitude where mass density drops off by factor of e, pc
    r0: float = 10000.0; #characteristic radius ^^^


    #loop over possibilities here in experiments

    #parameters for other 26Al distribution options:

    #add here...

    #parameters for other e+ source-type distributions:

    #add here...

    ###positrons' spawn coordinates------------------

    SagA_MWcenter = 0 #Place SagA e+ at MW center;
    SagA_arbitrary = 1 #Place SagA e+ at arbitrary location in the MW;

    SagA_x_customspawncoordindex = 20 #Select grid index where point-src e+ spawns
    SagA_y_customspawncoordindex = 20
    SagA_z_customspawncoordindex = 20

    ###positrons' initial movement directions------------------

    #when all e+ initial dirs are isotropically sampled:---------
    # thetaintervals = 360; #must be even to avoid 0
    altintervals = 181 #every integer degree, including 0;
    #how finely grained positrons' initial directions are isotropically sampled;

    #when all e+ have same initial movement dirs:---------

    p_dir_uniform_mag_x: float = 2.0
    #non-normalized x component of initial e+ direction;
    p_dir_uniform_mag_y: float = 1.0
    p_dir_uniform_mag_z: float = 0.0

    ###positrons' initial KEs------------------

    gamma_bin_count: int = 100 #number of gamma bins
    #coarse-grains probabilities over the positrons' initial energy distributions;
    #used for tabulated interaction cross-sections;

    ###En loss rate------------------

    #H-only scenario:
    Z_bound_H: float = 1.0 #bound e- count per atom; Z_B in Milne
    atomicweight: float = 1.0 #Amu; A in Milne; 

    #generalize this to scenarios with multiple species;

    #add here...

    Z_avg: float = 1.0 #avg bound e- count per atom

    ###Spatial propagator------------------

    MaxKELossFracPerStep: float = 0.005 #maximum e+ KE fraction lost per step
    maxframecount: int = 300 #maximum allowed animation framecount before program is forced to halt;