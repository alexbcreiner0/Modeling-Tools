from dataclasses import dataclass, field
from numpy import array, ndarray
from math import exp

@dataclass
class Params:
    ISMcond_test1_ctrl: bool = True
    p_movement_dirs: str = 'isotropic'
    Al_positroncount: int = 50
    SagA_positroncount: int = 100
    track_p_history_count: int = 50
    track_p_time: bool = True
    track_p_coord: bool = True
    track_p_gammabin_index: bool = True
    track_p_beta: bool = True
    track_p_absvel: bool = True
    track_p_cyclotronrad: bool = True
    track_p_KE: bool = True
    track_HelixDPhase: bool = True
    track_p_dir: bool = True
    track_p_pitchangle: bool = True
    track_p_pitch: bool = True
    track_p_dispperpitch: bool = True
    track_valboundcrossenergy: bool = True
    track_enlossbtvalboundcross: bool = True
    track_p_KELossPerPathlength: bool = True
    track_p_KELossPerPathlength_Plasma: bool = True
    track_validitybox: bool = True
    track_validitygrid: bool = True
    phistorytoplot: bool = True
    coord_translation = 0.01; #pc
    #how far we offset grid in xyz to avoid zero-valued coords
    #for physics or coordinate transformation computations.
    xexpbase: float = 1.5; #base to raise by exp to build x coords;
    xexpintervals: int = 52; #x coord exponent interval count; MUST BE EVEN INT;
    xexpfactor_min: float = -0.5; #set up range of exponent space. AVOID changing.
    xexpfactor_max: float = 25.0; #^";
    yexpbase: float = 1.5; #base to raise by exp to build y coords; axisymm when equal to xexpbase;
    yexpintervals: int = 52; #x coord exponent interval count; MUST BE EVEN INT;
    yexpfactor_min: float = -0.5; #set up range of exponent space;
    yexpfactor_max: float = 25.0; #^";
    zexpbase: float = exp(1.0);
    zexpintervals: int = 22; #z coord exponent interval count; MUST BE EVEN INT;
    zexpfactor_min: float = -0.5; #set up range of exponent space;
    zexpfactor_max: float = 10.0; #^";
