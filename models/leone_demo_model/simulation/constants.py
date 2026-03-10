dim = 3; #spatial dimension count
mp = 9.109*10**-31.0; #positron rest mass, kg
mp_eV = 510998.95069; #positron rest mass, eV/c^2
cp = 1.602176634*10**-19.0; #positron charge, Coulombs; also J per eV
pi = 3.141592653589793238462643383279502884197169399; #better pi
sagAdist = 8.277; #distance from Earth to galactic center, pc
c = 299792458.0; #lightspeed, SI
alpha = 0.0072973525643; #fine structure constant
h = 6.62607015*10**-34.0; #Planck's constant, J/hz
hbar = h/(2.0*pi);
g = 2.0; #coupling constant in weak interaction
m_in_pc = 30856775814913673.0; #meters in a parsec
Msol = 1.98845*10**30.0; #kg in a solar mass
positronQrad = 2.818*10**-15.0; #positron classical charge radius, m; r0 in Berger+Seltzer
protonmass_kg = 1.6726*10**-27.0; #kg; m_n in milne; also H mass
kgPerAMU = 1.660540199*10**-27.0; #kg per amu;
eVperJ = 6.241506*10**18; #eV per J

#KE constants------------------------------

#Assume, for now, 26Al beta+ decay for all positron production. [modify later]
Al_Z = 12.0; #daughter product charge, Mg2+ for 26Al beta decay
Al_Emax = 1140000.0; #Maximum 26Al beta decay energy in a given e+, eV
M_if = 1.0; #Fermi golden rule perturbation amplitude between states (value = irrel)
#discover value by normalizing, if desired.

#add here...

#Set phi for spherical coords: elevation angle
altmin = -pi/2.0;
altmax = pi/2.0;
#azimuth angle
# thetamin = 0.0;
# thetamax = 2*pi;

#Bloc endstep------------------

# print(varnames_universalconst);
