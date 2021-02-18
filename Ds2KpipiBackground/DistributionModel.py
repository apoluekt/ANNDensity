import math, sys

import tensorflow as tf

import amplitf.interface as atfi
import amplitf.kinematics as atfk
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace
from amplitf.phasespace.veto_phasespace import VetoPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr

# Masses of the final state particles
md  = 1.96834
mpi = 0.13957
mk  = 0.49368

# Masses and widths of the resonances
mkstar = 0.892
wkstar = 0.050
mrho = 0.770
wrho = 0.150

# Eta acceptance
etamin = 2. 
etamax = 5.

# Declare phase spaces
dlz_phsp = DalitzPhaseSpace( mk, mpi, mpi, md )   # conventional Dalitz plot for Ds -> K pi pi 
sqdlz_phsp = RectangularPhaseSpace( ((0., 1.), (0., 1.)) )     # square Dalitz plot
m_phsp = RectangularPhaseSpace( ((1.97-0.2, 1.97+0.2), ) )     # m(Ds) range
observables_phase_space = CombinedPhaseSpace(sqdlz_phsp, m_phsp)  # Combination of square DP and m(Ds) as the observables phase space

exp_phase_space = VetoPhaseSpace(observables_phase_space, 2, (1.97-0.05, 1.97+0.05) )  # The phase space of the "experimental" data sample
                                                                                   # with vetoed "signal" region
#expPhaseSpace = VetoPhaseSpace(observablesPhaseSpace, 2, (1.97-0.2, 1.97+0.05) )  # The phase space of the "experimental" data sample
#                                                                                   # with vetoed "signal" region

random_array_size = 12 # Number of rows for auxiliary random tensor to be used for toy MC

#observablesTitles = [ "m'", "#theta'", "m(K^{+}#pi^{#font[122]{-}}#pi^{+}) (GeV)" ]
observables_titles = [ r"$m'$", r"$\theta'$", r"$m_{D}$ (GeV)" ]

observables_data = [ "mprime", "thetaprime", "md" ]

observables_toys = [ "mprime", "thetaprime", "md" ]

generated_variables = [ "mprime", "thetaprime", "md", "m2kpi", "m2pipi" ]

parameters_list = [
    ("kmeanpt",  r"Mean $p_{T}(K)$ (GeV)",   (0.2, 1.),  None), 
    ("pimeanpt", r"Mean $p_{T}(\pi)$ (GeV)", (0.2, 1.),  None), 
    ("ptcut",    r"Track $p_{T}$ cut (GeV)", (0.1, 0.5), None), 
    ("pcut",     r"Track $p$ cut (GeV)",     (1.0, 4.0), -6.), 
    ("kstarmeanpt", r"Mean $p_{T}(K^{*})$ (GeV)",   (0.5, 3.),  None), 
    ("rhomeanpt",   r"Mean $p_{T}(\rho)$ (GeV)",    (0.5, 3.),  None), 
    ("kstarfrac",   r"$K^{*}$ fraction",   (0., 0.3),  None), 
    ("rhofrac",     r"$\rho$ fraction",    (0., 0.3),  None), 
]

# "True" values of the model parameters to be used for generation of the test sample
#trueCuts = [ Const(0.3), Const(0.6), Const(0.3), Const(3.0), Const(2.0), Const(2.0), Const(0.10), Const(0.20) ]
true_cuts = [ atfi.const(0.3), atfi.const(0.6), atfi.const(0.3), atfi.const(3.0), 
              atfi.const(2.0), atfi.const(2.0), atfi.const(0.10), atfi.const(0.20) ]

bounds = { i[0] : (i[2], i[3]) for i in parameters_list }

def uniform_random(rnd, x1, x2) : 
  """
    Uniform random numbers from x1 to x2
  """
  return x1 + rnd*(x2-x1)

def normal_random(rnd1, rnd2) : 
  """
    Normal distribution from two random numbers
  """
  return atfi.sqrt(-2.*atfi.log(rnd1))*atfi.cos(2.*math.pi*rnd2)

def breit_wigner_random(rnd, mean, gamma) : 
  """
    Random Breit-Wigner distribution with specified mean and width
  """
  rval = 2.*rnd - 1
  displ = 0.5*gamma*atfi.tan(rval*math.pi/2.)
  return mean + displ

def generate_exp(rnd, x1, x2, alpha = None) : 
  """
    Exponential random distribution with constant "alpha", 
    limited to the range x1 to x2
  """
  if isinstance(x1, float) : x1 = atfi.const(x1)
  if isinstance(x2, float) : x2 = atfi.const(x2)
  if alpha is None or (isinstance(alpha, float) and alpha == 0.) : 
    return uniform_random(rnd, x1, x2)
  else : 
    if isinstance(alpha, float) : alpha = atfi.const(alpha)
    xi1 = atfi.exp(-x1/alpha)
    xi2 = atfi.exp(-x2/alpha)
    ximin = atfi.min(xi1, xi2)
    ximax = atfi.max(xi1, xi2)
    return atfi.abs(alpha*atfi.log(uniform_random(rnd, ximin, ximax) ) )

def generate_pt(rnd, mean, cut1, cut2) : 
  """
    Generate Pt distribution, with mean "mean", and miminum Pt "cut"
  """
  return generate_exp(rnd, cut1, cut2, mean)

def generate_eta(rnd) : 
  """
    Generate pseudorapidity, uniform from 2 to 5.
  """
  return uniform_random(rnd, 2., 5.)    # Eta, uniform in (2., 5.)

def generate_phi(rnd) : 
  """
    Generate phi, uniform in 0, 2pi
  """
  return uniform_random(rnd, 0., 2.*math.pi) # Phi, uniform in (0, 2pi)

def momentum_resolution(p) :
  """
    Relative momentum resolution as a function of momentum (here = 0.5%, constant)
  """
  return 0.005

def momentum_scale(dm, moms) :
  """
    Function to calculate the momentum scale factor for kinematic fit
      dm   : invariant mass shift from the desired value
      moms : list of 4-momenta of the final state particles
  """
  psum = sum(moms)  # sum of 4-momenta
  pvecsum = atfk.spatial_components(psum)
  esum = atfk.time_component(psum)
  dedd = atfi.const(0.)
  pdpdd = atfi.const(0.)
  for mom in moms : 
    pm   = atfk.p(mom)  # Absolute momentum
    em   = atfk.time_component(mom) # Energy
    pvec = atfk.spatial_components(mom) # 3-momentum
    s    = momentum_resolution(pm)
    dedd += s*pm**2/em
    pdpdd += atfk.scalar_product(pvecsum, pvec)*s
  return -dm/(2.*esum*dedd - 2.*pdpdd)

def kinematic_fit(mfit, moms) : 
  """
    Kinematic fit to a fixed invariant mass for a multibody decay. 
    Returns the fitted mass and the list of final state 4-momenta. 
  """
  mcorr = atfk.mass(sum(moms))
  for l in range(3) :
    dm2 = mcorr**2-mfit**2
    delta = momentum_scale(dm2, moms)
    moms2 = []
    for mom in moms :
      m2 = atfk.mass(mom)**2
      momvec = atfk.spatial_components(mom)*atfk.scalar(1+delta*momentum_resolution(atfk.p(mom)))
      mom2 = atfk.lorentz_vector(
        momvec, 
        atfi.sqrt(m2 + atfk.norm(momvec)**2)
      )
      moms2 += [ mom2 ]
    moms = moms2
    mcorr = atfk.mass(sum(moms))
  return mcorr, moms

def generate_4momenta(rnd, meanpt, ptcut, m) : 
  """
    Generate random 4-momenta according to specified mean Pt, minimum Pt, and mass of the particle, flat in eta and phi
  """
  pt  = generate_pt(rnd[:,0], meanpt, ptcut, atfi.const(200.))  # Pt in GeV
  eta = generate_eta(rnd[:,1])         # Eta
  phi = generate_phi(rnd[:,2])         # Phi

  theta = 2.*atfi.atan(atfi.exp(-eta))
  p  = pt/atfi.sin(theta)     # Full momentum
  e  = atfi.sqrt(p**2 + m**2) # Energy
  px = p*atfi.sin(theta)*atfi.sin(phi)
  py = p*atfi.sin(theta)*atfi.cos(phi)
  pz = p*atfi.cos(theta)
  return atfk.lorentz_vector(atfk.vector(px, py, pz), e)

def generate_rotation_and_boost(moms, minit, meanpt, ptcut, rnd) : 
  """
    Generate 4-momenta of final state products boosted to lab frame and randomly rotated
      moms   - initial particle momenta (in the rest frame)
      minit  - mass of the initial particle
      meanpt - mean Pt of the initial particle
      ptcut  - miminum Pt of the initial particle
      rnd    - Auxiliary random tensor with 6 rows
  """

  pt  = generate_pt(rnd[:,0], meanpt, ptcut, 200.)  # Pt in GeV
  eta = generate_eta(rnd[:,1])          # Eta
  phi = generate_phi(rnd[:,2])          # Phi

  theta = 2.*atfi.atan(atfi.exp(-eta))     # Theta angle
  p  = pt/atfi.sin(theta)             # Full momentum
  e  = atfi.sqrt(p**2 + minit**2)     # Energy 

  px = p*atfi.sin(theta)*atfi.sin(phi)     # 3-momentum of initial particle
  py = p*atfi.sin(theta)*atfi.cos(phi)
  pz = p*atfi.cos(theta)

  p4 = atfk.lorentz_vector(atfk.vector(px, py, pz), e)  # 4-momentum of initial particle

  rotphi   = uniform_random(rnd[:,3], 0., 2*atfi.pi())
  rotpsi   = uniform_random(rnd[:,4], 0., 2*atfi.pi())
  rottheta = atfi.acos(uniform_random(rnd[:,5], -1, 1.))

  moms2 = []
  for m in moms : 
    m1 = atfk.rotate_lorentz_vector(m, rotphi, rottheta, rotpsi)
    moms2 += [ atfk.boost_from_rest(m1, p4) ]

  return moms2

def generate_combinatorial(cuts, rnd) : 
  """
    Generate random combinations of three tracks
  """
  meankpt  = cuts[0]
  meanpipt = cuts[1]
  ptcut    = cuts[2]

  p4pi1 = generate_4momenta(rnd[:,0:3], meanpipt, ptcut, atfi.const(mpi) )
  p4pi2 = generate_4momenta(rnd[:,3:6], meanpipt, ptcut, atfi.const(mpi) )
  p4k   = generate_4momenta(rnd[:,6:9], meankpt,  ptcut, atfi.const(mk) )

  return p4k, p4pi1, p4pi2

def generate_kstar(cuts, rnd) : 
  """
    Generate random combinations of Kstar and a pion track
  """
  meankstarpt = cuts[4]
  meanpipt  = cuts[1]
  ptcut     = cuts[2]

  mkstargen  = breit_wigner_random(rnd[:,0], mkstar, wkstar)

  ones = atfi.ones(rnd[:,0])
  p = atfk.two_body_momentum( mkstargen, mk*ones, mpi*ones )
  zeros = atfi.zeros(p)
  mom = [ atfk.lorentz_vector( atfk.vector(p, zeros, zeros), atfi.sqrt(p**2 + mk**2)), 
          atfk.lorentz_vector(-atfk.vector(p, zeros, zeros), atfi.sqrt(p**2 + mpi**2)) ]

  mom   = generate_rotation_and_boost(mom, mkstargen, meankstarpt, ptcut, rnd[:,2:8] )
  p4k = mom[0]
  p4pi1 = mom[1]
  p4pi2 = generate_4momenta(rnd[:,8:11], meanpipt, ptcut, atfi.const(mpi) )

  return p4k, p4pi1, p4pi2

def generate_rho(cuts, rnd) : 
  """
    Generate random combinations of rho -> pi pi  and a kaon track
  """
  meanrhopt = cuts[5]
  meankpt   = cuts[0]
  ptcut     = cuts[2]

  mrhogen  = breit_wigner_random(rnd[:,0], mrho, wrho)

  ones = atfi.ones(rnd[:,0])
  p = atfk.two_body_momentum( mrhogen, mpi*ones, mpi*ones )
  zeros = atfi.zeros(p)
  mom = [ atfk.lorentz_vector( atfk.vector(p, zeros, zeros), atfi.sqrt(p**2 + mpi**2)), 
          atfk.lorentz_vector(-atfk.vector(p, zeros, zeros), atfi.sqrt(p**2 + mpi**2)) ]

  mom   = generate_rotation_and_boost(mom, mrhogen, meanrhopt, ptcut, rnd[:,2:8] )
  p4k   = generate_4momenta(rnd[:,8:11], meankpt, ptcut, atfi.const(mk) )
  p4pi1 = mom[0]
  p4pi2 = mom[1]

  return p4k, p4pi1, p4pi2

def generate_selection(cuts, rnd, constant_cuts = False) : 
  """
    Call generation of fully combinatorial or combinatorial with intermediate K* or rho resonances with specified fractions. 
    Apply cuts to the final state particles and fill in output arrays. 
  """
  meankpt     = cuts[0]
  meanpipt    = cuts[1]
  ptcut       = cuts[2]
  pcut        = cuts[3]
  meankstarpt = cuts[4]
  meanrhopt   = cuts[5]
  kstarfrac   = cuts[6]
  rhofrac     = cuts[7]

  p4k_1, p4pi1_1, p4pi2_1 = generate_combinatorial(cuts, rnd)
  p4k_2, p4pi1_2, p4pi2_2 = generate_kstar(cuts, rnd)
  p4k_3, p4pi1_3, p4pi2_3 = generate_rho(cuts, rnd)

  thr1 = 1.-kstarfrac-rhofrac
  thr2 = 1.-rhofrac

  cond1 = atfi.stack(4*[rnd[:,11]<thr1], axis = 1)
  cond2 = atfi.stack(4*[rnd[:,11]<thr2], axis = 1)

  p4k   = tf.where(cond1, p4k_1, tf.where(cond2, p4k_2, p4k_3))
  p4pi1 = tf.where(cond1, p4pi1_1, tf.where(cond2, p4pi1_2, p4pi1_3))
  p4pi2 = tf.where(cond1, p4pi2_1, tf.where(cond2, p4pi2_2, p4pi2_3))

  mb = atfk.mass(p4k + p4pi1 + p4pi2)
  mfit, moms = kinematic_fit(atfi.const(md), [ p4k, p4pi1, p4pi2 ] )

  sel = tf.greater( atfk.p(moms[0]), pcut )
  sel = tf.logical_and(sel, tf.greater( atfk.p(moms[1]), pcut ) )
  sel = tf.logical_and(sel, tf.greater( atfk.p(moms[2]), pcut ) )
  sel = tf.logical_and(sel, tf.greater( atfk.pt(moms[0]), ptcut ) )
  sel = tf.logical_and(sel, tf.greater( atfk.pt(moms[1]), ptcut ) )
  sel = tf.logical_and(sel, tf.greater( atfk.pt(moms[2]), ptcut ) )

  m2kpi  = atfk.mass(moms[0] + moms[1])**2
  m2pipi = atfk.mass(moms[1] + moms[2])**2

  sample = dlz_phsp.from_vectors(m2kpi, m2pipi)
  mprime = dlz_phsp.m_prime_bc(sample)
  thetaprime = dlz_phsp.theta_prime_bc(sample)

  sel = tf.logical_and(sel, observables_phase_space.inside( tf.stack( [mprime, thetaprime, mb] , axis = 1) ) )

  observables = []
  outlist = [ mprime, thetaprime, mb, m2kpi, m2pipi ]

  if not constant_cuts : outlist += [ meankpt, meanpipt, ptcut, pcut, meankstarpt, meanrhopt, kstarfrac, rhofrac ]
  for i in outlist : 
    observables += [ tf.boolean_mask(i, sel) ]

  return observables

def generate_candidates_and_cuts(rnd) : 
  """
    Generate random cuts and call toy MC generation with those cuts to train parametric background model. 
  """

  cuts = []
  for i in range(len(bounds)) : 
    par = parameters_list[i][0]
    alpha = parameters_list[i][3]
    cuts += [ generate_exp(rnd[:,i], bounds[par][0][0], bounds[par][0][1], alpha) ]

  return generate_selection(cuts, rnd[:,len(bounds):])
