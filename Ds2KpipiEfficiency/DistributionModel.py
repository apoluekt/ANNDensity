import math, sys

import amplitf.interface as atfi
import amplitf.kinematics as atfk
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr

md  = 1.96834
mpi = 0.13957
mk  = 0.49368

etamin = 2. 
etamax = 5.
meanpt = 1.

dalitz_phase_space = DalitzPhaseSpace( mk, mpi, mpi, md )
observables_phase_space = RectangularPhaseSpace( ((0., 1.), (0., 1.)) )
observables_titles = [ r"$m'$", r"$\theta'$" ]
observables_data = [ "mprime", "thetaprime" ]
observables_toys = [ "mprime", "thetaprime" ]

parameters_list = [
  ("ptcut",     r"Track $p_{T}$ cut (GeV)",       (0.1, 1.), -1.0), 
  ("pcut",      r"Track $p$ cut (GeV)",           (1., 10.), -10.),
  ("dptcut",    r"$D^{+}_{s}$ $p_{T}$ cut (GeV)",   (0.,  5.),  1.3), 
  ("maxptcut",  r"max $p_{T}$ cut (GeV)",         (0.5, 3.), -2.5), 
  ("sumptcut",  r"sum $p_{T}$ cut (GeV)",         (2.5, 6.), -2.5), 
]

true_cuts = [ atfi.const(0.4), atfi.const(3.), atfi.const(2.0), atfi.const(1.), atfi.const(3.) ]

random_array_size = 11

bounds = { i[0] : (i[2], i[3]) for i in parameters_list }

def uniform_random(rnd, x1, x2) : 
  """
    Uniform random numbers from x1 to x2
  """
  if isinstance(x1, float) : x1 = atfi.const(x1)
  if isinstance(x2, float) : x2 = atfi.const(x2)
  return x1 + rnd*(x2-x1)

def generate_exp(rnd, x1, x2, alpha = None) : 
  """
    Exponential random distribution with constant "alpha", 
    limited to the range x1 to x2
  """
  if isinstance(x1, float) : x1 = atfi.const(x1)
  if isinstance(x2, float) : x2 = atfi.const(x2)
  if alpha is None or alpha == 0 : 
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

def generate_rotation_and_boost(moms, minit, meanpt, ptcut, rnd) : 
  """
    Generate 4-momenta of final state products according to 3-body phase space distribution
      moms   - initial particle momenta (in the rest frame)
      meanpt - mean Pt of the initial particle
      ptcut  - miminum Pt of the initial particle
      rnd    - Auxiliary random tensor
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

def selection(sample, cuts, rnd, constant_cuts = False) : 
    ptcut     = cuts[0]
    pcut      = cuts[1]
    d_ptcut   = cuts[2]
    max_ptcut = cuts[3]
    sum_ptcut = cuts[4]

    dalitz_sample = dalitz_phase_space.from_square_dalitz_plot(sample[:,0], sample[:,1])

    # Random momenta for a given Dalitz plot sample
    mom = dalitz_phase_space.final_state_momenta( 
             dalitz_phase_space.m2ac(dalitz_sample),   # Note AB<->AC because FroSquareDalitzPlot works with AC
             dalitz_phase_space.m2bc(dalitz_sample) 
          )  # Generate momenta in a frame of the decaying particle

    mom = generate_rotation_and_boost(mom, md, meanpt, d_ptcut, rnd[:,:]) # Rotate and boost to the lab frame

    # Apply cuts
    sel = atfi.greater(atfk.pt(mom[0]), ptcut)
    sel = atfi.logical_and(sel, atfi.greater(atfk.pt(mom[1]), ptcut))
    sel = atfi.logical_and(sel, atfi.greater(atfk.pt(mom[2]), ptcut))
    sel = atfi.logical_and(sel, atfi.greater(atfk.pt(mom[0]+mom[1]+mom[2]), d_ptcut))
    sel = atfi.logical_and(sel, atfi.greater(atfk.p(mom[0]), pcut))
    sel = atfi.logical_and(sel, atfi.greater(atfk.p(mom[1]), pcut))
    sel = atfi.logical_and(sel, atfi.greater(atfk.p(mom[2]), pcut))
    sel = atfi.logical_and(sel, atfi.greater(atfi.max(atfi.max(atfk.pt(mom[0]), atfk.pt(mom[1])), atfk.pt(mom[2])), max_ptcut))
    sel = atfi.logical_and(sel, atfi.greater(atfk.pt(mom[0])+atfk.pt(mom[1])+atfk.pt(mom[2]), sum_ptcut))
    sel = atfi.logical_and(sel, atfi.greater(atfk.eta(mom[0]), etamin))
    sel = atfi.logical_and(sel, atfi.greater(atfk.eta(mom[1]), etamin))
    sel = atfi.logical_and(sel, atfi.greater(atfk.eta(mom[2]), etamin))
    sel = atfi.logical_and(sel, atfi.less(atfk.eta(mom[0]), etamax))
    sel = atfi.logical_and(sel, atfi.less(atfk.eta(mom[1]), etamax))
    sel = atfi.logical_and(sel, atfi.less(atfk.eta(mom[2]), etamax))

    m2ab = atfk.mass(mom[0]+mom[1])**2
    m2bc = atfk.mass(mom[1]+mom[2])**2

    mprime = sample[:,0]
    thetaprime = sample[:,1]

    arrays = [] 
    outlist = [ mprime, thetaprime ]
    if not constant_cuts : outlist += [ ptcut, pcut, d_ptcut, max_ptcut, sum_ptcut ]
    for i in outlist : 
      arrays += [ i[sel] ]
    return arrays

def selection_with_random_cuts(sample, rnd) : 
    """
      Function to generate random cuts, compute final state momenta from the Dalitz plot sample, and run 
      the generated events through selection.
    """
    ptcut      = generate_exp(rnd[:,0], bounds['ptcut'][0][0],    bounds['ptcut'][0][1],    bounds['ptcut'][1]  )
    pcut       = generate_exp(rnd[:,1], bounds['pcut'][0][0],     bounds['pcut'][0][1],     bounds['pcut'][1]  )
    d_ptcut    = generate_exp(rnd[:,2], bounds['dptcut'][0][0],   bounds['dptcut'][0][1],   bounds['dptcut'][1]  )
    max_ptcut  = generate_exp(rnd[:,3], bounds['maxptcut'][0][0], bounds['maxptcut'][0][1], bounds['maxptcut'][1] )
    sum_ptcut  = generate_exp(rnd[:,4], bounds['sumptcut'][0][0], bounds['sumptcut'][0][1], bounds['sumptcut'][1] )

    cuts = [ ptcut, pcut, d_ptcut, max_ptcut, sum_ptcut ]

    return selection(sample, cuts, rnd[:,5:])
