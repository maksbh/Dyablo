#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_MHD.hpp"

namespace dyablo{

/**
 * Brio-Wu shock tube
 * 
 * Based on : Guillet et al. "High-order magnetohydrodynamics for astrophysics with an adaptive mesh refinement discontinuous Galerkin scheme", 2019
 *            MNRAS, Vol 445., No. 3.
 */
struct AnalyticalFormula_MHD_Brio_Wu : public AnalyticalFormula_base_MHD
{  
  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;
  const real_t xmin, ymin;
  const real_t xmax, ymax;
  const real_t xmid, ymid;
  const real_t rhoL, pL, ByL;
  const real_t rhoR, pR, ByR;
  const real_t Bx0, Bz0;

  AnalyticalFormula_MHD_Brio_Wu( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
    ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
    xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
    ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
    xmid(0.5*(xmin+xmax)),
    ymid(0.5*(ymin+ymax)),
    rhoL(configMap.getValue<real_t>("brio_wu", "rhoL", 1.0)),
    pL(configMap.getValue<real_t>("brio_wu", "pL", 1.0)),
    ByL(configMap.getValue<real_t>("brio_wu", "ByL", 1.0)),
    rhoR(configMap.getValue<real_t>("brio_wu", "rhoR", 0.125)),
    pR(configMap.getValue<real_t>("brio_wu", "pR", 0.1)),
    ByR(configMap.getValue<real_t>("brio_wu", "ByR", -1.0)),
    Bx0(configMap.getValue<real_t>("brio_wu", "Bx0", 0.75)),
    Bz0(configMap.getValue<real_t>("brio_wu", "Bz0", 0.0))
  {
    DYABLO_ASSERT_HOST_RELEASE(ndim == 2, "Initial conditions only for 2D");
  }

  using ConsState = HyperbolicPolicy_State_GLMMHD::ConsState;
  using PrimState = HyperbolicPolicy_State_GLMMHD::PrimState;

  KOKKOS_INLINE_FUNCTION
  ConsState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    PrimState q;
    if (x < xmid) {
      q.rho = rhoL;
      q.p = pL;
      q.By = ByL;
    }
    else {
      q.rho = rhoR;
      q.p = pR;
      q.By = ByR;
    }

    q.u = 0.0;
    q.v = 0.0;
    q.w = 0.0;
    q.Bx = Bx0;
    q.Bz = Bz0;
    q.psi = 0.0;

    return HyperbolicPolicy_State_GLMMHD({3, gamma0}).primToCons(q); 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_Brio_Wu>, 
                 "MHD_brio_wu");
