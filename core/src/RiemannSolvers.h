/**
 * \file RiemannSolvers.h
 * All possible Riemann solvers or so.
 */
#ifndef RIEMANN_SOLVERS_H_
#define RIEMANN_SOLVERS_H_

#include <math.h>
#include "states/State_Nd.h"
#include "states/State_forward.h"

namespace dyablo {

//! Riemann solver type for hydro fluxes
enum RiemannSolverType {
  RIEMANN_HLLC,       /*!< HLLC hydro-only Riemann solver */
  RIEMANN_HLLD        /*!< HLLD MHD-only Riemann solver */
};

} // namespace dyablo

template<>
inline named_enum<dyablo::RiemannSolverType>::init_list named_enum<dyablo::RiemannSolverType>::names()
{
  return{
    {dyablo::RiemannSolverType::RIEMANN_HLLC,       "hllc"},
    {dyablo::RiemannSolverType::RIEMANN_HLLD,       "hlld"}
  };
}

namespace dyablo {

struct RiemannParams
{
  RiemannParams( ConfigMap& configMap )
  : riemannSolverType( configMap.getValue<RiemannSolverType>("hydro","riemann", RIEMANN_HLLC)),
    gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
    gamma6( (gamma0 + 1) / (2*gamma0)),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallp( configMap.getValue<real_t>("hydro","smallp", 1e-10) ),
    smalle( configMap.getValue<real_t>("hydro","smalle", 1e-5) ),
    smallc( sqrt(smallp*gamma0/smallr) ),
    smallpp( smallr*smallp ),
    three_waves( configMap.getValue<bool>("hydro", "three_waves", false))
  { 
    const bool use_glm = configMap.getValue<std::string>("hydro", "update", "HyperbolicUpdate_hancock").find("GLMMHD") != std::string::npos;
    if (use_glm) {
      const uint32_t ndim = configMap.getValue<uint32_t>("mesh", "ndim", 3);
      const real_t xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0);
      const real_t xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0);
      const real_t ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0);
      const real_t ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0);
      const real_t zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0);
      const real_t zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0);

      const uint32_t level_max = configMap.getValue<uint32_t>("amr", "level_max", 10);

      const uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 0);
      const uint32_t by = configMap.getValue<uint32_t>("amr", "by", 0);
      const uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", 1);

      const real_t Lx = xmax - xmin;
      const real_t Ly = ymax - ymin;
      const real_t Lz = zmax - zmin; 

      const real_t min_dx = Lx / ((1 << level_max) * bx);
      const real_t min_dy = Ly / ((1 << level_max) * by);
      const real_t min_dz = Lz / ((1 << level_max) * bz); 

      real_t min_dh = FMIN(min_dx, min_dy);
      if (ndim == 3)
        min_dh = FMIN(min_dh, min_dz);

      const real_t cfl = configMap.getValue<real_t>("hydro", "cfl", 0.5);

      ch = 0.5*cfl * min_dh; // This value must be divided by dt in the kernels !!!
      cr = configMap.getValue<real_t>("hydro", "cr", 0.1);
    }

  }

  RiemannSolverType riemannSolverType;

  real_t gamma0;
  real_t gamma6;
  real_t smallr;
  real_t smallp;
  real_t smalle;
  real_t smallc;
  real_t smallpp;

  bool three_waves;
  real_t ch, cr;
};

} // namespace dyablo

#endif // RIEMANN_SOLVERS_H_
