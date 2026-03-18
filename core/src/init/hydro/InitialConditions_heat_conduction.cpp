#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_hydro.hpp"

namespace dyablo{

/**
 * Heat conduction test in 1D
 * Based on :
 *  . Navarro et al, "Modeling the thermal conduction in the solar atmosphere with code MANCHA3D", 2022, A&A
 *  . Rempel et al, "Extension of the MURAM radiative MHD code for coronal simulations", 2016, ApJ
 */
struct AnalyticalFormula_heat_conduction : public AnalyticalFormula_base_hydro {
  const int    ndim;
  const real_t gamma0;

  AnalyticalFormula_heat_conduction( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4))
  {
  }

  KOKKOS_INLINE_FUNCTION
  State value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    const real_t rho = 1.0;
    const real_t T   = 0.1 + 0.9*Kokkos::pow(x, 5.0);
    const real_t p   = rho * T;

    State res;
    res.rho   = rho;
    res.rho_u = 0.0;
    res.rho_v = 0.0;
    res.rho_w = 0.0;
    res.e_tot = p / (gamma0-1.0);
    return res;
  }
};
} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_heat_conduction>, 
                 "heat_conduction");
