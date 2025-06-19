#include "InitialConditions_analytical.h"
#include "hydro/AnalyticalFormula_base_hydro.hpp"

#include "Cosmo.h"

namespace dyablo{

struct AnalyticalFormula_Zeldovitch_pancake : public AnalyticalFormula_base_hydro
{
  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;
  const real_t xmin, xmax;
  real_t omegab;
  real_t omegam;
  real_t omegav;
  real_t rho_fact;
  real_t dplus_ratio;
  real_t vfact;

  KOKKOS_INLINE_FUNCTION
  static int nrun_()
  {
    static int res = 0;
    return res++;
  }
  int nrun; 

  AnalyticalFormula_Zeldovitch_pancake( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    xmin( configMap.getValue<real_t>("mesh","xmin", 0) ),
    xmax( configMap.getValue<real_t>("mesh","xmax", 1) ),
    omegab(configMap.getValue<real_t>("cosmology","omegab", 0.049)),
    omegam(configMap.getValue<real_t>("cosmology","omegam", 0.3)),
    omegav(configMap.getValue<real_t>("cosmology","omegav", 0.7)),
    nrun( nrun_() )
  {
    DYABLO_ASSERT_HOST_RELEASE(ndim == 3, "Initial conditions only for 3D");

    real_t across = configMap.getValue<real_t>("zeldovitch_pancake", "aCross", 0.1);
    real_t astart = configMap.getValue<real_t>("cosmology", "aStart", 0.01);

    real_t omegam = this->omegam;
    real_t omegav = this->omegav;

    auto eta = [&omegam, &omegav](real_t a)
    {
      return sqrt(omegam/a+omegav*a*a+1.0-omegam-omegav);
    };

    auto dplus = [&eta]( real_t a )
    {
      auto ddplus = [&eta]( real_t a )
      {
        if( a==0 ) return 0.0;
        return 2.5 / (eta(a)*eta(a)*eta(a));
      };
      constexpr int max_iter = 20;
      real_t precision = 1e-10;
      real_t romberg_res = Impl::romberg<max_iter>(ddplus, 0, a, precision );
      return eta(a)/a*romberg_res;
    };

    auto dladt = [&eta]( real_t a )
    {
      return a*eta(a);
    };

    auto fomega = [&]( real_t a )
    {
      if (omegam==1.0 && omegav==0.0)
        return 1.0;
      
      real_t omegak = 1-omegam-omegav;
      return (2.5 / dplus(a) - 1.5 * omegam / a - omegak ) / (eta(a)*eta(a));
    };

    real_t H0 = configMap.getValue<real_t>("cosmology", "H0");
    real_t four_pi_G = configMap.getValue<real_t>("gravity", "4_pi_G");
    real_t Lbox = xmax-xmin;

    real_t rhoc = 3 * H0*H0 / (2*four_pi_G);

    this->rho_fact = omegab * rhoc;
    this->dplus_ratio = dplus(astart) / dplus(across);
    this->vfact = (this->dplus_ratio * Lbox / (2*M_PI) * fomega(astart)*dladt(astart) * H0);
  }

  KOKKOS_INLINE_FUNCTION
  State value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    HyperbolicPolicy_State_Hydro::PrimState res {};

    real_t A = this->dplus_ratio;
    auto newton_raphson_q = [&A](double x)
    {
      real_t eps = 1e-10;
      double q0=0.5;
      double q1=0.5;
      do
      {
        q0 = q1;
        real_t f = q0 + A/(2*M_PI) * sin( 2*M_PI*q0) - x;
        real_t f_prim = 1 + A * cos( 2*M_PI*q0 );
        q1 = q0 - f/f_prim;
      } while( fabs( q1 - q0 ) > eps );

      return q1;
    };

    real_t q;
    if(nrun == 1)
      q = newton_raphson_q(x/(xmax-xmin));
    else
      q = x/(xmax-xmin);

    res.rho = this->rho_fact/( 1 + this->dplus_ratio * cos(2*M_PI*q) );
    res.p   = this->smallp;
    res.u   = this->vfact * sin(2*M_PI*q);
    res.v   = 0.0;
    res.w   = 0.0;

    HyperbolicPolicy_State_Hydro policy ({ndim, gamma0});
    State cons_res = policy.primToCons(res);
    return cons_res;
  }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_Zeldovitch_pancake>, 
                 "zeldovitch_pancake");

