#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

#include "hyperbolic/scheme/Hyperbolic_euler.h"

namespace dyablo{

class HydroUpdate_euler 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro>::Hyperbolic_euler;
};



using HyperbolicPolicy_Hydro_impl_minmod_periodic = 
  HyperbolicPolicy_Hydro_impl< 
    HyperbolicPolicy_Slope_minmod<HyperbolicPolicy_State_Hydro>,
    HyperbolicPolicy_BoundaryConditions_PeriodicOnly<HyperbolicPolicy_State_Hydro>
  >;

using HydroUdpate_euler_minmod_periodic = Hyperbolic_euler< HyperbolicPolicy_base< HyperbolicPolicy_Hydro_impl_minmod_periodic >>;

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_euler, 
                  "HydroUpdate_euler")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUdpate_euler_minmod_periodic, 
                  "HydroUpdate_euler_minmod_periodic")