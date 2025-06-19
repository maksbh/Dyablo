#pragma once

#include <vector>
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

namespace dyablo
{

struct AnalyticalFormula_base_hydro
{
  using ConsHydroState = HyperbolicPolicy_State_Hydro::ConsState;
  using State = ConsHydroState;

  std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo() const
  {
    return State::getFieldsInfo();
  }

  KOKKOS_INLINE_FUNCTION
  void setState(const UserData::FieldAccessor &Uout,
                const ForeachCell::CellIndex &iCell, const State &u) const
  {
    HyperbolicPolicy_State_Hydro({3}).setConsState(Uout, iCell, u);
  }
};

} // namespace dyablo