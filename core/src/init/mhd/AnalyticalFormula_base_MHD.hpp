#pragma once

#include <vector>
#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

namespace dyablo
{

struct AnalyticalFormula_base_MHD
{
  using State = HyperbolicPolicy_State_GLMMHD::ConsState;

  std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo() const
  {
    return State::getFieldsInfo();
  }

  KOKKOS_INLINE_FUNCTION
  void setState(const UserData::FieldAccessor &Uout,
                const ForeachCell::CellIndex &iCell, const State &u) const
  {
    HyperbolicPolicy_State_GLMMHD({3}).setConsState(Uout, iCell, u);
  }
};

} // namespace dyablo