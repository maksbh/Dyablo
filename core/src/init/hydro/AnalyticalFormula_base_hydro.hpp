#pragma once

#include <vector>

namespace dyablo
{

struct AnalyticalFormula_base_hydro
{
  struct State
  {
    real_t rho   = 0;
    real_t e_tot = 0;
    real_t rho_u = 0;
    real_t rho_v = 0;
    real_t rho_w = 0;
  };

  enum VarIndex : dyablo::VarIndex
  {
    Irho,
    Ie_tot,
    Irho_vx,
    Irho_vy,
    Irho_vz
  };

  std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo() const
  {
    return {
        {"rho"   , VarIndex::Irho},
        {"e_tot" , VarIndex::Ie_tot},
        {"rho_vx", VarIndex::Irho_vx},
        {"rho_vy", VarIndex::Irho_vy},
        {"rho_vz", VarIndex::Irho_vz},
    };
  }

  KOKKOS_INLINE_FUNCTION
  void setState(const UserData::FieldAccessor &Uout,
                const ForeachCell::CellIndex &iCell, const State &u) const
  {
    Uout.at(iCell, Irho)    = u.rho;
    Uout.at(iCell, Ie_tot)  = u.e_tot;
    Uout.at(iCell, Irho_vx) = u.rho_u;
    Uout.at(iCell, Irho_vy) = u.rho_v;
    Uout.at(iCell, Irho_vz) = u.rho_w;
  }
};

} // namespace dyablo