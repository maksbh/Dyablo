#pragma once

#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

namespace dyablo {

/**
 * @brief Structure grouping the primitive and conservative hydro state as well
 *        as information on the number of fields to store per state
 */
struct HydroState : public HyperbolicPolicy_State_Hydro
{
  using HyperbolicPolicy_State_Hydro::HyperbolicPolicy_State_Hydro;
  static constexpr size_t N = State_traits<PrimState>::nvars; 
};
using ConsHydroState = HydroState::ConsState;
using PrimHydroState = HydroState::PrimState;


template< int ndim, 
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getConservativeState(const Array_t& U, const CellIndex& iCell, ConsHydroState &res) 
{
  res = HydroState({ndim}).getConsState( U, iCell );
}

template< int ndim,
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getPrimitiveState(const Array_t& U, const CellIndex& iCell, PrimHydroState &res)
{
  res = HydroState({ndim}).getPrimState( U, iCell );
}

template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimHydroState u) {
  HydroState({ndim}).setPrimState( U, iCell, u );
}

template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setConservativeState( const Array_t& U, const CellIndex& iCell, ConsHydroState u) {
  HydroState({ndim}).setConsState( U, iCell, u );
}

template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void atomic_add_ConservativeState( const Array_t& U, const CellIndex& iCell, ConsHydroState u) {
  HydroState({ndim}).atomic_addConsState( U, iCell, u );
}

template<int ndim>
KOKKOS_INLINE_FUNCTION
PrimHydroState consToPrim(const ConsHydroState &U, real_t gamma0) {
  return HydroState({ndim, gamma0}).consToPrim( U );
}

/**
 * @brief Converts from a hydro primitive state to a hydro conservative state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param Q the initial primitive state
 * @param gamma0 adiabatic index
 * @return the conservative version of Q
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
ConsHydroState primToCons(const PrimHydroState &Q, real_t gamma0) {
  return HydroState({ndim, gamma0}).primToCons( Q );
}

} // namespace dyablo

