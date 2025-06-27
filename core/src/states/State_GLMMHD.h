#pragma once

#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

namespace dyablo {

/**
 * @brief Structure grouping the primitive and conservative GLMMHD state as well
 *        as information on the number of fields to store per state
 */
struct GLMMHDState : public HyperbolicPolicy_State_GLMMHD
{
  using HyperbolicPolicy_State_GLMMHD::HyperbolicPolicy_State_GLMMHD;
  static constexpr size_t N = State_traits<PrimState>::nvars; 
};
using ConsGLMMHDState = GLMMHDState::ConsState;
using PrimGLMMHDState = GLMMHDState::PrimState;


template< int ndim, 
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getConservativeState(const Array_t& U, const CellIndex& iCell, ConsGLMMHDState &res) 
{
  res = GLMMHDState({ndim}).getConsState( U, iCell );
}

template< int ndim,
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getPrimitiveState(const Array_t& U, const CellIndex& iCell, PrimGLMMHDState &res)
{
  res = GLMMHDState({ndim}).getPrimState( U, iCell );
}

template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimGLMMHDState u) {
  GLMMHDState({ndim}).setPrimState( U, iCell, u );
}

template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setConservativeState( const Array_t& U, const CellIndex& iCell, ConsGLMMHDState u) {
  GLMMHDState({ndim}).setConsState( U, iCell, u );
}

template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void atomic_add_ConservativeState( const Array_t& U, const CellIndex& iCell, ConsGLMMHDState u) {
  GLMMHDState({ndim}).atomic_addConsState( U, iCell, u );
}

template<int ndim>
KOKKOS_INLINE_FUNCTION
PrimGLMMHDState consToPrim(const ConsGLMMHDState &U, real_t gamma0) {
  return GLMMHDState({ndim, gamma0}).consToPrim( U );
}

/**
 * @brief Converts from a GLMMHD primitive state to a GLMMHD conservative state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param Q the initial primitive state
 * @param gamma0 adiabatic index
 * @return the conservative version of Q
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
ConsGLMMHDState primToCons(const PrimGLMMHDState &Q, real_t gamma0) {
  return GLMMHDState({ndim, gamma0}).primToCons( Q );
}

} // namespace dyablo


