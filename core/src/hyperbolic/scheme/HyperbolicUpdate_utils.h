#pragma once

#include "utils_hydro.h"
#include "RiemannSolvers.h"
#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "boundary_conditions/BoundaryConditions.h"

namespace dyablo {

namespace {

struct GravityInfo {
  real_t gx, gy, gz;
  bool apply_gravity_in_step;
  bool well_balanced;
};
}

/**
 * @brief Computes the primitive variables at a given cell index in an array
 *        and stores it in another array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array passed to the method
 * @tparam CellIndex the type of index passed to the method
 * @param params the hydro parameters for the conversion
 * @param Ugroup The array where the conservative variables are stored
 * @param iCell_Ugroup the index where to look and where to store
 * @param Qgroup the array where the primitive variables are written
 * 
 * @note Ugroup and Qgroup should have the same sizes and properties
 */
template< 
  int ndim, 
  typename State,
  typename Array_t, 
  typename CellIndex >
KOKKOS_INLINE_FUNCTION
void compute_primitives(const RiemannParams& params,   const Array_t& Ugroup, 
                        const CellIndex& iCell_Ugroup, const Array_t& Qgroup)
{
  using ConsState = typename State::ConsState;
  using PrimState = typename State::PrimState;

  ConsState uLoc{};
  getConservativeState<ndim>(Ugroup, iCell_Ugroup, uLoc);
  PrimState qLoc = consToPrim<ndim>(uLoc, params.gamma0);
  setPrimitiveState<ndim>(Qgroup, iCell_Ugroup, qLoc);
}

/**
 * @brief Computes the slope for a given state according to given neighbors
 * 
 * @tparam ndim the number of dimensions
 * @param qMinus_ the "left" neighbor considered
 * @param q_ the current cell state
 * @param qPlus_ the "right" neighbor considered
 * @param dL the relative distance separating q_ from qMinus_
 * @param dR the relative distance separating q_ from qPlus_
 * @return the slope for each variable of q_
 * 
 * @note the relative distance is expressed in units of the size of the current cell.
 *       hence, if the neighbor has the same size, the distance will be 1.0. If the neighbor
 *       is bigger, it will be 3/2 and if it is smaller it will be 3/4.
 */
template< int ndim, typename PrimState >
KOKKOS_INLINE_FUNCTION
PrimState compute_slope( const PrimState& qMinus, 
                         const PrimState& q, 
                         const PrimState& qPlus, 
                         real_t dL, real_t dR)
{
  auto dqp = (qPlus - q)  / dR;
  auto dqm = (q - qMinus) / dL;
  
  PrimState dq{};
  state_foreach_var( [](real_t& res, real_t dvp, real_t dvm) {
    if (dvp * dvm <= 0.0)
      res = 0.0;
    else
      res = fabs(dvp) > fabs(dvm) ? dvm : dvp;
  }, dq, dqp, dqm);

  return dq;
}
/**
   * @brief Computes the source term from the muscl-hancock algorithm
   *        Specialized for Hydro states
   * 
   * @tparam ndim The number of dimensions
   * @param q Centered primitive variable
   * @param slopeX Slopes along each direction
   * @param slopeY 
   * @param slopeZ 
   * @param dtdx time step over space step along each direction
   * @param dtdy 
   * @param dtdz 
   * @param gamma adiabatic index
   * @return A primitive state corresponding to the half-step evolved variable in the cell
   * 
   * @todo Adapt this to any PrimState possible !
   */
  template<int ndim >
  KOKKOS_INLINE_FUNCTION
  PrimHydroState compute_source( const PrimHydroState& q,
                                 const PrimHydroState& slopeX,
                                 const PrimHydroState& slopeY,
                                 const PrimHydroState& slopeZ,
                                 real_t dtdx, real_t dtdy, real_t dtdz,
                                 real_t gamma )
  {
    // retrieve primitive variables in current quadrant
    const real_t r = q.rho;
    const real_t p = q.p;
    const real_t u = q.u;
    const real_t v = q.v;
    const real_t w = q.w;

    // retrieve variations = dx * slopes
    const real_t drx = slopeX.rho * 0.5;
    const real_t dpx = slopeX.p   * 0.5;
    const real_t dux = slopeX.u   * 0.5;
    const real_t dvx = slopeX.v   * 0.5;
    const real_t dwx = slopeX.w   * 0.5;    
    const real_t dry = slopeY.rho * 0.5;
    const real_t dpy = slopeY.p   * 0.5;
    const real_t duy = slopeY.u   * 0.5;
    const real_t dvy = slopeY.v   * 0.5;
    const real_t dwy = slopeY.w   * 0.5;    
    const real_t drz = slopeZ.rho * 0.5;
    const real_t dpz = slopeZ.p   * 0.5;
    const real_t duz = slopeZ.u   * 0.5;
    const real_t dvz = slopeZ.v   * 0.5;
    const real_t dwz = slopeZ.w   * 0.5;

    PrimHydroState source{};
    if( ndim == 3 )
    {
      source.rho = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy + (-w * drz - dwz * r) * dtdz;
      source.u   = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy + (-w * duz) * dtdz;
      source.v   = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy + (-w * dvz) * dtdz;
      source.w   = w + (-u * dwx) * dtdx + (-v * dwy) * dtdy + (-w * dwz - dpz / r) * dtdz;
      source.p   = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy + (-w * dpz - dwz * gamma * p) * dtdz;
    }
    else
    {
      source.rho = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
      source.u   = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
      source.v   = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
      source.p   = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;
    }
    return source;
  }
  
}
