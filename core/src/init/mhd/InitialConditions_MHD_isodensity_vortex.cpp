#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_MHD.hpp"

namespace dyablo{

/**
 * @brief Isodensity vortex with magnetic fields.
 * 
 * Isodensity vortex with magnetic fields. The solution should be stationary, and so, when advected through a periodic box
 * it should return to it's original position without any change. Change is measured as an error on the solution
 * 
 * Setup adapted from Bandopadhyay et al. "SADHANA: A Doubly Linked List-based Multidimensional Adaptive Mesh Refinement
 * Framework for Solving Hyperbolic Conservation Laws with Application to Astrophysical Hydrodynamics and Magnetohydrodynamics",
 * The Astrophysical Journal Supplement Series, Volume 263, Issue 2, id.32, 25 pp
 */
struct AnalyticalFormula_MHD_isodensity_vortex : public AnalyticalFormula_base_MHD
{
    const int ndim;
    const real_t rho0, p0, v0, B0;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;    
    const real_t gamma0;

    real_t xmid, ymid, zmid;
    
    AnalyticalFormula_MHD_isodensity_vortex( ConfigMap& configMap ) :
        ndim( configMap.getValue<int>("mesh", "ndim", 3) ),
        
        rho0( configMap.getValue<real_t>("isodensity_vortex", "rho0", 1.0)),
        p0  ( configMap.getValue<real_t>("isodensity_vortex", "p0", 1.0)),
        v0  ( configMap.getValue<real_t>("isodensity_vortex", "v0", 1.0)),
        B0  ( configMap.getValue<real_t>("isodensity_vortex", "B0", 1.0)),
        
        xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
        ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
        zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
        gamma0 ( configMap.getValue<real_t>("hydro","gamma0", 1.666666667) )
    {
        xmid = 0.5 * (xmin+xmax);
        ymid = 0.5 * (ymin+ymax);
        zmid = 0.5 * (zmin+zmax);
    }

    using ConsState = HyperbolicPolicy_State_GLMMHD::ConsState;
    using PrimState = HyperbolicPolicy_State_GLMMHD::PrimState;

    KOKKOS_INLINE_FUNCTION
    ConsState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        const real_t rho = rho0;
        const real_t p   = p0;
        const real_t u   = v0;
        const real_t v   = v0;
        const real_t w   = 0.0;

        const real_t ddx = x-xmid;
        const real_t ddy = y-ymid;
        const real_t ddz = z-zmid;

        const real_t r2 = ddx*ddx+ddy*ddy+ddz*ddz;
       
        const real_t fac = Kokkos::exp((1.0-r2)*0.5) / (2.0*M_PI);
        const real_t du = -y*fac;
        const real_t dv =  x*fac;
        const real_t dw = 0.0;

        const real_t dp = -Kokkos::pow(x/M_PI*0.5, 2.0) * r2*0.5*Kokkos::exp(1.0-r2);

        const real_t dBx = -y*fac;
        const real_t dBy =  x*fac;
        const real_t dBz = 0.0;

        PrimState res;
        res.rho = rho;
        res.u   = u + du;
        res.v   = v + dv;
        res.w   = w + dw;
        res.p   = p + dp;
        res.Bx  = B0 + dBx;
        res.By  = B0 + dBy;
        res.Bz  = B0 + dBz;

        HyperbolicPolicy_State_GLMMHD policy ({ndim, gamma0});
        ConsState cons_res = policy.primToCons(res);
        return cons_res; 
    } 
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_isodensity_vortex>, 
                 "MHD_isodensity_vortex");