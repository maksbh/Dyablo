#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_MHD.hpp"

namespace dyablo{

/**
 * @brief MHD Cloud-Shock interaction.
 * 
 * A strong shockwave interacts with a dense cloud.
 * 
 * Setup adapted from Bandopadhyay et al. "SADHANA: A Doubly Linked List-based Multidimensional Adaptive Mesh Refinement
 * Framework for Solving Hyperbolic Conservation Laws with Application to Astrophysical Hydrodynamics and Magnetohydrodynamics",
 * The Astrophysical Journal Supplement Series, Volume 263, Issue 2, id.32, 25 pp
 */
struct AnalyticalFormula_MHD_cloud_shock_interaction : public AnalyticalFormula_base_MHD
{
    using ConsState = HyperbolicPolicy_State_GLMMHD::ConsState;
    using PrimState = HyperbolicPolicy_State_GLMMHD::PrimState;

    const int ndim;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;    
    const real_t gamma0;

    const real_t xs;         // Initial position of the shock
    const real_t xc, yc, zc; // Cloud position
    const real_t rc;         // Cloud radius

    ConsState uleft, uright, ububble;

    
    AnalyticalFormula_MHD_cloud_shock_interaction( ConfigMap& configMap ) :
        ndim( configMap.getValue<int>("mesh", "ndim", 3) ),
        
        xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
        ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
        zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
        gamma0 ( configMap.getValue<real_t>("hydro","gamma0", 1.666666667) ),

        xs( configMap.getValue<real_t>("cloud_shock_interaction", "xs", 1.2) ),
        xc( configMap.getValue<real_t>("cloud_shock_interaction", "xc", 1.6) ),
        yc( configMap.getValue<real_t>("cloud_shock_interaction", "yc", 0.5) ),
        zc( configMap.getValue<real_t>("cloud_shock_interaction", "zc", 0.5) ),
        rc( configMap.getValue<real_t>("cloud_shock_interaction", "rc", 0.15) )
    {
        PrimState qleft, qright, qbubble;

        // Left of the shock
        qleft.rho =  3.86859;
        qleft.By  =  2.1826182;
        qleft.Bz  = -2.1826182;
        const real_t pmag_left = 0.5 * (qleft.Bx*qleft.Bx + qleft.By*qleft.By + qleft.Bz*qleft.Bz);
        qleft.p   = 167.345 - pmag_left; // Total pressure is given in the paper ...

        // Right of the shock
        qright.rho = 1.0;
        qright.u   = -11.2536;
        qright.By  =  0.56418959;
        qright.Bz  =  0.56418959;
        const real_t pmag_right= 0.5 * (qright.Bx*qright.Bx + qright.By*qright.By + qright.Bz*qright.Bz);
        qright.p   = 1.0 - pmag_right;

        // Bubble
        qbubble = qright;
        qbubble.rho = 10.0;

        HyperbolicPolicy_State_GLMMHD policy ({ndim, gamma0});
        uleft = policy.primToCons(qleft);
        uright = policy.primToCons(qright);
        ububble = policy.primToCons(qbubble);
    }


    KOKKOS_INLINE_FUNCTION
    ConsState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        const real_t ddx = x - xc;
        const real_t ddy = y - yc;
        const real_t ddz = z - zc;

        ConsState res;
        const real_t r2 = (ddx*ddx + ddy*ddy + (ndim == 2 ? 0.0 : ddz*ddz));
        if (r2 < rc*rc)
            res = ububble;
        else if (x < xs)
            res = uleft;
        else
            res = uright;

        
        return res; 
    } 
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_cloud_shock_interaction>, 
                 "MHD_cloud_shock_interaction");