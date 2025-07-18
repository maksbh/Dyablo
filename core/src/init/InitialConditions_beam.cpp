#include "InitialConditions_uniform.h"

namespace dyablo{

 class InitialConditions_beam : public InitialConditions_uniform{

    public:

    InitialConditions_beam(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
        : InitialConditions_uniform(foreach_cell)
    {

        if( !configMap.hasValue("cosmology","ctilde") )
        {
            real_t clight_fraction = configMap.getValue_in_unit( "cosmology", "clight_fraction", Units::one(), "10%");
            real_t astart = configMap.getValue<real_t>("cosmology", "astart", 1.0);
            real_t ctilde = clight_fraction * Units::constant_to_code_units( Units::SPEEDOFLIGHT() ) * astart;

            configMap.getValue<real_t>("cosmology", "ctilde", ctilde);
        }
        
        this->fields = { "rho","rho_HII","e_tot","rho_vx","rho_vy","rho_vz",
                        "e_rad","fx_rad","fy_rad","fz_rad"};

        this->values = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_beam, 
                 "beam");
