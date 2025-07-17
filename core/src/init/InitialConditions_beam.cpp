#include "../InitialConditions_uniform.h"

namespace dyablo{

 class InitialConditions_beam : public InitialConditions_uniform{

    public:

    using SpawnRate = decltype( Units::mol()/Units::code_units().getUnit<Units::Time>() );

    InitialConditions_beam(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
        : InitialConditions_uniform(foreach_cell)
    {

        if( !configMap.hasValue("cosmology","ctilde") )
        {
            real_t clight_fraction = configMap.getValue_in_unit( "cosmology", "clight_fraction", Units::one(), "10%");
            real_t ctilde = clight_fraction * Units::constant_to_code_units( Units::SPEEDOFLIGHT() );
            configMap.getValue<real_t>("cosmology", "ctilde", ctilde);
        }
        
        real_t e_rad_start = configMap.getValue_in_code_unit<SpawnRate>("rad", "e_rad_start", "1e-10 atom/s");

        this->fields = { "rho","rho_HII","e_tot","rho_vx","rho_vy","rho_vz",
                        "e_rad","fx_rad","fy_rad","fz_rad"};

        this->values = {0, 0, 0, 0, 0, 0, e_rad_start, 0, 0, 0};
    }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_beam, 
                 "beam");
