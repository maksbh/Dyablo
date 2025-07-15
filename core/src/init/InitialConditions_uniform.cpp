#include "InitialConditions_uniform.h"

/* 
namespace dyablo {

//Example of a derived class using custom fields and values
class InitialConditions_zeroinit : public InitialConditions_uniform
{
public:
    InitialConditions_zeroinit(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
        :  InitialConditions_uniform(foreach_cell)
    {
        this->fields = configMap.getValue<std::vector<std::string>>( "InitialConditions_zeroinit", "fields" );
        this->values = std::vector<real_t>( this->fields.size() );
    }
};

} // namespace dyablo
*/

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
    dyablo::InitialConditions_uniform, 
    "uniform");
