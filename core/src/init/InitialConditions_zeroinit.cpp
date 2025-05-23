#include "InitialConditions_base.h"

namespace dyablo{

class InitialConditions_zeroinit : public InitialConditions
{
public:
    std::vector<std::string> fields;

    InitialConditions_zeroinit(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
        : fields( configMap.getValue<std::vector<std::string>>( "zeroinit", "fields" ) )
    {}

    void init( UserData& U )
    {
        std::set<std::string> new_fields;
        for( const std::string& field : fields )
        {
            if( U.has_field(field) )
            {
                U.delete_field(field);
                std::cout << "WARNING : field " <<  field << " exists but will be overwritten by InitialConditions_zeroinit" << std::endl;
            }
            new_fields.insert(field);     
        }
        U.new_fields( new_fields );
    }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
    dyablo::InitialConditions_zeroinit, 
    "zeroinit");
