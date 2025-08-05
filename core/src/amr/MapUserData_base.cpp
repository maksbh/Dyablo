#include "amr/MapUserData_base.h"
#include "amr/CellIndexRemapper.h"
#include "mpi/GhostCommunicator_full_blocks.h"
#include "UserData.h"

namespace dyablo {

void MapUserData_base::save_old_mesh()
{
  this->lmesh_old = this->foreach_cell.get_amr_mesh().getLightOctree();
}

void MapUserData_base::remap( UserData& user_data )
{ 
  CellIndexRemapper remapper( this->lmesh_old, this->foreach_cell );

  UserData::FieldAccessor Uin = user_data.backup_and_realloc();
  UserData::FieldAccessor Uout;
  {
    std::vector<UserData::FieldAccessor::FieldInfo> all_fields;
    int i=0;
    for( const std::string& field : user_data.getEnabledFields() )
      all_fields.push_back({field, i++});
    Uout = user_data.getAccessor( all_fields );
  }

  remap_aux( Uin, Uout, remapper );

  // Deallocate fields_old before reallocating empty fields
  Uin = UserData::FieldAccessor();

  user_data.extend_fields();
}



} //namespace dyablo 