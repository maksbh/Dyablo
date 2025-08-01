#include "amr/MapUserData_base.h"
#include "amr/CellIndexRemapper.h"
#include "UserData.h"

namespace dyablo {

void MapUserData_base::save_old_mesh( UserData& U )
{
  this->lmesh_old = this->foreach_cell.get_amr_mesh().getLightOctree();
  this->ghost_comm_full = std::make_unique<GhostCommunicator_full_blocks>( this->foreach_cell.get_amr_mesh(), U.getShape(), -1 );
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

  this->ghost_comm_full->exchange_ghosts(Uin);

  remap_aux( Uin, Uout, remapper );

  // Deallocate fields_old before reallocating empty fields
  Uin = UserData::FieldAccessor();

  user_data.extend_fields();
}



} //namespace dyablo 