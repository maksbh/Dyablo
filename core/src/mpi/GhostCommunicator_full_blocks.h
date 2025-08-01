#pragma once

#include "UserData.h"
#include "mpi/ViewCommunicator.h"

namespace dyablo {


class GhostCommunicator_full_blocks : protected ViewCommunicator
{
public:
    template< typename AMRmesh_t >
    GhostCommunicator_full_blocks( const AMRmesh_t& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape,  int ghost_count, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    : ViewCommunicator( ViewCommunicator::from_mesh(amr_mesh, mpi_comm) )
    {}

    static std::string name()
    {
      return "GhostCommunicator_full_blocks";
    }
     
    /// @copydoc GhostCommunicator_base::getNumGhosts
    uint32_t getNumGhosts() const
    {
      return ViewCommunicator::getNumGhosts();
    }

    void exchange_ghosts( const UserData::FieldAccessor& U ) const;

    void exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const;

    void reduce_ghosts( UserData::FieldAccessor& U ) const;

    void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const; 
};

} // namespace dyablo