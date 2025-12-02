#pragma once

#include "UserData.h"
#include "mpi/ViewCommunicator.h"

namespace dyablo {

class GhostCommunicator_full_blocks : protected ViewCommunicator
{
private:
    bool intermediates;
public:
    template< typename AMRmesh_t >
    GhostCommunicator_full_blocks( const AMRmesh_t& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape,  int ghost_count, bool intermediates=false, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    : ViewCommunicator( ViewCommunicator::from_mesh(amr_mesh, intermediates, mpi_comm) ),
      intermediates(intermediates)
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
    void exchange_ghosts( const UserData::FieldAccessor_intermediates& U ) const;

    void exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const;

    struct OctSubset
    {
      OctSubset(const GhostCommunicator_full_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts );

      uint32_t nbGhosts() const
      {
        return partial_comm->getNumGhosts();
      }
    
      std::unique_ptr<ViewCommunicator> partial_comm;
      Kokkos::View<uint32_t*> subset_iOcts;
    };

    void exchange_ghosts_subset( const UserData::FieldAccessor& U, const OctSubset& subset ) const;
    void exchange_ghosts_subset( const UserData::FieldAccessor_intermediates& U, const OctSubset& subset ) const;

    void reduce_ghosts( UserData::FieldAccessor& U ) const;
    void reduce_ghosts( UserData::FieldAccessor_intermediates& U ) const;

    void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const; 
};

} // namespace dyablo