#pragma once

#include "UserData.h"

namespace dyablo {

/**
 * Ghost communicator for partial blocks (ghosts) 
 * then serialize/deserialize in Kokkos kernels and use CUDA-aware MPI 
 **/
class GhostCommunicator_partial_blocks
{
public:
    GhostCommunicator_partial_blocks( const AMRmesh& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape, uint32_t ghost_count, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() );
    ~GhostCommunicator_partial_blocks();

    static std::string name()
    {
      return "GhostCommunicator_partial_blocks";
    }

    /// @copydoc GhostCommunicator_base::getNumGhosts
    uint32_t getNumGhosts() const;

    void exchange_ghosts( const UserData::FieldAccessor& U ) const;
    void exchange_ghosts( const ForeachCell::CellArray_global_ghosted& U ) const;

    struct OctSubset
    {
      OctSubset(const GhostCommunicator_partial_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts );
    };

    void exchange_ghosts_subset( UserData::FieldAccessor& U, const OctSubset& subset ) const;

    void reduce_ghosts( UserData::FieldAccessor& U ) const;
    void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const; 

    struct Pdata;
private:
    std::unique_ptr<Pdata> pdata;  
};

} // namespace dyablo