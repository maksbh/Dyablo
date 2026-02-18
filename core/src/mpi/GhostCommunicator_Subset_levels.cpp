#include "GhostCommunicator_Subset_levels.hpp"

#include "foreach_cell/Binned_iOcts.hpp"

namespace dyablo{

struct Subset_levels::Pdata
{
    Binned_iOcts_levels binned_iOcts_levels;
};

} // namespace dyablo

namespace{

/**
 * Helper function for Subset_levels::getGhostCommunicatorSubset_level
 * 
 * TODO : Optimize implementation to avoid mpi communications to get octs to send in OctSubset instanciation
 */
template<typename GhostComm_t>
typename GhostComm_t::OctSubset getGhostCommunicatorSubset_level_impl(dyablo::Subset_levels::Pdata& pdata, int level, const GhostComm_t& comm )
{
    if( comm.has_intermediates() )
        return typename GhostComm_t::OctSubset( comm, pdata.binned_iOcts_levels.get_iOcts_ghost_intermediates(level) );
    else
        return typename GhostComm_t::OctSubset( comm, pdata.binned_iOcts_levels.get_iOcts_ghost_leaves(level) );
}
    
} //namespace

namespace dyablo{

GhostCommunicator_full_blocks::OctSubset Subset_levels::getGhostCommunicatorSubset_level( int level, const GhostCommunicator_full_blocks& comm )
{
    return getGhostCommunicatorSubset_level_impl(*pdata, level, comm);
}

GhostCommunicator_partial_blocks::OctSubset Subset_levels::getGhostCommunicatorSubset_level( int level, const GhostCommunicator_partial_blocks& comm )
{
    return getGhostCommunicatorSubset_level_impl(*pdata, level, comm);
}

Subset_levels::Subset_levels(const LightOctree& lmesh, int level_max)
    : pdata( std::make_unique<Pdata>(Pdata{.binned_iOcts_levels = Binned_iOcts_levels(lmesh, level_max)}) )
{/*empty*/}

Subset_levels::~Subset_levels() = default;

} // namespace dyablo