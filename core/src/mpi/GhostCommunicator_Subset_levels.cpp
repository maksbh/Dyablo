#include "GhostCommunicator_Subset_levels.hpp"

#include "foreach_cell/Binned_iOcts.hpp"

namespace dyablo{

struct Subset_levels::Pdata
{
    int level_max;
    const LightOctree& lmesh;
    Binned_iOcts_levels binned_iOcts_levels;

    template< typename GhostComm_t >
    using CacheMap_t = std::map< std::pair<bool, int>, 
                                 std::unique_ptr<typename GhostComm_t::OctSubset> >;
    std::tuple< CacheMap_t<GhostCommunicator_full_blocks>,
                CacheMap_t<GhostCommunicator_partial_blocks>
    > caches_tuple;
};


namespace{


/// Helper function for Subset_levels::getGhostCommunicatorSubset_level
template<typename GhostComm_t>
typename GhostComm_t::OctSubset getGhostCommunicatorSubset_level_impl(dyablo::Subset_levels::Pdata& pdata, int level, const GhostComm_t& comm )
{
    using Subset_t = typename GhostComm_t::OctSubset;

    bool intermediate = comm.has_intermediates();

    auto& cache = std::get<Subset_levels::Pdata::CacheMap_t<GhostComm_t>>(pdata.caches_tuple);
    auto key = std::make_pair( intermediate, level );    

    if( !cache.contains(key) )
    {
        const Kokkos::View<uint32_t*>& iOcts_send_full = comm.iOcts_send(); // Octants to send in full communicator (leaves or intermediates)
        
        const dyablo::LightOctree& lmesh = pdata.lmesh;
            
        int nbOcts = iOcts_send_full.size();
        auto gen_bin = KOKKOS_LAMBDA(uint32_t i)
        {
            uint32_t iOct = iOcts_send_full(i);
            return lmesh.getLevel( {iOct, false, intermediate} );
        };

        Binned_iOcts iOcts_send_binned = bin_iOcts( pdata.level_max, nbOcts, gen_bin); // Not iOcts directly, but indices from iOcts_send_full
        auto iOcts_send_level = get_bin( iOcts_send_binned, level );
        auto iOcts_recv_level = intermediate ? pdata.binned_iOcts_levels.get_iOcts_ghost_intermediates(level) : pdata.binned_iOcts_levels.get_iOcts_ghost_leaves(level);
        std::unique_ptr<Subset_t> subset = std::make_unique<Subset_t>( comm, iOcts_send_level, iOcts_recv_level );
        cache.emplace( key, std::move(subset) );
    }

    return *cache.at( key );
}
    
} //namespace

GhostCommunicator_full_blocks::OctSubset Subset_levels::getGhostCommunicatorSubset_level( int level, const GhostCommunicator_full_blocks& comm )
{
    return getGhostCommunicatorSubset_level_impl(*pdata, level, comm);
}

GhostCommunicator_partial_blocks::OctSubset Subset_levels::getGhostCommunicatorSubset_level( int level, const GhostCommunicator_partial_blocks& comm )
{
    return getGhostCommunicatorSubset_level_impl(*pdata, level, comm);
}

Subset_levels::Subset_levels(const LightOctree& lmesh, int level_max)
    : pdata( std::make_unique<Pdata>(
        Pdata
        {
            .level_max=level_max,
            .lmesh=lmesh,
            .binned_iOcts_levels = Binned_iOcts_levels(lmesh, level_max)
        }))
{/*empty*/}

Subset_levels::~Subset_levels() = default;

} // namespace dyablo