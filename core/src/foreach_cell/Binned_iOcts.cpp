#include "Binned_iOcts.hpp"

#include "foreach_cell/ForeachCell.h"
#include "amr/LightOctree.h"

namespace dyablo {

namespace {

struct Binned_iOcts{
    Kokkos::View<uint32_t*> iOcts; /// All iOcts binned by level
    std::vector<uint32_t> begin_v; /// first index for bin `level` in iOcts;
};

/***
 * Sort iOcts to gather octants from the same bin together
 * @param bin_max number of bins (gen_bin must return value between 0 and bin_max)
 * @param nbOcts the number of octants to sort (gen_bin will be called from 0 to nbOcts)
 * @param gen_bin is a uint32_t -> uint32_t function that returns the bin number for a given iOct
 */
template<typename F>
Binned_iOcts bin_iOcts( int bin_max, int nbOcts, const F& gen_bin)
{
    int bin_count = bin_max + 1;
    Kokkos::View<uint32_t*> bin_count_device("level_count",bin_count);
    Kokkos::parallel_for( "bin_iOcts_count", nbOcts, 
        KOKKOS_LAMBDA( uint32_t iOct )
    {
        int bin = gen_bin(iOct);
        DYABLO_ASSERT_KOKKOS_DEBUG( 0 <= bin && bin <= bin_max, "bin_iOcts : gen_bin out of bounds" );

        Kokkos::atomic_inc( &bin_count_device(bin) );
    });

    auto bin_count_host = Kokkos::create_mirror_view( bin_count_device );
    Kokkos::deep_copy( bin_count_host, bin_count_device );

    Kokkos::View<uint32_t*> iOcts("binned_iOcts", nbOcts);
    std::vector<uint32_t> begin_v(bin_count+1);
    uint32_t bin_begin = 0;
    for( int bin = 0; bin <= bin_max; bin++ )
    {
        begin_v[bin] = bin_begin;
        Kokkos::parallel_scan( "bin_iOcts_sort", nbOcts,
            KOKKOS_LAMBDA( uint32_t iOct, int& i_local, bool final )
        {
            int oct_bin = gen_bin(iOct);
            if( oct_bin == bin )
            {
                if( final )
                   iOcts(bin_begin + i_local) = iOct; 
                i_local++;
            }
        });
        bin_begin += bin_count_host(bin);
    }
    begin_v[bin_max+1] = bin_begin;

    return Binned_iOcts{
        .iOcts = iOcts,
        .begin_v = begin_v,
    };
}

/// Get a subview from all iOcts in a given bin
Kokkos::View<uint32_t*> get_bin( const Binned_iOcts& iOcts, int bin )
{
    DYABLO_ASSERT_HOST_DEBUG( bin+1 < iOcts.begin_v.size(), "Subset_levels::get_bin : bin out of bounds" );

    return Kokkos::subview( iOcts.iOcts, std::make_pair( iOcts.begin_v.at(bin), iOcts.begin_v.at(bin+1) ) );
}

} // namespace

struct Binned_iOcts_levels::Pdata
{
    const LightOctree& lmesh;
    int level_max;
    std::unique_ptr<Binned_iOcts> local_leaves;
    std::unique_ptr<Binned_iOcts> ghost_leaves;
    std::unique_ptr<Binned_iOcts> local_intermediates;
    std::unique_ptr<Binned_iOcts> ghost_intermediates;
};

namespace {

/// Helper function for get_iOcts_xxx
template< bool ghost, bool intermediate >
Kokkos::View<uint32_t*> get_iOcts( const Binned_iOcts_levels::Pdata& pdata, std::unique_ptr<Binned_iOcts>& binned_iOcts, uint32_t nbOcts, int level )
{
    if( !binned_iOcts )
    {
        auto& lmesh = pdata.lmesh;
        auto gen_bin = KOKKOS_LAMBDA( uint32_t iOct )
        {
            return lmesh.getLevel({iOct, ghost, intermediate});
        };
        binned_iOcts = std::make_unique<Binned_iOcts>( 
            bin_iOcts( pdata.level_max, nbOcts, gen_bin)
        );
    }
    return get_bin( *binned_iOcts, level );
}

} // namespace

Binned_iOcts_levels::Binned_iOcts_levels(const LightOctree& lmesh, int level_max)
    : pdata(std::make_unique<Binned_iOcts_levels::Pdata>(
                    Binned_iOcts_levels::Pdata{
                        .lmesh=lmesh, 
                        .level_max=level_max,
                    }))
{/*empty*/}

Binned_iOcts_levels::Binned_iOcts_levels(Binned_iOcts_levels&&) = default;

Binned_iOcts_levels::~Binned_iOcts_levels()
{/*empty*/}

Kokkos::View<uint32_t*> Binned_iOcts_levels::get_iOcts_leaves(int level)
{ 
    return get_iOcts<false, false>( *pdata, pdata->local_leaves, pdata->lmesh.getNumOctants(), level );
}

Kokkos::View<uint32_t*> Binned_iOcts_levels::get_iOcts_ghost_leaves(int level)
{   
    return get_iOcts<true, false>( *pdata, pdata->ghost_leaves, pdata->lmesh.getNumGhosts(), level );
}

Kokkos::View<uint32_t*> Binned_iOcts_levels::get_iOcts_intermediates(int level)
{
    return get_iOcts<false, true>( *pdata, pdata->local_intermediates, pdata->lmesh.getNumIntermediates(), level );
}

Kokkos::View<uint32_t*> Binned_iOcts_levels::get_iOcts_ghost_intermediates(int level)
{
    return get_iOcts<true, true>( *pdata, pdata->ghost_intermediates, pdata->lmesh.getNumIntermediateGhosts(), level );
}


} // namespace dyablo