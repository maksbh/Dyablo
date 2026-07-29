#include "LightOctree_hashmap.h"

namespace dyablo{

namespace {

LightOctree_storage<>::pos_t get_first_pos( const LightOctree_storage<>& storage )
{
    using pos_t = LightOctree_storage<>::pos_t;
    Kokkos::View<real_t[3]> pos_device("pos");
    Kokkos::parallel_for("get_first_oct_corner", 1,
        KOKKOS_LAMBDA(int i)
    {
        auto p = storage.getCorner({(uint32_t)0,false});
        pos_device(0) = p[0];
        pos_device(1) = p[1];
        pos_device(2) = p[2];
    });
    auto pos_host = Kokkos::create_mirror_view(pos_device);
    Kokkos::deep_copy(pos_host, pos_device);
    return pos_t{pos_host(0),pos_host(1),pos_host(2)};
}

/**
 * this is here because KOKKOS_LAMBDAS cannot be declared in constructors or private methods
 **/
void private_init(const LightOctree_storage<>& storage, LightOctree_hashmap::oct_map_t& oct_map)
{
    uint32_t numOctants_tot = storage.getNumOctants()
                            + storage.getNumGhosts()
                            + storage.getNumIntermediates()
                            + storage.getNumIntermediateGhosts();

    uint32_t map_capacity = numOctants_tot;

    do
    {
        map_capacity *= 1.5;
        oct_map = LightOctree_hashmap::oct_map_t(map_capacity);

        // Put octants into hashmap on device
        Kokkos::parallel_for( "LightOctree_hashmap::hash",
                            numOctants_tot,
                            KOKKOS_LAMBDA(uint32_t ioct_local)
        { 
            LightOctree_hashmap::OctantIndex iOct = storage.iOctLocal_to_OctantIndex( ioct_local );

            auto logical_pos = storage.get_logical_coords( iOct );
            LightOctree_hashmap::level_t level = storage.getLevel( iOct );
            LightOctree_hashmap::key_t logical_coords;
            logical_coords.level = level;
            logical_coords.i = logical_pos[IX];
            logical_coords.j = logical_pos[IY];
            logical_coords.k = logical_pos[IZ];
            
            [[maybe_unused]] LightOctree_hashmap::oct_map_t::insert_result inserted = oct_map.insert( logical_coords, iOct );
            DYABLO_ASSERT_KOKKOS_DEBUG(!inserted.existing(), "oct_map::insert() failed : already exists!");
        });
    
    // Insert can fail and the algorithm (may) need to be restarted with increased capacity
    // (This seems to be very unlikely)
    // https://kokkos.org/kokkos-core-wiki/API/containers/Unordered-Map.html
    // > Maximum number of insert attempts: An insert can fail if no free space is found 
    // > in less than a certain number of (internal) attempts to insert. This can happen 
    // > independently of the capacity of the map.
    } while ( oct_map.failed_insert() != 0 );

    DYABLO_ASSERT_KOKKOS_DEBUG( oct_map.size() == numOctants_tot, "oct_map is missing octants" );
}

} // namespace

LightOctree_hashmap::LightOctree_hashmap( Storage_t&& storage, 
                      uint8_t level_min, uint8_t level_max,
                      Kokkos::Array<bool,3> periodic )
: storage( std::move(storage) ),
  min_level(level_min), max_level(level_max),
  is_periodic(periodic)
{
    private_init(this->storage, this->oct_map);
}

LightOctree_hashmap::LightOctree_hashmap( const AMRmesh* pmesh, uint8_t level_min, uint8_t level_max )
: storage( pmesh->getStorage().template deep_copy<Storage_t::MemorySpace>() ),
  min_level(level_min), max_level(level_max),
  is_periodic( {pmesh->getPeriodic(IX), pmesh->getPeriodic(IY), pmesh->getPeriodic(IZ)} ),
  morton_intervals( "morton_intervals", pmesh->getMpiComm().MPI_Comm_size()+1 )
{    
    private_init(this->storage, this->oct_map);

    // TODO use logical coords directly or even morton_intervals from AMRmesh
    morton_t first_morton;
    {
        int ndim = getNdim();
        pos_t pos = get_first_pos(storage);
        index_t<3> logical_coords;
        uint32_t octant_count = std::pow(2, max_level );
        real_t octant_size = 1.0/octant_count;
        logical_coords[IX] = std::floor(pos[IX]/octant_size);
        logical_coords[IY] = std::floor(pos[IY]/octant_size);
        logical_coords[IZ] = (ndim-2)*std::floor(pos[IZ]/octant_size);

        first_morton = compute_morton_key( logical_coords );
    }
    auto morton_intervals_host = Kokkos::create_mirror_view( morton_intervals );
    pmesh->getMpiComm().MPI_Allgather( &first_morton, morton_intervals_host.data(), 1 );
    morton_intervals_host(morton_intervals_host.size()-1) = uint64_t(-1);
    Kokkos::deep_copy(morton_intervals, morton_intervals_host);

}

} // namespace dyablo
