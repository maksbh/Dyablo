#pragma once

#include "amr/LightOctree_hashmap.h"

namespace dyablo { 

namespace{
KOKKOS_INLINE_FUNCTION 
static uint32_t offset_to_index(const LightOctree_base::offset_t& offset, uint32_t ndims) 
{
    return offset[IX]+1 + 3*(offset[IY]+1) + (ndims==3)*3*3*(offset[IZ]+1);
}

KOKKOS_INLINE_FUNCTION 
static LightOctree_base::offset_t index_to_offset(uint32_t index, uint32_t ndims)
{
    int8_t iz = index / (3*3);
    int8_t iy = (index - 3*3*iz)/3;
    int8_t ix = index % 3;

    return LightOctree_base::offset_t{ (int8_t)(ix-1), (int8_t)(iy-1), (int8_t)((ndims==2)?0:iz-1)};
}
} // namespace

inline void LightOctree_hashmap_precompute_init( 
    const LightOctree_hashmap& lmesh_hashmap, 
    const LightOctree_hashmap::Storage_t& storage,
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOcts_leaves,
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOcts_intermediates,
    uint32_t ndims )
{
    uint32_t nneighbors = (ndims==2) ? 3*3 : 3*3*3;
    uint32_t nbOcts_total = storage.getNumOctants()
                          + storage.getNumGhosts()
                          + storage.getNumIntermediates()
                          + storage.getNumIntermediateGhosts();
    neighbor_iOcts_leaves = Kokkos::View< uint32_t**, Kokkos::LayoutLeft >("LightOctree_hashmap_precompute::neighbor_iOcts", nbOcts_total, nneighbors);
    neighbor_iOcts_intermediates = Kokkos::View< uint32_t**, Kokkos::LayoutLeft >("LightOctree_hashmap_precompute::neighbor_iOcts", nbOcts_total, nneighbors);

    Kokkos::parallel_for( "precompute_neighbors", nbOcts_total*nneighbors,
        KOKKOS_LAMBDA(uint32_t index)
    {
        uint32_t iOct_local = index%nbOcts_total;
        uint32_t neighbor_id = index/nbOcts_total;

        LightOctree_base::OctantIndex iOct = storage.iOctLocal_to_OctantIndex(iOct_local);
        LightOctree_base::offset_t offset = index_to_offset(neighbor_id, ndims);

        if( !(iOct.isGhost || iOct.isIntermediate) )
        {
            LightOctree_base::OctantIndex iOct_neighbor_leaves = lmesh_hashmap.findNeighbor( iOct, offset );
            uint32_t iOct_local_neighbor_leaves = storage.OctantIndex_to_iOctLocal( iOct_neighbor_leaves );
            neighbor_iOcts_leaves(iOct_local, neighbor_id) = iOct_local_neighbor_leaves;
        }

        if( !iOct.isGhost )
        {
            LightOctree_base::OctantIndex iOct_neighbor_intermediates = lmesh_hashmap.findNeighbor_intermediate( iOct, offset );
            uint32_t iOct_local_neighbor_intermediates = storage.OctantIndex_to_iOctLocal( iOct_neighbor_intermediates );
            neighbor_iOcts_intermediates(iOct_local, neighbor_id) = iOct_local_neighbor_intermediates;
        }
    });
}


class LightOctree_hashmap_precompute : public LightOctree_hashmap{
public:
    LightOctree_hashmap_precompute() = default;
    LightOctree_hashmap_precompute(const LightOctree_hashmap_precompute& lmesh) = default;

    template < typename AMRmesh_t >
    LightOctree_hashmap_precompute( const AMRmesh_t* pmesh, uint8_t level_min, uint8_t level_max )
    : LightOctree_hashmap(pmesh, level_min, level_max)
    {
        LightOctree_hashmap_precompute_init(*this, this->storage, this->neighbor_iOct_leaves, this->neighbor_iOct_intermediates, getNdim());
    }

    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor_aux(const OctantIndex& iOct, const offset_t& offset, bool has_intermediates, const Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOcts) const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG( !this->isBoundary(iOct, offset), "findNeighbor not compatible with boundaries, please check isBoundary() before" );
        if( !has_intermediates )
        {
            DYABLO_ASSERT_KOKKOS_DEBUG( !iOct.isIntermediate, "findNeighbor can't get neighbor of intermediate octant, use findNeighbor_intermediate" );
        }

        //DYABLO_ASSERT_KOKKOS_DEBUG( !iOct.isGhost, "findNeighbor can't get neighbor of ghost octant" );
        if( iOct.isGhost )
            return LightOctree_hashmap::findNeighbor( iOct, offset ); //Ghosts are only on demand because we don't have all their neighbors.

        uint32_t iOct_local = storage.OctantIndex_to_iOctLocal( iOct );
        uint32_t neighbor_id = offset_to_index(offset, getNdim());
        uint32_t iOct_local_neighbor = neighbor_iOcts( iOct_local, neighbor_id );

        OctantIndex iOct_n = storage.iOctLocal_to_OctantIndex(iOct_local_neighbor);
        return iOct_n;
    }

    //! @copydoc LightOctree_base::findNeighbor()
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor(const OctantIndex& iOct, const offset_t& offset) const
    {
        return findNeighbor_aux( iOct, offset, false, neighbor_iOct_leaves );
    }

    //! @copydoc LightOctree_base::findNeighbor_intermediates()
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor_intermediate(const OctantIndex& iOct, const offset_t& offset) const
    {
        return findNeighbor_aux( iOct, offset, true, neighbor_iOct_intermediates );
    }


private:
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOct_leaves; 
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOct_intermediates; 
};

} //namespace dyablo
