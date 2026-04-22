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
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOct_leaves,
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOct_intermediates,
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOct_parents,
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOct_children,
    uint32_t ndims )
{
    using OctantIndex = LightOctree_base::OctantIndex;
    using offset_t = LightOctree_base::offset_t;

    uint32_t nneighbors = (ndims==2) ? 3*3 : 3*3*3;
    
    auto precompute = [&](std::string name, uint32_t nbOcts_total, uint32_t nneighbors,
                          const auto& generate_iOct,
                          const auto& findNeighbor)
    {
        Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOcts(name, nbOcts_total, nneighbors );
        Kokkos::parallel_for( name+"::compute", nbOcts_total*nneighbors,
            KOKKOS_LAMBDA(uint32_t index)
        {
            uint32_t iOct_local = index%nbOcts_total;
            uint32_t neighbor_id = index/nbOcts_total;

            LightOctree_base::OctantIndex iOct = generate_iOct(iOct_local);
            LightOctree_base::offset_t offset = index_to_offset(neighbor_id, ndims);

            OctantIndex iOct_neighbor = findNeighbor( iOct, offset );
            uint32_t iOct_local_neighbor = storage.OctantIndex_to_iOctLocal( iOct_neighbor );
            neighbor_iOcts(iOct_local, neighbor_id) = iOct_local_neighbor;
        });
        return neighbor_iOcts;
    };

    uint32_t nbOcts = storage.getNumOctants();
    uint32_t nbIntermediates = storage.getNumIntermediates();
    neighbor_iOct_leaves = precompute( "LightOctree_hashmap_precompute::neighbor_iOct_leaves", nbOcts, nneighbors, 
        []( uint32_t i ){return OctantIndex{i};},
        [&]( const OctantIndex& iOct, const offset_t& offset )
        {
            if(lmesh_hashmap.isBoundary( iOct, offset ))
                return OctantIndex{};
            else 
                return lmesh_hashmap.findNeighbor(iOct, offset); 
        } 
    );

    neighbor_iOct_intermediates = precompute( "LightOctree_hashmap_precompute::neighbor_iOct_intermediates", nbOcts+nbIntermediates, nneighbors, 
        [&]( uint32_t i )
        {
            if( i<nbOcts )
                return OctantIndex{i};
            else
                return OctantIndex{i-nbOcts, false, true};
        },
        [&]( const OctantIndex& iOct, const offset_t& offset )
        {
            if(lmesh_hashmap.isBoundary( iOct, offset ))
                return OctantIndex{};
            else 
                return lmesh_hashmap.findNeighbor_intermediate(iOct, offset); 
        } 
    );

    int level_min = storage.level_min;
    neighbor_iOct_parents = precompute( "LightOctree_hashmap_precompute::neighbor_iOct_parents", nbOcts+nbIntermediates, 1, 
        [&]( uint32_t i )
        {
            if( i<nbOcts )
                return OctantIndex{i};
            else
                return OctantIndex{i-nbOcts, false, true};
        },
        [&]( const OctantIndex& iOct, const offset_t& offset )
        {
            if(lmesh_hashmap.getLevel( iOct ) == level_min)
                return OctantIndex{};
            else 
                return lmesh_hashmap.findParent(iOct); 
        } 
    );

    neighbor_iOct_children = precompute( "LightOctree_hashmap_precompute::neighbor_iOct_children", nbIntermediates, 1, 
        []( uint32_t i ){ return OctantIndex{i, false, true}; },
        [&]( const OctantIndex& iOct, const offset_t& offset )
        { return lmesh_hashmap.findChild(iOct, {0,0,0}); } 
    );
}


class LightOctree_hashmap_precompute : public LightOctree_hashmap{
public:
    LightOctree_hashmap_precompute() = default;
    LightOctree_hashmap_precompute(const LightOctree_hashmap_precompute& lmesh) = default;

    template < typename AMRmesh_t >
    LightOctree_hashmap_precompute( const AMRmesh_t* pmesh, uint8_t level_min, uint8_t level_max )
    : LightOctree_hashmap(pmesh, level_min, level_max)
    {
        LightOctree_hashmap_precompute_init(
            *this, this->storage, 
            this->neighbor_iOct_leaves, 
            this->neighbor_iOct_intermediates, 
            this->neighbor_iOct_parents, 
            this->neighbor_iOct_children, 
            getNdim());
    }

    template< bool has_intermediates >
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor_aux(const OctantIndex& iOct, const offset_t& offset, const Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOcts) const
    {
        if( offset[IX] == 0 && offset[IY] == 0 && offset[IZ] == 0 )
            return iOct;

        DYABLO_ASSERT_KOKKOS_DEBUG( !this->isBoundary(iOct, offset), "findNeighbor not compatible with boundaries, please check isBoundary() before" );
        if( !has_intermediates )
        {
            DYABLO_ASSERT_KOKKOS_DEBUG( !iOct.isIntermediate, "findNeighbor can't get neighbor of intermediate octant, use findNeighbor_intermediate" );
        }

        //DYABLO_ASSERT_KOKKOS_DEBUG( !iOct.isGhost, "findNeighbor can't get neighbor of ghost octant" );
        if( iOct.isGhost )
        {   
            //Ghosts are only on demand because we don't have all their neighbors.
            return LightOctree_hashmap::findNeighbor_aux<has_intermediates>( iOct, offset );
        }

        uint32_t nbOcts = storage.getNumOctants();

        uint32_t iOct_local = iOct.iOct;
        if constexpr( has_intermediates ) 
            iOct_local += iOct.isIntermediate * nbOcts;
        uint32_t neighbor_id = offset_to_index(offset, getNdim());
        uint32_t iOct_local_neighbor = neighbor_iOcts( iOct_local, neighbor_id );

        OctantIndex iOct_n = storage.iOctLocal_to_OctantIndex(iOct_local_neighbor);

        DYABLO_ASSERT_KOKKOS_DEBUG( [&]()
            {
                OctantIndex iOct_expected = LightOctree_hashmap::findNeighbor_aux<has_intermediates>( iOct, offset );
                return iOct_expected.iOct == iOct_n.iOct && iOct_expected.isGhost == iOct_n.isGhost && iOct_expected.isIntermediate == iOct_n.isIntermediate;
            }() 
            , "LightOctree_hashmap_precompute::findNeighbor does not match LightOctree_hashmap" );

        return iOct_n;
    }

    //! @copydoc LightOctree_base::findNeighbor()
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor(const OctantIndex& iOct, const offset_t& offset) const
    {
        return findNeighbor_aux<false>( iOct, offset, neighbor_iOct_leaves );
    }

    //! @copydoc LightOctree_base::findNeighbor_intermediates()
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor_intermediate(const OctantIndex& iOct, const offset_t& offset) const
    {
        return findNeighbor_aux<true>( iOct, offset, neighbor_iOct_intermediates );
    }

    KOKKOS_INLINE_FUNCTION
    OctantIndex findParent( const OctantIndex& iOct )  const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG( this->getLevel( iOct ) > this->min_level, "Can't get parent at coarse level" );
        
        if( iOct.isGhost )
            return LightOctree_hashmap::findParent( iOct );
        
        uint32_t iOct_local = iOct.iOct + iOct.isIntermediate * getNumOctants();
        uint32_t iOct_local_p = neighbor_iOct_parents(iOct_local,0);

        OctantIndex iOct_p = storage.iOctLocal_to_OctantIndex( iOct_local_p );

        DYABLO_ASSERT_KOKKOS_DEBUG( [&]()
            {
                OctantIndex iOct_expected = LightOctree_hashmap::findParent( iOct );
                return iOct_expected.iOct == iOct_p.iOct && iOct_expected.isGhost == iOct_p.isGhost && iOct_expected.isIntermediate == iOct_p.isIntermediate;
            }() 
            , "LightOctree_hashmap_precompute::findParent does not match LightOctree_hashmap" );

        return iOct_p;
    }

    KOKKOS_INLINE_FUNCTION
    OctantIndex findChild( const OctantIndex& iOct, const offset_t& offset )  const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG( iOct.isIntermediate, "Leaves have no children" );
    
        if( iOct.isGhost )
            return LightOctree_hashmap::findChild( iOct, offset );
        
        uint32_t iOct_local = iOct.iOct;
        uint32_t iOct_local_c0 = neighbor_iOct_children(iOct_local,0);
        OctantIndex iOct_c0 = storage.iOctLocal_to_OctantIndex( iOct_local_c0 );

        OctantIndex iOct_c = findNeighbor_intermediate(iOct_c0, offset);
        DYABLO_ASSERT_KOKKOS_DEBUG( [&]()
            {
                OctantIndex iOct_expected = LightOctree_hashmap::findChild( iOct, offset );
                return iOct_expected.iOct == iOct_c.iOct && iOct_expected.isGhost == iOct_c.isGhost && iOct_expected.isIntermediate == iOct_c.isIntermediate;
            }() 
            , "LightOctree_hashmap_precompute::findChild does not match LightOctree_hashmap" );
        return iOct_c;
    }


private:
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOct_leaves; 
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOct_intermediates; 
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOct_parents; 
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOct_children; 
};

} //namespace dyablo
