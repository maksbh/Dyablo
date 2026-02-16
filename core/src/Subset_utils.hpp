#pragma once

#include <memory>
#include "kokkos_shared.h"
#include "amr/LightOctree_forward.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {

template<bool _locals, bool _ghosts, bool _intermediates>
class IterationSpace_level_impl
{
private:

ForeachCell::IterationSpace_fullArray_impl<_locals, _ghosts, _intermediates> full_array;

Kokkos::View<uint32_t*> iOcts_locals;
Kokkos::View<uint32_t*> iOcts_ghosts;
Kokkos::View<uint32_t*> iOcts_intermediates;
Kokkos::View<uint32_t*> iOcts_intermediate_ghosts;

public:
IterationSpace_level_impl(  const ForeachCell::CellArray_shape& iter_space_level,
                            Kokkos::View<uint32_t*>& iOcts_locals,
                            Kokkos::View<uint32_t*>& iOcts_ghosts,
                            Kokkos::View<uint32_t*>& iOcts_intermediates,
                            Kokkos::View<uint32_t*>& iOcts_intermediate_ghosts)
: full_array( iter_space_level ),
  iOcts_locals(iOcts_locals),
  iOcts_ghosts(iOcts_ghosts),
  iOcts_intermediates(iOcts_intermediates),
  iOcts_intermediate_ghosts(iOcts_intermediate_ghosts)
{}

KOKKOS_INLINE_FUNCTION
uint32_t bx() const         { return full_array.bx(); }
KOKKOS_INLINE_FUNCTION
uint32_t by() const         { return full_array.by(); }
KOKKOS_INLINE_FUNCTION
uint32_t bz() const         { return full_array.bz(); }

KOKKOS_INLINE_FUNCTION
uint32_t iOct_count() const { return full_array.iOct_count(); }

KOKKOS_INLINE_FUNCTION
ForeachCell::CellIndex getCellIndex(uint32_t iOct_in, uint32_t i, uint32_t j, uint32_t k) const
{
    ForeachCell::CellIndex iCell = full_array.getCellIndex(iOct_in, i, j, k);
    LightOctree::OctantIndex iOct_raw = iCell.iOct;

    uint32_t iOct_filtered = 0;
    if(      !iOct_raw.isGhost && !iOct_raw.isIntermediate )
        iOct_filtered = iOcts_locals(iOct_raw.iOct);
    else if(  iOct_raw.isGhost && !iOct_raw.isIntermediate )
        iOct_filtered = iOcts_ghosts(iOct_raw.iOct);
    else if( !iOct_raw.isGhost &&  iOct_raw.isIntermediate )
        iOct_filtered = iOcts_intermediates(iOct_raw.iOct);
    else//if( iOct_raw.isGhost &&  iOct_raw.isIntermediate )
        iOct_filtered = iOcts_intermediate_ghosts(iOct_raw.iOct);
    iCell.iOct.iOct = iOct_filtered;

    return iCell;
}
};


class Subset_levels
{
public:
    Subset_levels(const LightOctree& lmesh, int level_max);
    ~Subset_levels();

    template<typename GhostComm_t>
    typename GhostComm_t::OctSubset getGhostCommunicatorSubset_level( int level, const GhostComm_t& comm )
    {
        return typename GhostComm_t::OctSubset( comm, get_iOcts_ghost_leaves(level) );
    }

    template<bool locals, bool ghosts, bool intermediates>
    IterationSpace_level_impl<locals, ghosts, intermediates> getIterationSpace(int level, const ForeachCell::CellArray_shape& shape )
    {
        uint32_t bx = shape.bx;
        uint32_t by = shape.by;
        uint32_t bz = shape.bz;
        uint32_t nbFields = shape.nbFields;

        Kokkos::View<uint32_t*> iOcts_locals, iOcts_ghosts, iOcts_intermediates, iOcts_intermediate_ghosts;
        if( locals ) iOcts_locals = this->get_iOcts_leaves(level);
        if( ghosts ) iOcts_ghosts = this->get_iOcts_ghost_leaves(level);
        if( intermediates ) iOcts_intermediates = this->get_iOcts_intermediates(level);
        if( intermediates && ghosts ) iOcts_intermediate_ghosts = this->get_iOcts_ghost_intermediates(level);
        
        ForeachCell::CellArray_shape iter_space{
            .bx=bx, .by=by, .bz=bz, 
            .nbFields = nbFields,
            .nbOcts = (uint32_t)iOcts_locals.size(),
            .nbGhosts = (uint32_t)iOcts_ghosts.size(),
            .nbIntermediates = (uint32_t)iOcts_intermediates.size(),
        };

        return IterationSpace_level_impl<locals, ghosts, intermediates>( iter_space, iOcts_locals, iOcts_ghosts, iOcts_intermediates, iOcts_intermediate_ghosts );
    }
protected:
    Kokkos::View<uint32_t*> get_iOcts_leaves(int level);
    Kokkos::View<uint32_t*> get_iOcts_ghost_leaves(int level);
    Kokkos::View<uint32_t*> get_iOcts_intermediates(int level);
    Kokkos::View<uint32_t*> get_iOcts_ghost_intermediates(int level);

//private:
    struct Pdata;
    std::unique_ptr<Pdata> pdata;
};

} // namespace dyablo