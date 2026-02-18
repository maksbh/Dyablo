#pragma once

#include <memory>

#include "kokkos_shared.h"
#include "amr/LightOctree_forward.h"

namespace dyablo {

/**
 * Class to generate per-level octant lists
 * 
 * This precomputes octant lists for every level to quickly get 
 * per-level octant lists for multiple levels.
 **/
class Binned_iOcts_levels
{
public:
    Binned_iOcts_levels(const LightOctree& lmesh, int level_max);
    Binned_iOcts_levels( Binned_iOcts_levels&& );
    ~Binned_iOcts_levels();

    /// get list of non-ghost leave iOcts
    Kokkos::View<uint32_t*> get_iOcts_leaves(int level);
    /// get list of ghost leave iOcts
    Kokkos::View<uint32_t*> get_iOcts_ghost_leaves(int level);
    /// get list of non-ghost intermediate iOcts
    Kokkos::View<uint32_t*> get_iOcts_intermediates(int level);
    /// get list of ghost intermediate iOcts
    Kokkos::View<uint32_t*> get_iOcts_ghost_intermediates(int level);

//private:
public:
    struct Pdata;
    std::unique_ptr<Pdata> pdata;
};

} // namespace dyablo