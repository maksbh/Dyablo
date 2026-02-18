#pragma once

#include <memory>

#include "kokkos_shared.h"
#include "amr/LightOctree_forward.h"

namespace dyablo {

class Binned_iOcts_levels
{
public:
    Binned_iOcts_levels(const LightOctree& lmesh, int level_max);
    Binned_iOcts_levels( Binned_iOcts_levels&& );
    ~Binned_iOcts_levels();

    Kokkos::View<uint32_t*> get_iOcts_leaves(int level);
    Kokkos::View<uint32_t*> get_iOcts_ghost_leaves(int level);
    Kokkos::View<uint32_t*> get_iOcts_intermediates(int level);
    Kokkos::View<uint32_t*> get_iOcts_ghost_intermediates(int level);

//private:
public:
    struct Pdata;
    std::unique_ptr<Pdata> pdata;
};

} // namespace dyablo