#pragma once

#include <memory>
#include "kokkos_shared.h"
#include "amr/LightOctree_forward.h"

namespace dyablo {


class Subset_levels
{
public:
    Subset_levels(const LightOctree& lmesh, int level_max);
    ~Subset_levels();

    template<typename GhostComm_t>
    typename GhostComm_t::OctSubset getGhostCommunicatorSubset_level( int level, const GhostComm_t& comm )
    {
        return typename GhostComm_t::OctSubset( comm, get_ghost_leaves_iOct_list(level) );
    }
protected:
    Kokkos::View<uint32_t*> get_ghost_leaves_iOct_list(int level);

//private:
    struct Pdata;
    std::unique_ptr<Pdata> pdata;
};

} // namespace dyablo