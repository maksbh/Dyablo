#pragma once

#include <memory>
#include "mpi/GhostCommunicator.h"

namespace dyablo{


class Subset_levels
{
public:
    Subset_levels(const LightOctree& lmesh, int level_max);
    ~Subset_levels();

    GhostCommunicator_full_blocks::OctSubset getGhostCommunicatorSubset_level( int level, const GhostCommunicator_full_blocks& comm );
    GhostCommunicator_partial_blocks::OctSubset getGhostCommunicatorSubset_level( int level, const GhostCommunicator_partial_blocks& comm );

public:
    struct Pdata;
    std::unique_ptr<Pdata> pdata;
};

} // namespace dyablo