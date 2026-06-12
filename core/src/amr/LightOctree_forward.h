#pragma once

namespace dyablo { 

// class LightOctree_hashmap;

// #ifdef KOKKOS_ENABLE_CUDA
// using LightOctree = LightOctree_hashmap;
// #endif

class LightOctree_hashmap_precompute;
using LightOctree = LightOctree_hashmap_precompute;

} //namespace dyablo
