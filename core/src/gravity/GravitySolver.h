#pragma once

#include "gravity/GravitySolver_base.h"

template<>
inline bool dyablo::GravitySolverFactory::init()
{
  return true;
}

