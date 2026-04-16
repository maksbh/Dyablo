#pragma once

#include "gravity/GravitySolver_base.h"
#include "plugins_lib.h"

template<>
inline bool dyablo::GravitySolverFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}

