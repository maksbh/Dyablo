#pragma once

#include "init/InitialConditions_base.h"
#include "plugins_lib.h"

template<>
bool dyablo::InitialConditionsFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}
