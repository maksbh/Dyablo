#pragma once

#include "init/InitialConditions_base.h"

template<>
bool dyablo::InitialConditionsFactory::init()
{
  return true;
}
