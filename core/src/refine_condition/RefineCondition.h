#pragma once

#include "refine_condition/RefineCondition_base.h"
#include "plugins_lib.h"

template<>
inline bool dyablo::RefineConditionFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}