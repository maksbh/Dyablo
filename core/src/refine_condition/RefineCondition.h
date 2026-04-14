#pragma once

#include "refine_condition/RefineCondition_base.h"

template<>
inline bool dyablo::RefineConditionFactory::init()
{
  return true;
}