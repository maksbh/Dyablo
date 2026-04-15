#pragma once

#include "MapUserData_base.h"
#include "plugins_lib.h"

template<>
inline bool dyablo::MapUserDataFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}