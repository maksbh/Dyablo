#pragma once

#include "scheme/HyperbolicUpdate_base.h"
#include "plugins_lib.h"

template<>
inline bool dyablo::HyperbolicUpdateFactory::init()
{ 
  dyablo::load_dyablo_plugins_lib();
  return true;
}
