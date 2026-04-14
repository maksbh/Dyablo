#pragma once

#include "IOManager_base.h"

template<>
inline bool dyablo::IOManagerFactory::init()
{
  return true;
}