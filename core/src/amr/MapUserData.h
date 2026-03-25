#pragma once

#include "MapUserData_base.h"

namespace dyablo {

class MapUserData_mean;
class MapUserData_linear;

} //namespace dyablo 


template<>
inline bool dyablo::MapUserDataFactory::init()
{
  //  DECLARE_REGISTERED(dyablo::MapUserData_mean);
  //  DECLARE_REGISTERED(dyablo::MapUserData_linear);

  return true;
}