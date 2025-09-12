#pragma once

#include "derived_fields/DerivedFields_base.h"

namespace dyablo{

class DerivedFields_divB;

} // namespace dyablo



template<>
bool dyablo::DerivedFieldsFactory::init()
{
  DECLARE_REGISTERED( dyablo::DerivedFields_divB);

  return true;
}

