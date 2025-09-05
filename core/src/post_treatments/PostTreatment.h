#pragma once

#include "post_treatments/PostTreatment_base.h"

namespace dyablo{

class PostTreatment_divB;

} // namespace dyablo



template<>
bool dyablo::PostTreatmentFactory::init()
{
  DECLARE_REGISTERED( dyablo::PostTreatment_divB);

  return true;
}

