#pragma once

#include "source_terms/SourceUpdate_base.h"

template<>
inline bool dyablo::SourceUpdateFactory::init()
{
  return true;
}

