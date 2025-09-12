#pragma once 

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "UserData.h"
#include "utils/io/HDF5ViewWriter.h"

namespace dyablo{


class DerivedFields{
public:
  using OutputArray = ForeachCell::CellArray_global;
  // DerivedFields(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,  
  //     Timers& timers);
  virtual std::vector<std::string> get_fields_names() const = 0;
  virtual void compute_derived_fields( OutputArray &out, const UserData &U ) const = 0;
  virtual ~DerivedFields(){}
};

using DerivedFieldsFactory = RegisteringFactory<DerivedFields, 
  ConfigMap&,
  ForeachCell&, 
  Timers&>;


} // namespace dyablo