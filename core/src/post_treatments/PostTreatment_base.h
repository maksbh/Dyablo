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


class PostTreatment{
public:
  using OutputArray = ForeachCell::CellArray_global;
  // PostTreatment(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,  
  //     Timers& timers);
  virtual std::vector<std::string> get_fields_names() const = 0;
  virtual void compute_post_treatment( OutputArray &out, const UserData &U ) const = 0;
  virtual ~PostTreatment(){}
};

using PostTreatmentFactory = RegisteringFactory<PostTreatment, 
  ConfigMap&,
  ForeachCell&, 
  Timers&>;


} // namespace dyablo