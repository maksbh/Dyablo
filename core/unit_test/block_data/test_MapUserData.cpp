/**
 * \file test_MapUserData.cpp
 * \author A. Durocher
 * Test user data remapping after amr cycle
 * 
 * Create an AMR mesh and fill user data with cell positions
 * Refine/coarsen mesh and then test if new user data contains the right positions
 * 
 * This also works with multiple MPI process
 */


#include "gtest/gtest.h"

#include "amr/MapUserData.h"

#include "amr/AMRmesh.h"
#include "amr/LightOctree.h"
#include "io/IOManager.h"

namespace dyablo
{

namespace {
struct MapDataTestParams {
  std::shared_ptr<AMRmesh> amr_mesh;
  ForeachCell &foreach_cell;
  UserData &U;
  std::string mapUserData_id;
  uint32_t bx, by, bz;

  uint nbfields;
  int ndim;
};

enum VarIndex {Px, Py, Pz};

const real_t gradX = 1.0;
const real_t gradY = 3.0;
const real_t gradZ = 7.0;

void init_position(MapDataTestParams &test_params) {
  auto &foreach_cell = test_params.foreach_cell;
  auto &U = test_params.U;

  U.new_fields({"px", "py", "pz"});
  UserData::FieldAccessor Uin = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}} );
  const ForeachCell::CellMetaData& cells = test_params.foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell( "Init_U", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
  {
    auto c = cells.getCellCenter( iCell );
    Uin.at(iCell, Px) = c[IX];
    Uin.at(iCell, Py) = c[IY];
    Uin.at(iCell, Pz) = c[IZ];
  });
  //U.exchange_ghosts( ViewCommunicator( amr_mesh ) );
}
}

void init_linear(MapDataTestParams &test_params) {
  auto &foreach_cell = test_params.foreach_cell;
  auto &U = test_params.U;

  U.new_fields({"px", "py", "pz"});
  UserData::FieldAccessor Uin = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}} );
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell( "Init_U", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
  {
    auto c = cells.getCellCenter( iCell );
    Uin.at(iCell, Px) = c[IX] * gradX;
    Uin.at(iCell, Py) = c[IY] * gradY;
    Uin.at(iCell, Pz) = c[IZ] * gradZ;
  });
}


void init_test(MapDataTestParams &test_params) {
  if (test_params.mapUserData_id == "MapUserData_linear") 
    init_linear(test_params);
  else
    init_position(test_params);

  test_params.nbfields = test_params.U.nbFields();
}



void check_position(MapDataTestParams &test_params) {
  auto &amr_mesh = test_params.amr_mesh;
  auto &U = test_params.U;
  auto ndim = test_params.ndim;
  auto bx = test_params.bx;
  auto by = test_params.by;
  auto bz = test_params.bz;

  uint32_t nbOcts = amr_mesh->getNumOctants();

  std::cout << "Check remapped U ( nbOcts=" << nbOcts << ")" << std::endl;

  EXPECT_EQ( U.nbFields(), 3 );

  EXPECT_EQ( U.getField("px").nbOcts, nbOcts );
  EXPECT_EQ( U.getField("py").nbOcts, nbOcts );
  EXPECT_EQ( U.getField("pz").nbOcts, nbOcts );

  uint32_t nbCellsPerOct = bx*by*bz;
  uint32_t expected_size = nbOcts*nbCellsPerOct;
  EXPECT_EQ( U.getField("px").U.size(), expected_size );
  EXPECT_EQ( U.getField("py").U.size(), expected_size );
  EXPECT_EQ( U.getField("pz").U.size(), expected_size );

  auto Uhost_px = Kokkos::create_mirror_view(U.getField("px").U);
  auto Uhost_py = Kokkos::create_mirror_view(U.getField("py").U);
  auto Uhost_pz = Kokkos::create_mirror_view(U.getField("pz").U);
  Kokkos::deep_copy(Uhost_px, U.getField("px").U);
  Kokkos::deep_copy(Uhost_py, U.getField("py").U);
  Kokkos::deep_copy(Uhost_pz, U.getField("pz").U);

  real_t oct_size_initial = 1./8; 
  for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
  {
    auto oct_pos = amr_mesh->getCoordinates(iOct);
    real_t oct_size = amr_mesh->getSize(iOct)[0];
    
    for( uint32_t c=0; c<nbCellsPerOct; c++ )
    {
      uint32_t cz = c/(bx*by);
      uint32_t cy = (c - cz*bx*by)/bx;
      uint32_t cx = c - cz*bx*by - cy*bx;

      real_t expected_x = oct_pos[IX] + (cx+0.5)*oct_size/bx;
      real_t expected_y = oct_pos[IY] + (cy+0.5)*oct_size/by;
      real_t expected_z = oct_pos[IZ] + (cz+0.5)*oct_size/bz; 

      if(oct_size < oct_size_initial)
      { // When cell is newly refined there is no interpolation
        expected_x += ((cx%2)?-1:1) * oct_size/(2*bx);
        expected_y += ((cy%2)?-1:1) * oct_size/(2*by);
        expected_z += ((cz%2)?-1:1) * oct_size/(2*bz);
      }

      if(ndim==2)
        expected_z = 0;

      EXPECT_NEAR( Uhost_px(c, 0, iOct), expected_x , 0.0001);
      EXPECT_NEAR( Uhost_py(c, 0, iOct), expected_y , 0.0001);
      EXPECT_NEAR( Uhost_pz(c, 0, iOct), expected_z , 0.0001);
    }
  }
}

void check_linear(MapDataTestParams &test_params) {
  auto &amr_mesh = test_params.amr_mesh;
  auto &U = test_params.U;
  auto ndim = test_params.ndim;
  auto bx = test_params.bx;
  auto by = test_params.by;
  auto bz = test_params.bz;

  uint32_t nbOcts = amr_mesh->getNumOctants();

  std::cout << "Check remapped U ( nbOcts=" << nbOcts << ")" << std::endl;

  EXPECT_EQ( U.nbFields(), 3 );

  EXPECT_EQ( U.getField("px").nbOcts, nbOcts );
  EXPECT_EQ( U.getField("py").nbOcts, nbOcts );
  EXPECT_EQ( U.getField("pz").nbOcts, nbOcts );

  uint32_t nbCellsPerOct = bx*by*bz;
  uint32_t expected_size = nbOcts*nbCellsPerOct;
  EXPECT_EQ( U.getField("px").U.size(), expected_size );
  EXPECT_EQ( U.getField("py").U.size(), expected_size );
  EXPECT_EQ( U.getField("pz").U.size(), expected_size );

  auto Uhost_px = Kokkos::create_mirror_view(U.getField("px").U);
  auto Uhost_py = Kokkos::create_mirror_view(U.getField("py").U);
  auto Uhost_pz = Kokkos::create_mirror_view(U.getField("pz").U);
  Kokkos::deep_copy(Uhost_px, U.getField("px").U);
  Kokkos::deep_copy(Uhost_py, U.getField("py").U);
  Kokkos::deep_copy(Uhost_pz, U.getField("pz").U);

  real_t oct_size_initial = 1./8; 
  for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
  {
    auto oct_pos = amr_mesh->getCoordinates(iOct);
    real_t oct_size = amr_mesh->getSize(iOct)[0];
    
    for( uint32_t c=0; c<nbCellsPerOct; c++ )
    {
      uint32_t cz = c/(bx*by);
      uint32_t cy = (c - cz*bx*by)/bx;
      uint32_t cx = c - cz*bx*by - cy*bx;

      real_t expected_x = oct_pos[IX] + (cx+0.5)*oct_size/bx;
      real_t expected_y = oct_pos[IY] + (cy+0.5)*oct_size/by;
      real_t expected_z = oct_pos[IZ] + (cz+0.5)*oct_size/bz; 

      expected_x *= gradX;
      expected_y *= gradY;
      expected_z *= gradZ;

      if(ndim==2)
        expected_z = 0;

      EXPECT_NEAR( Uhost_px(c, 0, iOct), expected_x, 0.0001);
      EXPECT_NEAR( Uhost_py(c, 0, iOct), expected_y, 0.0001);
      EXPECT_NEAR( Uhost_pz(c, 0, iOct), expected_z, 0.0001);
    }
  }
}

void check_test(MapDataTestParams &test_params) {
  // Checking results
  if (test_params.mapUserData_id == "MapUserData_linear")
    check_linear(test_params);
  else
    check_position(test_params);
}

// =======================================================================
// =======================================================================
void run_test(int ndim, std::string mapUserData_id)
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing MapUserData with linear mapping ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = (ndim==3)?8:1;

  int level_min = 2;
  int level_max = 8;
  std::shared_ptr<AMRmesh> amr_mesh;
  //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, level_min, level_max );
    amr_mesh->adaptGlobalRefine();
    // amr_mesh->setBalanceCodimension(ndim);
    // uint32_t idx = 0;
    // amr_mesh->setBalance(idx,true);
    // amr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    // amr_mesh->setPeriodic(4);
    // amr_mesh->setPeriodic(5);
  }

  Timers timers;

  std::string config_str = 
    "[output]\n"
    "hdf5_enabled=true\n"
    "write_mesh_info=true\n"
    "write_variables=px,py,pz\n"
    "write_iOct=false\n"
    "outputPrefix=output\n"
    "outputDir=./\n"
    "[amr]\n"
    "use_block_data=true\n"
    "bx=8\n"
    "by=8\n";
  ConfigMap configMap(config_str); //Use default values

  configMap.getValue<int>("mesh", "ndim", ndim);
  configMap.getValue<int>("amr", "bz", ndim==2?1:8);
  
  ForeachCell foreach_cell( *amr_mesh, configMap );

  std::unique_ptr<MapUserData> mapUserData = MapUserDataFactory::make_instance( mapUserData_id,
    configMap,
    foreach_cell,
    timers
  );

  uint32_t nbOcts = amr_mesh->getNumOctants();

  UserData U( configMap, foreach_cell ); 

  MapDataTestParams test_params{amr_mesh, foreach_cell, U, mapUserData_id, bx, by, bz, 0, ndim};
  
  std::cout << "Initialize User Data..." << std::endl;
  init_test(test_params);

  ScalarSimulationData scalar_data;
  scalar_data.set<int>("iter", 0);
  scalar_data.set<real_t>("time",0.0);
  std::string iomanager_id = "IOManager_hdf5";
  std::unique_ptr<IOManager> io_manager = IOManagerFactory::make_instance( iomanager_id,
    configMap,
    foreach_cell,
    timers
  );
  io_manager->save_snapshot(U, scalar_data);

  mapUserData->save_old_mesh();
  {
    std::cout << "Coarsening octants" << std::endl;

     for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
     {
       amr_mesh->setMarker(iOct , -1);
     }

    // // Refine 0 because it is at an MPI boundary 
    // // 2:1 balance should create cells that are neither coarsened nor refined
    amr_mesh->setMarker((uint32_t)0 , 1);
    // // Refine a random octant in the middle
    amr_mesh->setMarker(nbOcts/2 , 1);

    amr_mesh->adapt(true);
  }
  
  std::cout << "Remapping user data..." << std::endl;

  mapUserData->remap(U);

  std::cout << "Checking data" << std::endl;

  check_test(test_params);

  scalar_data.set<int>("iter", 1);
  scalar_data.set<real_t>("time", 1.0);
  io_manager->save_snapshot(U, scalar_data);

} // run_test

} // namespace dyablo

class Test_MapUserData
  : public testing::TestWithParam<std::tuple<int, std::string>> 
{};

TEST_P(Test_MapUserData, position_field_conserved)
{
  int ndim = std::get<0>(GetParam());
  std::string id = std::get<1>(GetParam());
  dyablo::run_test(ndim, id );
}

INSTANTIATE_TEST_SUITE_P(
    Test_MapUserData, Test_MapUserData,
    testing::Combine(
        testing::Values(2,3),
        testing::ValuesIn( dyablo::MapUserDataFactory::get_available_ids() )
    ),
    [](const testing::TestParamInfo<Test_MapUserData::ParamType>& info) {
      std::string name = 
          (std::get<0>(info.param) == 2 ? std::string("2D") : std::string("3D"))
          + "_" + std::get<1>(info.param);
      return name;
    }
);
