#include "gtest/gtest.h"

#include "amr/AMRmesh.h"
#include "utils/mpi/GlobalMpiSession.h"
#include "utils/monitoring/Timers.h"
#include "io/IOManager.h"

#include "foreach_cell/ForeachCell.h"

namespace dyablo {

enum VarIndex_test{
  ID,IE,IP,IU,IV,IW,IGX,IGY,IGZ
};

struct Spot{
  real_t x, y, z; // At position {x,y,z}
  real_t r0;      // Refine at distance ( r0*cell_size ) for each level
  int level; // until this level
};

struct Test_data{
  int level_min;
  int level_max;
  real_t width;
  int ndim = 3;
  // WARNING : volume check need to be modified if testing something else tha powers of 2
  int coarse_size_x = -1;
  int coarse_size_y = -1;
  int coarse_size_z = -1;
};

template< typename AMRmesh_t >
void run_test(const Test_data& test_data)
{
  int dim = test_data.ndim;
  bool perodic_x = true;
  bool perodic_y = true;
  bool perodic_z = true;
  int level_min = test_data.level_min;
  int level_max = test_data.level_max;
  real_t width = test_data.width;

  Timers timers;

  std::vector<Spot> spots = {
    {0.5,0.5,(dim==3)?0.5:0,2,10},
    {0.25,0.25,(dim==3)?0.25:0,width,level_max}
  };

  uint32_t max_coarse_size = (1U << level_min);

  std::array<uint32_t,3> coarse_grid_size = {
    (test_data.coarse_size_x == -1) ? (max_coarse_size) : test_data.coarse_size_x,
    (test_data.coarse_size_y == -1) ? (max_coarse_size) : test_data.coarse_size_y,
    (test_data.coarse_size_z == -1) ? ((dim==3)?max_coarse_size:1) : test_data.coarse_size_z
  };

  AMRmesh_t amr_mesh(dim, {perodic_x,perodic_y,perodic_z}, level_min, level_max, coarse_grid_size);
  if( test_data.level_max > amr_mesh.get_max_supported_level() )
  {
    std::cerr << "Test skipped : h=" << test_data.level_max << " unsupported by this AMRmesh implementation" << std::endl;
    GTEST_SKIP();
  }



  {
    uint32_t nbOcts = amr_mesh.getNumOctants();
    const auto lmesh_host = amr_mesh.getStorage();
    for( int level=level_min; level<level_max; level++ )
    {
      std::cout << "Refine level " << level << std::endl;  
      
      timers.get("MarkCells").start();
      int refine_count = 0;     
      for(uint32_t iOct=0; iOct<nbOcts; iOct++)
      {
        auto c = lmesh_host.getCenter({iOct, false});
        real_t s = lmesh_host.getSize({iOct, false})[0];
        bool refine = false;
        for( size_t i=0; i<spots.size(); i++ )
        {
          if( spots[i].level > level )
          {
            real_t dist_x = spots[i].x - c[0]; 
            real_t dist_y = spots[i].y - c[1]; 
            real_t dist_z = spots[i].z - c[2]; 

            real_t dist2 = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;
            real_t r2 = spots[i].r0*spots[i].r0*s*s;

            if( dist2 < r2  )
              refine = true;
          }
        }
        if(refine)
          refine_count++;
        amr_mesh.setMarker(iOct, refine?1:0);
      }
      timers.get("MarkCells").stop();

      std::cout << "Refine count " << refine_count << std::endl;
      
      timers.get("Adapt").start();
      amr_mesh.adapt();
      timers.get("Adapt").stop();
      timers.get("loadBalance").start();
      amr_mesh.loadBalance();
      timers.get("loadBalance").stop(); 
    }      
  }

  uint32_t nbOcts = amr_mesh.getNumOctants();

  // Check total volume is 1
  {
    LightOctree lmesh = amr_mesh.getLightOctree();

    real_t V=0;
    for(int i=level_max; i>=level_min; i--)
    { // Iterate over levels to avoid rounding errors
      uint64_t count_level = 0;
      real_t V_level = 0;
      Kokkos::parallel_reduce( "Check_volume", nbOcts,
        KOKKOS_LAMBDA( uint32_t iOct, uint64_t& count_level, real_t& V )
      {
        auto s = lmesh.getSize( {iOct, false} );
        int level = lmesh.getLevel( {iOct, false} );
        if( level==i )
        {
          V += s[IX]*s[IY]*( (dim==3)?s[IZ]:1 );
          count_level ++;
        }
      }, count_level, V_level);
      V+=V_level;
      std::cout << "level " << i << " : " << count_level << " octants, Volume = " << V_level << std::endl;
    }
    real_t Vtot;
    GlobalMpiSession::get_comm_world().MPI_Allreduce(&V, &Vtot, 1, MpiComm::MPI_Op_t::SUM );
    
    EXPECT_DOUBLE_EQ( 1.0, Vtot );
  }

  if constexpr( std::is_same<AMRmesh_t, AMRmesh>::value )
  {
    // Output generated mesh
    std::string configmap_str = 
      "[output]\n"
      "hdf5_enabled=true\n"
      "write_mesh_info=true\n"
      "write_variables=\n"
      "write_iOct=false\n"
      "outputPrefix=output\n"
      "outputDir=./\n"
      "[amr]\n"
      "use_block_data=true\n"
      "bx=1\n"
      "by=1\n";
    ConfigMap configMap(configmap_str);
    ForeachCell foreach_cell( amr_mesh, configMap );
    UserData U(configMap, foreach_cell);
    ScalarSimulationData scalar_data;
    scalar_data.set<int>("iter", 0);
    scalar_data.set<real_t>("time",1.0);
    IOManagerFactory::make_instance("IOManager_hdf5",configMap,foreach_cell,timers)->save_snapshot( U, scalar_data);
  }

  // Verify 2:1 balance
  {
    dyablo::LightOctree lmesh( &amr_mesh, amr_mesh.get_level_min(), amr_mesh.get_level_max() );

    int error_count = 0;
    Kokkos::parallel_reduce( "Check 2:1", nbOcts,
      KOKKOS_LAMBDA( uint32_t iOct, int& error_count_local )
      {
        LightOctree::OctantIndex iOct_c = {iOct, false};
        int current_level = lmesh.getLevel(iOct_c);
        int z_off_max = (dim==3)? 1 : 0;
        for( int32_t nz=-z_off_max; nz<=z_off_max; nz++ )
          for( int32_t ny=-1; ny<=1; ny++ )
              for( int32_t nx=-1; nx<=1; nx++ )
                if( nx!=0 || ny!=0 || nz!=0 )
                {
                  LightOctree::offset_t offset = {nx,ny,nz};
                  if( !lmesh.isBoundary( iOct_c, offset ) )
                  {
                    LightOctree::OctantIndex iOct_neighbor = lmesh.findNeighbor( iOct_c, offset );
                    LightOctree_tools::foreach_neighbor_octant( lmesh, iOct_c, iOct_neighbor, offset,
                      [&]( const LightOctree::OctantIndex& iOct_neighbor_i )
                    {
                      int neighbor_level = lmesh.getLevel(iOct_neighbor_i);
                      //EXPECT_TRUE( std::abs(neighbor_level - current_level ) <= 1 );
                      if( abs(neighbor_level - current_level ) > 1 )
                        error_count_local++;
                    });
                  }
                }
      }, error_count);

    EXPECT_EQ( 0, error_count );

  }


  timers.print();
}

} // namespace dyablo

using namespace dyablo;

template< typename AMRmesh_t_ >
class Test_AMRmesh
  : public testing::Test
{
public:
  using AMRmesh_t = AMRmesh_t_;
};

using AMRmesh_types = ::testing::Types<AMRmesh>;
TYPED_TEST_SUITE( Test_AMRmesh, AMRmesh_types );

TYPED_TEST(Test_AMRmesh, narrow_h6_3D)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 6;
  td.width = 5;
  run_test<typename TestFixture::AMRmesh_t>(td);
}

TYPED_TEST(Test_AMRmesh, narrow_h6_2D)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 6;
  td.width = 5;
  td.ndim = 2;
  run_test<typename TestFixture::AMRmesh_t>(td);
}

TYPED_TEST(Test_AMRmesh, narrow_h6_3D_nonsquare)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 6;
  td.width = 5;
  td.coarse_size_x = 16;
  td.coarse_size_y = 8;
  td.coarse_size_z = 4;
  run_test<typename TestFixture::AMRmesh_t>(td);
}

TYPED_TEST(Test_AMRmesh, narrow_h6_2D_nonsquare)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 6;
  td.width = 5;
  td.ndim = 2;
  td.coarse_size_x = 8;
  td.coarse_size_y = 16;
  run_test<typename TestFixture::AMRmesh_t>(td);
}

TYPED_TEST(Test_AMRmesh, narrow_h18)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 18;
  td.width = 5;
  run_test<typename TestFixture::AMRmesh_t>(td);
}

TYPED_TEST(Test_AMRmesh, narrow_h20)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 20;
  td.width = 5;
  run_test<typename TestFixture::AMRmesh_t>(td);
}

TYPED_TEST(Test_AMRmesh, wide_h10)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 10;
  td.width = 15;
  run_test<typename TestFixture::AMRmesh_t>(td);
}

TYPED_TEST(Test_AMRmesh, wide_h10_2D)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 10;
  td.width = 15;
  td.ndim = 2;
  run_test<typename TestFixture::AMRmesh_t>(td);
}