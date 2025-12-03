/**
 * \file test_CommGhosts.cpp
 * \author A. Durocher
 * Test intermediate cells computations for full tree
 * 
 * Leaves are filled with cell center positions, intermediate cells are filled with average values from children, 
 * Intermediate cells should have cell center as values.
 */
#include "gtest/gtest.h"

#include "mpi/ViewCommunicator.h"

#include "amr/AMRmesh.h"
#include "utils/io/AMRMesh_output_vtk.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

#include "mpi/GhostCommunicator.h"
#include "UserData.h"
#include "utils/config/ConfigMap.h"

namespace dyablo {
namespace {

using OctantIndex = LightOctree::OctantIndex;
using CellIndex = ForeachCell::CellIndex;
using CellArray_shape = ForeachCell::CellArray_shape;

void IterationSpace_level_init( uint32_t& nbIntermediates,
                                Kokkos::View<uint32_t*>& iOcts,
                                const LightOctree& lmesh, 
                                int filter_level)
{
  // Count intermediate octants at required level
  Kokkos::parallel_reduce( "count_intermediate_octs_level", lmesh.getNumIntermediates(),  
    KOKKOS_LAMBDA( uint32_t iOct, uint32_t& count )
  {
    int oct_level = lmesh.getLevel( {.iOct=iOct, .isGhost=false, .isIntermediate=true} );
    if( oct_level == filter_level )
      count++;
  }, nbIntermediates);

  // Allocate filtered Octs array
  iOcts = Kokkos::View<uint32_t*>( "IterationSpace_level::iOcts", nbIntermediates );

  // Fill Oct array with local intermediates
  Kokkos::parallel_scan( "fill_intermediate_octs_level", lmesh.getNumIntermediates(),  
    KOKKOS_LAMBDA( uint32_t iOct, uint32_t& count, bool final )
  {
    int oct_level = lmesh.getLevel( {.iOct=iOct, .isGhost=false, .isIntermediate=true} );
    if( oct_level == filter_level )
    {
      if( final )
        iOcts( count ) = iOct;
      count++;
    }
  });
}

class IterationSpace_level_intermediates
{
private:
  uint32_t _bx, _by, _bz;
  Kokkos::View<uint32_t*> _iOcts;
  uint32_t _nbIntermediates;
public:
  IterationSpace_level_intermediates(const CellArray_shape& shape, const LightOctree& lmesh, int level)
  : _bx(shape.bx),_by(shape.by),_bz(shape.bz)
  {
    IterationSpace_level_init( _nbIntermediates, _iOcts, lmesh, level );
  }

  KOKKOS_INLINE_FUNCTION
  uint32_t bx() const         { return _bx; }
  KOKKOS_INLINE_FUNCTION
  uint32_t by() const         { return _by; }
  KOKKOS_INLINE_FUNCTION
  uint32_t bz() const         { return _bz; }

  KOKKOS_INLINE_FUNCTION
  uint32_t iOct_count() const { return _iOcts.size();}

  KOKKOS_INLINE_FUNCTION
  CellIndex getCellIndex(uint32_t iOct_raw, uint32_t i, uint32_t j, uint32_t k) const
  {
    uint32_t iOct_filtered = _iOcts(iOct_raw);
    OctantIndex iOct{ 
      .iOct=iOct_filtered,
      .isGhost=false, 
      .isIntermediate=true
    };
    CellIndex iCell = {iOct, i, j, k, bx(), by(), bz()};
    return iCell;
  }
};

} // namespace
} // namespace dyablo


/***
 * Fill cells with cell centers, average to parent cells until level_min and check if value is still cell center. 
 * Values are propagated with getParent and atomic_add through the whole local tree, MPI ghosts are then reduced
 ***/
void test_FullTree_average_children()
{
  using namespace dyablo;

  std::cout << "// =========================================\n";
  std::cout << "// Testing FullTree Average ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  int ndim = 3;
  int level_min = 3;
  int level_max = 7;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, std::array<bool,3>{false,false,false}, level_min, level_max);
    //amr_mesh->setBalanceCodimension(ndim);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);

    int local_rank = amr_mesh->getMpiComm().MPI_Comm_rank();

    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();

    amr_mesh->loadBalance(0);
  }

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  ConfigMap configMap ("");
  configMap.getValue<uint32_t>("amr", "bx", bx);
  configMap.getValue<uint32_t>("amr", "by", by);
  configMap.getValue<uint32_t>("amr", "bz", bz);
  
  ForeachCell foreach_cell(*amr_mesh, configMap);  
  UserData U ( configMap, foreach_cell );

  U.new_fields({"px", "dummy", "py", "pz"});

  std::cout << "Initialize User Data..." << std::endl;

  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  using pos_t = ForeachCell::CellMetaData::pos_t;

  { // Initialize U
    enum VarIndex_test{Px,Py,Pz,Dummy};

    UserData::FieldAccessor Uin = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}, {"dummy", Dummy}} );
    foreach_cell.foreach_cell( "Init_U", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto c = cells.getCellCenter( iCell );
      Uin.at(iCell, Px) = c[IX];
      Uin.at(iCell, Py) = c[IY];
      Uin.at(iCell, Pz) = c[IZ];
      Uin.at(iCell, Dummy) = 99;
    });
  }

  U.new_intermediate_fields( {"px","py","pz"} );

  enum VarIndex_test{Px,Py,Pz};
  UserData::FieldAccessor_intermediates Ua = U.getAccessor_intermediates( {{"px", Px}, {"py", Py}, {"pz", Pz}} );

  const LightOctree& lmesh = amr_mesh->getLightOctree();
  GhostCommunicator_full_blocks ghost_communicator_leaves( *amr_mesh, Ua.getShape(), -1, false );
  GhostCommunicator_full_blocks ghost_communicator_intermediates( *amr_mesh, Ua.getShape(), -1, true );

  for( int level = level_max-1; level >= level_min; level-- )
  {
    #warning TODO : filter only intermediates
    ghost_communicator_leaves.exchange_ghosts( Ua );
    ghost_communicator_intermediates.exchange_ghosts( Ua );

    IterationSpace_level_intermediates iter_space_level_intermediates(Ua.getShape(), lmesh, level);
    foreach_cell.foreach_cell("average_parent_cell", iter_space_level_intermediates,
      KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
    {
      ForeachCell::SearchMode_intermediates search_neighbor_intermediate(lmesh, ForeachCell::SearchMode_intermediates::BiggerNeighborMode::ASSERT);

      ForeachCell::CellIndex iCell_c0 = iCell.getChildren(lmesh);

      pos_t p{};
      int ns = foreach_sibling( ndim, iCell_c0, search_neighbor_intermediate,
        [&]( const ForeachCell::CellIndex& iCell_c )
      {
        real_t px = Ua.at(iCell_c, Px);
        real_t py = Ua.at(iCell_c, Py);
        real_t pz = Ua.at(iCell_c, Pz);

        p[IX] += px;
        p[IY] += py;
        p[IZ] += pz;
      });

      Ua.at( iCell, Px ) = p[IX]/ns;
      Ua.at( iCell, Py ) = p[IY]/ns;
      Ua.at( iCell, Pz ) = p[IZ]/ns;
    }); 
    
  }
  
  constexpr bool with_locals = true;
  constexpr bool without_ghosts = false;
  constexpr bool with_intermediates = true;
  ForeachCell::IterationSpace_fullArray_impl<with_locals, without_ghosts, with_intermediates> iter_space_with_intermediates(Ua.getShape());
  int error_count = 0;
  foreach_cell.reduce_cell( "test_values", iter_space_with_intermediates,
    KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell, int& error_count )
  {
    auto test_equal = [&](real_t a, real_t b)
    {
      //EXPECT_DOUBLE_EQ(a, b);
      if( a != b )
      {
        error_count++;
        printf("%f != %f\n", a, b);
      }
    };

    auto pos = cells.getCellCenter(iCell);
    test_equal( pos[IX], Ua.at(iCell, Px) );
    test_equal( pos[IY], Ua.at(iCell, Py) );
    test_equal( pos[IZ], Ua.at(iCell, Pz) );    
  }, error_count);

  EXPECT_EQ(0, error_count);
}

TEST(dyablo, test_FullTree_average_children)
{
  using namespace dyablo;
  test_FullTree_average_children();
}