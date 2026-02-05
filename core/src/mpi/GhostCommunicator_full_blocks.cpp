#include "GhostCommunicator_full_blocks.h"

namespace dyablo {

void GhostCommunicator_full_blocks::exchange_ghosts( const UserData::FieldAccessor& U ) const
{
  DYABLO_ASSERT_HOST_DEBUG( !this->intermediates, "Trying to echange intermediate ghosts but FieldAccessor don't have intermediates" );
  auto &fields = U.fields;

  for(int i=0; i<U.nbFields(); i++)
  {
    int iVar = U.get_index_from_ivar_host(i);
    auto U_subview      = Kokkos::subview( fields.U,      Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );
    auto Ughost_subview = Kokkos::subview( fields.Ughost, Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );

    ViewCommunicator::exchange_ghosts<2>(U_subview, Ughost_subview);
  }
}

void GhostCommunicator_full_blocks::exchange_ghosts( const UserData::FieldAccessor_intermediates& U ) const
{
  auto &fields = this->intermediates?U.fields_intermediates:U.fields;

  for(int i=0; i<U.nbFields(); i++)
  {
    int iVar = this->intermediates?U.get_index_from_ivar_host_intermediates(i):U.get_index_from_ivar_host(i);
    auto U_subview      = Kokkos::subview( fields.U,      Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );
    auto Ughost_subview = Kokkos::subview( fields.Ughost, Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );

    ViewCommunicator::exchange_ghosts<2>(U_subview, Ughost_subview);
  }
}

void GhostCommunicator_full_blocks::exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
{
  ViewCommunicator::exchange_ghosts<2>(U.U, U.Ughost);
}


void OctSubset_init( MpiComm mpi_comm, 
  const Kokkos::View<uint32_t*>& recv_sizes_full,
  const Kokkos::View<uint32_t*>& send_iOcts_full,
  const Kokkos::View<uint32_t*>& send_sizes_full,
  std::unique_ptr<ViewCommunicator>& partial_comm,
  Kokkos::View<uint32_t*>& subset_iOcts)
{
  int nbProc = mpi_comm.MPI_Comm_size();

  uint32_t nbGhosts_subset = subset_iOcts.size();  

  // Number of subset octants requested from each process
  Kokkos::View<int*> subset_send_request_count_device("subset_send_request_count", nbProc);
  // Subset Octants requested from remote ranks, position is local to remote rank's local octant list
  Kokkos::View<uint32_t*> subset_send_request_iOcts_remotelocal("subset_send_request_iOcts_remotelocal",nbGhosts_subset);
  Kokkos::parallel_for( "OctSubset::convert_subset_to_remote", subset_iOcts.size(),
    KOKKOS_LAMBDA( uint32_t i )
  {
    uint32_t iOct_ghost_subset = subset_iOcts(i);

    uint32_t rank_iOct_end = 0;
    int rank = 0;
    for( rank=0; rank < nbProc; rank++ )
    {
      rank_iOct_end += recv_sizes_full(rank);
      if( iOct_ghost_subset < rank_iOct_end )
        break;        
    }
    uint32_t rank_iOct_begin = rank_iOct_end - recv_sizes_full(rank);
    DYABLO_ASSERT_KOKKOS_DEBUG( rank_iOct_begin <= iOct_ghost_subset && iOct_ghost_subset < rank_iOct_end, "OctSubset error - could not find subset octant rank" );
    Kokkos::atomic_inc(&subset_send_request_count_device(rank));
    subset_send_request_iOcts_remotelocal(i) = iOct_ghost_subset-rank_iOct_begin;
  });
  
  auto subset_send_request_count_host = Kokkos::create_mirror_view(subset_send_request_count_device);
  Kokkos::deep_copy(subset_send_request_count_host, subset_send_request_count_device);
  // Number of subset octants requested by each proc
  Kokkos::View<int*>::host_mirror_type subset_recv_request_count_host("subset_recv_request_count", nbProc);
  mpi_comm.MPI_Alltoall( subset_send_request_count_host.data(), 1, subset_recv_request_count_host.data(), 1 );
  uint32_t requested_local_total = std::reduce(subset_recv_request_count_host.data(), subset_recv_request_count_host.data()+nbProc);
  
  // Subset of local octants to send (index in ghost_comm.send_iOct)
  Kokkos::View<uint32_t*> subset_recv_request_iOcts_ghosts("subset_recv_request_iOcts_ghosts", requested_local_total);
  #ifdef MPI_IS_CUDA_AWARE 
  {
    mpi_comm.MPI_Alltoallv( subset_send_request_iOcts_remotelocal.data(), subset_send_request_count_host.data(),
                            subset_recv_request_iOcts_ghosts.data(), subset_recv_request_count_host.data() );
  }
  #else
  {
    auto subset_send_request_iOcts_remotelocal_host = Kokkos::create_mirror_view(subset_send_request_iOcts_remotelocal);
    auto subset_recv_request_iOcts_ghosts_host = Kokkos::create_mirror_view(subset_recv_request_iOcts_ghosts);
    Kokkos::deep_copy(subset_send_request_iOcts_remotelocal_host, subset_send_request_iOcts_remotelocal);
    mpi_comm.MPI_Alltoallv( subset_send_request_iOcts_remotelocal_host.data(), subset_send_request_count_host.data(),
                            subset_recv_request_iOcts_ghosts_host.data(), subset_recv_request_count_host.data() );
    Kokkos::deep_copy(subset_recv_request_iOcts_ghosts, subset_recv_request_iOcts_ghosts_host);
  }
  #endif
  
  Kokkos::View<uint32_t*> send_iOcts_sizes("send_iOcts_sizes", nbProc);
  auto send_iOcts_sizes_host = Kokkos::create_mirror_view(send_iOcts_sizes);
  Kokkos::deep_copy( send_iOcts_sizes_host, subset_recv_request_count_host ); // Cast from int to uint32_t
  Kokkos::deep_copy( send_iOcts_sizes, send_iOcts_sizes_host );
  Kokkos::View<uint32_t*> send_iOcts("send_iOcts", requested_local_total);

  auto send_sizes_full_host = Kokkos::create_mirror_view(send_sizes_full);
  Kokkos::deep_copy(send_sizes_full_host, send_sizes_full);
  
  int i_rank_begin = 0;
  int iGhost_rank_begin = 0;
  for( int rank=0; rank<nbProc; rank++ )
  {
    Kokkos::parallel_for("OctSubset::unpack_requested_octs", send_iOcts_sizes_host(rank),
      KOKKOS_LAMBDA( int i_rank )
    {
      int i = i_rank + i_rank_begin;
      // Position in full send_buffer
      uint32_t iGhost = subset_recv_request_iOcts_ghosts(i) + iGhost_rank_begin;
      // Requested local octant 
      uint32_t iOct_local = send_iOcts_full( iGhost );
      send_iOcts(i) = iOct_local;
    });
    i_rank_begin += send_iOcts_sizes_host(rank);
    iGhost_rank_begin += send_sizes_full_host(rank);
  }
  partial_comm = std::make_unique<ViewCommunicator>( send_iOcts_sizes, send_iOcts );
  DYABLO_ASSERT_HOST_RELEASE( partial_comm->getNumGhosts() == subset_iOcts.size(), "GhostCommunicator_full_blocks::OctSubset : ghost count mismatch" );
}

GhostCommunicator_full_blocks::OctSubset::OctSubset(const GhostCommunicator_full_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts )
  : subset_iOcts( subset_iOcts )
{
  // nvcc doesnt like kokkos kernels in constructors
  OctSubset_init( 
    comm_full.mpi_comm, 
    comm_full.recv_sizes,
    comm_full.send_iOcts,
    comm_full.send_sizes,
    this->partial_comm,
    this->subset_iOcts
   );
}

void GhostCommunicator_full_blocks::exchange_ghosts_subset( const UserData::FieldAccessor& U, const OctSubset& subset ) const
{
  DYABLO_ASSERT_HOST_DEBUG( !this->intermediates, "Trying to echange intermediate ghosts but FieldAccessor don't have intermediates" );
  auto &fields = U.fields;

  int nbCells = fields.U.extent(0);
  int nbFields = U.nbFields();

  Kokkos::View<real_t***, Kokkos::LayoutLeft> Ughost_subset( "Ughost_subset", nbCells, nbFields, subset.nbGhosts() );
  for(int i=0; i<U.nbFields(); i++)
  {
    int iVar = U.get_index_from_ivar_host(i);
    auto U_subview      = Kokkos::subview( fields.U,    Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );
    auto Ughost_subview = Kokkos::subview( Ughost_subset, Kokkos::ALL(), std::make_pair(i, i+1),       Kokkos::ALL() );
    
    subset.partial_comm->exchange_ghosts<2>( U_subview, Ughost_subview );
  }

  auto& subset_iOcts = subset.subset_iOcts;

  Kokkos::parallel_for( "exchange_ghosts_subset::unpack_ghosts", nbCells*nbFields*subset.nbGhosts(),
                    KOKKOS_LAMBDA(uint32_t index)
  {
    uint32_t iOct_src = index/(nbCells*nbFields);
    uint32_t iOct_dest = subset_iOcts(iOct_src);
    uint32_t i = index%(nbCells*nbFields);
    uint32_t ivar_src = i/nbCells;
    uint32_t ivar_dest = U.get_index_from_ivar_device(ivar_src);
    uint32_t iblock = i%nbCells;

    fields.Ughost( iblock, ivar_dest, iOct_dest ) = Ughost_subset( iblock, ivar_src, iOct_src );
  });      
}

void GhostCommunicator_full_blocks::exchange_ghosts_subset( const UserData::FieldAccessor_intermediates& U, const OctSubset& subset ) const
{
  auto &fields = this->intermediates?U.fields_intermediates:U.fields;

  int nbCells = fields.U.extent(0);
  int nbFields = U.nbFields();
  const bool isIntermediate = this->intermediates;

  Kokkos::View<real_t***, Kokkos::LayoutLeft> Ughost_subset( "Ughost_subset", nbCells, nbFields, subset.nbGhosts() );
  for(int i=0; i<U.nbFields(); i++)
  {
    int iVar = this->intermediates?U.get_index_from_ivar_host_intermediates(i):U.get_index_from_ivar_host(i);
    auto U_subview      = Kokkos::subview( fields.U,    Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );
    auto Ughost_subview = Kokkos::subview( Ughost_subset, Kokkos::ALL(), std::make_pair(i, i+1),       Kokkos::ALL() );
    
    subset.partial_comm->exchange_ghosts<2>( U_subview, Ughost_subview );
  }

  auto& subset_iOcts = subset.subset_iOcts;

  Kokkos::parallel_for( "exchange_ghosts_subset::unpack_ghosts", nbCells*nbFields*subset.nbGhosts(),
                    KOKKOS_LAMBDA(uint32_t index)
  {
    uint32_t iOct_src = index/(nbCells*nbFields);
    uint32_t iOct_dest = subset_iOcts(iOct_src);
    uint32_t i = index%(nbCells*nbFields);
    uint32_t ivar_src = i/nbCells;
    uint32_t ivar_dest = isIntermediate? U.get_index_from_ivar_device_intermediates(ivar_src) : U.get_index_from_ivar_device(ivar_src);
    uint32_t iblock = i%nbCells;

    fields.Ughost( iblock, ivar_dest, iOct_dest ) = Ughost_subset( iblock, ivar_src, iOct_src );
  });
}


void GhostCommunicator_full_blocks::reduce_ghosts( UserData::FieldAccessor& U ) const
{
  for(int i=0; i<U.nbFields(); i++)
  {
    int iVar = U.get_index_from_ivar_host(i);
    auto U_subview      = Kokkos::subview( U.fields.U,      Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );
    auto Ughost_subview = Kokkos::subview( U.fields.Ughost, Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );

    ViewCommunicator::reduce_ghosts<2>(U_subview, Ughost_subview);
  }
}

void GhostCommunicator_full_blocks::reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
{
  ViewCommunicator::reduce_ghosts<2>(U.U, U.Ughost);
}  

} // namespace dyablo