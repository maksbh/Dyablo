#include "GhostCommunicator_full_blocks.h"

namespace dyablo {

void GhostCommunicator_full_blocks::exchange_ghosts( const UserData::FieldAccessor& U ) const
{
  for(int i=0; i<U.nbFields(); i++)
  {
    int iVar = U.get_index_from_ivar_host(i);
    auto U_subview      = Kokkos::subview( U.fields.U,      Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );
    auto Ughost_subview = Kokkos::subview( U.fields.Ughost, Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );

    ViewCommunicator::exchange_ghosts<2>(U_subview, Ughost_subview);
  }
}

void GhostCommunicator_full_blocks::exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
{
  ViewCommunicator::exchange_ghosts<2>(U.U, U.Ughost);
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