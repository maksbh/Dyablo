#pragma once

#include "kokkos_shared.h"
#include <memory>

namespace dyablo{

class MpiBufferPool{
private:
  struct Pdata;
  std::unique_ptr<Pdata> pdata;

private:
  template<typename memory_space_t>
  void* alloc_impl( size_t size );

  template<typename memory_space_t>
  void free_impl( void* ptr );

public:
  MpiBufferPool();
  ~MpiBufferPool();
  void clear();

  template< typename View_t, typename... IntType >
  View_t MPI_Alloc_view( const std::string& name, const IntType&... extents )
  {
    using value_type = typename View_t::value_type;
    using memory_space = typename View_t::memory_space;

    size_t req_size = ((extents * ...)) * sizeof( value_type );    
    
    if(req_size > 0)
    {
      void* ptr = alloc_impl<memory_space>(req_size);
      return View_t( static_cast<value_type*>(ptr), extents... );
    }
    else 
      return View_t( "name", extents... );
  }

  template< typename View_t >
  void MPI_Free_view( View_t& view )
  {
    if( view.size() > 0 )
    {
      void* ptr = view.data();
      using memory_space = typename View_t::memory_space;
      free_impl<memory_space>(ptr);
    }
    view = View_t();
  }
};

extern template void MpiBufferPool::free_impl<Kokkos::DefaultHostExecutionSpace::memory_space>( void* ptr );
extern template void* MpiBufferPool::alloc_impl<Kokkos::DefaultHostExecutionSpace::memory_space>( size_t size );

}