#pragma once

#include "kokkos_shared.h"
#include <map>
#include <set>

#include <iostream>

namespace dyablo{

class MpiBufferPool{

struct alloc_t
{
  size_t size;
  void* ptr;

  bool operator<( const alloc_t& o ) const
  {
    return std::make_pair(size, ptr) < std::make_pair(o.size, o.ptr);
  }
};

struct pool_t
{
  std::map< void*, size_t > used_allocs;
  std::set< alloc_t > free_allocs;
};

template< typename memory_space >
static pool_t& get_pool()
{
  static pool_t pool;
  return pool;
}

public:
  static inline MpiBufferPool& get_MpiBufferPool()
  {
    static MpiBufferPool pool;
    return pool;
  }

  template< typename View_t, typename... IntType >
  View_t alloc( const std::string& name, IntType&... extents )
  {
    using value_type = typename View_t::value_type;
    using memory_space = typename View_t::memory_space;
    auto& pool = get_pool<memory_space>();

    size_t req_size = ((extents * ...)) * sizeof( value_type );    
    
    if(req_size > 0)
    {
      alloc_t alloc{};
      if( !pool.free_allocs.empty() )
      {
        alloc = *pool.free_allocs.rbegin(); // Get the biggest alloc
        pool.free_allocs.erase(alloc);
      }

      if( alloc.size <= req_size )
      {
        if( alloc.ptr )
          Kokkos::kokkos_free<memory_space>( alloc.ptr );
        
        std::cout << "MPI buffers realloc " << alloc.size << " -> " << req_size * 1.3 << std::endl;

        alloc.size = req_size * 1.3;
        alloc.ptr = Kokkos::kokkos_malloc<memory_space>( name, alloc.size );
      }

      pool.used_allocs.insert( std::make_pair( alloc.ptr, alloc.size ) );

      return View_t( static_cast<value_type*>(alloc.ptr), extents... );
    }
    else 
      return View_t( "name", extents... );
    
  }

  template< typename View_t >
  void free( View_t& view )
  {
    if( view.size() > 0 )
    {
      void* ptr = view.data();
      
      using memory_space = typename View_t::memory_space;
      auto& pool = get_pool<memory_space>();
      
      alloc_t alloc{
        .size = pool.used_allocs.at( ptr ),
        .ptr = ptr
      };

      pool.used_allocs.erase( ptr );
      pool.free_allocs.insert( alloc );
    }
    view = View_t();
  }
};

}