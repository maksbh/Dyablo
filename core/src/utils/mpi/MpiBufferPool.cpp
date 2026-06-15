#include "MpiBufferPool.h"

#include <map>
#include <set>
#include <iostream>

namespace dyablo{

namespace {

struct alloc_t
{
  size_t size;
  void* ptr;

  bool operator<( const alloc_t& o ) const
  {
    return std::make_pair(size, ptr) < std::make_pair(o.size, o.ptr);
  }
};

template< class memory_space >
class pool_t
{
private:
  std::map< void*, size_t > used_allocs;
  std::set< alloc_t > free_allocs;

public:
  pool_t() = default;
  ~pool_t()
  {
    clear();
  }

  void clear()
  {
    if( used_allocs.size() != 0 )
    {
      std::cout << "WARNING - ~MpiBufferPool : some allocations are still in use." << std::endl;
    }
    
    for( const alloc_t& a : free_allocs )
    {
      Kokkos::kokkos_free<memory_space>( a.ptr );
    }
  }

  void* alloc( size_t req_size )
  {
    alloc_t alloc{};
    if( !this->free_allocs.empty() )
    {
      alloc = *this->free_allocs.rbegin(); // Get the biggest alloc
      this->free_allocs.erase(alloc);
    }

    if( alloc.size <= req_size )
    {
      if( alloc.ptr )
        Kokkos::kokkos_free<memory_space>( alloc.ptr );
      
      //std::cout << "MPI buffers realloc " << alloc.size << " -> " << req_size * 1.3 << std::endl;

      alloc.size = req_size * 1.3;
      alloc.ptr = Kokkos::kokkos_malloc<memory_space>( "MPI Buffer", alloc.size );
    }

    this->used_allocs.insert( std::make_pair( alloc.ptr, alloc.size ) );

    return alloc.ptr;
  }

  void free( void* ptr )
  {   
    alloc_t alloc{
      .size = this->used_allocs.at( ptr ),
      .ptr = ptr
    };

    this->used_allocs.erase( ptr );
    this->free_allocs.insert( alloc );
  }
};

using HostMemSpace = Kokkos::DefaultHostExecutionSpace::memory_space;
using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

template <typename T, typename... Ts>
struct unique : std::type_identity<T> {};

template <typename... Ts, typename U, typename... Us>
struct unique<std::tuple<Ts...>, U, Us...>
    : std::conditional_t<(std::is_same_v<U, Ts> || ...)
                       , unique<std::tuple<Ts...>, Us...>
                       , unique<std::tuple<Ts..., U>, Us...>> {};

template <typename... Ts>
using unique_tuple = typename unique<std::tuple<>, Ts...>::type;

} // namespace

struct MpiBufferPool::Pdata
{
  unique_tuple< 
    pool_t<HostMemSpace>,
    pool_t<DeviceMemSpace>
  > pools;
};

MpiBufferPool::MpiBufferPool()
{/*empty*/}

MpiBufferPool::~MpiBufferPool()
{
  std::apply(
    [](auto&&... args) 
      {((args.clear()), ...);},
    this->pdata->pools );
}
  
template<typename memory_space_t>
void* MpiBufferPool::alloc_impl( size_t size )
{ 
  auto& pool = std::get<pool_t<memory_space_t>>(pdata->pools);
  return pool.alloc(size);
}

template<typename memory_space_t>
void MpiBufferPool::free_impl( void* ptr )
{ 
  auto& pool = std::get<pool_t<memory_space_t>>(pdata->pools);
  return pool.free(ptr);
}

template void MpiBufferPool::free_impl<HostMemSpace>( void* ptr );
template void* MpiBufferPool::alloc_impl<HostMemSpace>( size_t size );

} // namespace dyablo