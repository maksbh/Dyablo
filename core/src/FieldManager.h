#pragma once

#include <unordered_map>
#include <set>

#include "kokkos_shared.h"
#include "VarIndex.h"
#include "utils/config/ConfigMap.h"


namespace dyablo {

//! a convenience alias to map id to variable names
using int2str_t = std::unordered_map<int,std::string>;
//! a convenience alias to map variable names to id
using str2int_t = std::unordered_map<std::string,int>;

/**
 * a convenience alias to map id (enum) to index used in DataArray
 **/
class id2index_t{
public:
  static constexpr int MAX_INDEX_COUNT = 16; 
private:
  Kokkos::Array < int, MAX_INDEX_COUNT > id2index {};
  Kokkos::Array < bool,MAX_INDEX_COUNT > field_enabled {};
  int _nbfields = 0;
  bool _identity = false; // Special case with unlimited index
public:
  id2index_t()
  {}

  id2index_t( VarIndex index_count )
  :_nbfields( index_count ), _identity( true )
  {}

  void activate( VarIndex id )
  {
    DYABLO_ASSERT_HOST_RELEASE( !_identity, "Cannot edit identity id2index" );
    // DYABLO_ASSERT_ASSERT used because function is constexpr
    DYABLO_ASSERT_HOST_RELEASE( (int)id < MAX_INDEX_COUNT, 
      "VarIndex too big : id=" << id << " >= MAX_INDEX_COUNT=" << MAX_INDEX_COUNT);
    id2index[(int)id] = _nbfields;
    DYABLO_ASSERT_HOST_RELEASE(!field_enabled[(int)id], "Field already enabled" );
    field_enabled[(int)id] = true;
    _nbfields++;
  }
  void activate( VarIndex id, int field_index )
  {
    DYABLO_ASSERT_HOST_RELEASE( !_identity, "Cannot edit identity id2index" );
    DYABLO_ASSERT_HOST_RELEASE( (int)id < MAX_INDEX_COUNT, 
      "VarIndex too big : id=" << id << " >= MAX_INDEX_COUNT=" << MAX_INDEX_COUNT);
    id2index[(int)id] = field_index;
    DYABLO_ASSERT_HOST_RELEASE(!field_enabled[(int)id], "Field already enabled" );
    field_enabled[(int)id] = true;
    _nbfields++;
  }
  std::set<VarIndex> enabled_fields() const
  {
    std::set<VarIndex> res;
    if(_identity)
    {
      for(int i=0; i<nbfields(); i++)
        res.insert( (VarIndex)i );
    }
    else
    {
      for( int i=0; i<MAX_INDEX_COUNT; i++ )
        if( field_enabled[i] ) res.insert( (VarIndex)i );
    }
    return res;
  }
  KOKKOS_INLINE_FUNCTION
  int nbfields() const 
  {
    return _nbfields;
  }

  KOKKOS_INLINE_FUNCTION
  bool enabled(VarIndex id) const
  {
    if(_identity)
      return id < _nbfields;
    else
      return field_enabled[(int)id];
  }

  KOKKOS_INLINE_FUNCTION
  int operator[](VarIndex id) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( enabled(id), "This variable is not active");
    if(_identity)
      return (int)id;
    else
      return id2index[(int)id];
  }
};

/**
 * Field manager class.
 *
 * Initialize a std::unordered_map object to map enum ComponentIndex 
 * to an actual integer depending on runtime configuration (e.g. is 
 * MHD / magnetic field components valid, etc...).
 */
class FieldManager {  
public:
  /**
   * Create a new FieldManager with specific fields
   * All VarIndex must be created befor this constructor with getiVar()
   **/
  FieldManager( const std::set<VarIndex>& active_fields ) 
  {
    for( VarIndex id : active_fields )
    {
      id2index.activate(id);
    }
  }

  FieldManager( const std::initializer_list<VarIndex>& active_fields = {} ) 
    : FieldManager( std::set<VarIndex>( active_fields.begin(), active_fields.end() ) )
  {}

  /**
   * Create a new FieldManager with unnamed fields
   * Generated VarIndexes can be fetched using enabled_fields()
   * VarIndexes generated this way should not be used with var_name()
   * This is usually used for temporary arrays when VarIndexes don't need to be conserved between kernels
   **/
  FieldManager( VarIndex count ) 
    : id2index(count)
  {  }

  id2index_t get_id2index() const
  { 
    return id2index; 
  }

  int nbfields() const
  { 
    return id2index.nbfields(); 
  }

  std::set< VarIndex > enabled_fields() const
  {
    return id2index.enabled_fields();
  }
private:
  id2index_t id2index;
};

} // namespace dyablo 

