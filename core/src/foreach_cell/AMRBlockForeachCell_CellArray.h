#pragma once

#include "kokkos_shared.h"
#include "amr/LightOctree.h"

namespace dyablo {
namespace AMRBlockForeachCell_CellArray_impl{

struct  CellArray_shape;

/// Invalid index to rreturn as error value, use CellIndex::is_valid() to check for validity
#define CELLINDEX_INVALID CellIndex{{0,true},0,0,0,0,0,0,CellIndex::INVALID}

/**
 * Search mode for neighbors : don't search outside of local block
 * ErrorMode describes the behavior when target cell is outside of current block:
 * - ASSERT : convert_index fails/crashes (assert in debug, UB in release)
 * - INVALID : convert_index returns CELLINDEX_INVALID
 **/
class SearchMode_local
{
public:
  enum ErrorMode {ASSERT, INVALID};

  KOKKOS_INLINE_FUNCTION
  SearchMode_local( ErrorMode error_mode )
  : _error_mode(error_mode)
  {}
  SearchMode_local( const SearchMode_local& ) = default;

  KOKKOS_INLINE_FUNCTION
  ErrorMode error_mode() const
  {
    return _error_mode;
  }

private:
  ErrorMode _error_mode;
};

/**
 * Search mode for neighbors : use LightOctree to find neighbors
 * SmallerNeighborMode describes which cell is returned when neighbor is smaller
 * (big cell is origin, x is returned neighbor)
 * - ORIGIN : return the top-right cell of the corresponding virtual same-size cell
 *     _ _ ___
 *    |x|_|   |
 *    |_|_|___|
 * - CLOSEST : return the smaller cell closest to the initial cell
 *     _ _ ___
 *    |_|x|   |
 *    |_|_|___|
 */
class SearchMode_neighbor
{
public:
  enum SmallerNeighborMode{ CLOSEST, ORIGIN };
  KOKKOS_INLINE_FUNCTION
  SearchMode_neighbor( const LightOctree& lmesh, SmallerNeighborMode mode )
  : lmesh(lmesh), _smaller_neighbor_mode(mode)
  {}
  SearchMode_neighbor( const SearchMode_neighbor& ) = delete;

  KOKKOS_INLINE_FUNCTION
  SmallerNeighborMode smaller_neighbor_mode() const
  {
    return _smaller_neighbor_mode;
  }

  KOKKOS_INLINE_FUNCTION
  const LightOctree& getLightOctree() const
  {
    return lmesh;
  }
private:
  const LightOctree& lmesh;
  SmallerNeighborMode _smaller_neighbor_mode;
};

/**
 * Search mode to include intermediates in neighbor search.
 * GetNeighbor should return a same size neighbor that can be a leaf or an intermediate cell
 * The behavior when the neighbor is bigger (not refined at current cell level) can be configured with 
 **/
class SearchMode_intermediates
{
public:
  enum BiggerNeighborMode {ASSERT, INVALID, GETLEAF};

  KOKKOS_INLINE_FUNCTION
  SearchMode_intermediates( const LightOctree& lmesh, BiggerNeighborMode mode )
  : lmesh(lmesh), _bigger_neighbor_mode(mode)
  {}
  SearchMode_intermediates( const SearchMode_intermediates& ) = delete;

  KOKKOS_INLINE_FUNCTION
  BiggerNeighborMode bigger_neighbor_mode() const
  {
    return _bigger_neighbor_mode;
  }

  KOKKOS_INLINE_FUNCTION
  const LightOctree& getLightOctree() const
  {
    return lmesh;
  }
private:
  const LightOctree& lmesh;
  BiggerNeighborMode _bigger_neighbor_mode;
};

struct CellIndex
{
  LightOctree::OctantIndex iOct;
  uint32_t i,j,k;
  uint32_t bx,by,bz;
  enum Status {
    LOCAL_TO_BLOCK,
    SAME_SIZE,
    SMALLER,
    BIGGER,
    BOUNDARY,
    INVALID
  } status;

  /**
   * Difference of level between current cell and original cell used in convert_index() or getNeighbor()
   * note : is 0 for indexes that are local to the block
   **/
  KOKKOS_INLINE_FUNCTION
  int level_diff() const
  {
    return (status==BIGGER)-(status==SMALLER);
  }

  /**
   * Search during convert_index() or getNeighbor() resulted in a CellIndex outside 
   * of local block and neighbor search was disabled
   **/
  KOKKOS_INLINE_FUNCTION
  bool is_valid() const
  {
    return (status!=INVALID) && (status!=BOUNDARY);
  }

  /**
   * Returns true when index operation returns a cell outside of domain boundaries.
   **/
  KOKKOS_INLINE_FUNCTION
  bool is_boundary() const
  {
    return status==BOUNDARY;
  }

  /**
   * index operation did not need neighbor search to get this index. 
   * If is_local() == false, this index can only be used to interact with 
   * arrays with ghosts enabled
   **/
  KOKKOS_INLINE_FUNCTION
  bool is_local() const
  {
    return status==LOCAL_TO_BLOCK;
  }

  /**
   * Get local octant index (cannot be ghost)
   **/
  KOKKOS_INLINE_FUNCTION
  uint32_t getOct() const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG(!iOct.isGhost, "iOct must not be ghost");
    return iOct.iOct;
  }

  using offset_t = Kokkos::Array< int16_t, 3 >;

/**
   * Get a position inside the domain and an offset to get to the current boundary cell
   * compute iCell_inside and offset such that *this == iCell_inside.getNeighbor_ghost(offset, ...)
   * this->is_boundary() must be true
   * 
   * @param iCell_inside (out) closest cell inside domain
   * @param offset (out) offset to apply to iCell_inside to get to current cell
   **/
  KOKKOS_INLINE_FUNCTION
  void getBoundaryPosAndOffset(CellIndex& iCell_inside, offset_t& offset) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG(is_boundary(), "iOct must be in boundary");

    int32_t i = (int32_t)this->i - (int32_t)this->bx;
    int32_t j = (int32_t)this->j - (int32_t)this->by;
    int32_t k = (int32_t)this->k - (int32_t)this->bz;

    uint32_t i_inside = FMIN( FMAX( i, (int32_t)0 ), (int32_t)(bx-1) );
    int16_t i_offset = i - i_inside;
    uint32_t j_inside = FMIN( FMAX( j, (int32_t)0 ), (int32_t)(by-1) );
    int16_t j_offset = j - j_inside;
    uint32_t k_inside = FMIN( FMAX( k, (int32_t)0 ), (int32_t)(bz-1) );
    int16_t k_offset = k - k_inside;

    iCell_inside = {
      this->iOct,
      i_inside,j_inside,k_inside,
      this->bx,this->by,this->bz,
      CellIndex::LOCAL_TO_BLOCK 
    };
    offset = {i_offset, j_offset, k_offset};
  }


  /**
   * Compute neighbor cell index
   * 
   * @param offset offset from the original cell
   * @param search_mode configures how to search when neighbor cell is outside of local block
   * - SearchMode_local : does not look for neighbor octs
   * - SearchMode_neighbor : search cell in neighbor octants
   * - SearchMode_intermediates : search in neighbor octant, including intermediate levels
   * (Read SearchMode_* doc for more detail on how to configure them)
   * 
   * Offseting outside the block returns a CellIndex pointing to the 
   * corresponding neighbor octant or an invalid index depending search_mode settings.
   * If a neighbor outside of the local block is returned `is_local()==false`
   * and resulting cell might be non-conforming (`level_diff()` might be != 0).
   * When level_diff() >= 0, result points to the only neighbor cell
   * When level_diff() < 0 (neighbor is smaller), the results depends on search_mode.
   * NOTE: If offset >= 2 outside of the block, resulting cell is one of the subcells in same-size equivalent neighbor 
   * accessing octants that are not direcly contiguous to local octant is undefined behavior, so be careful with block size
   **/
  template<typename SearchMode>
  KOKKOS_INLINE_FUNCTION
  CellIndex getNeighbor( const offset_t& offset, const SearchMode& search_mode ) const
  {
    return getNeighbor(offset[IX], offset[IY], offset[IZ], search_mode);
  }


  template<typename SearchMode>
  KOKKOS_INLINE_FUNCTION
  CellIndex getNeighbor( int16_t offset_x, int16_t offset_y, int16_t offset_z, const SearchMode& search_mode ) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG(this->is_valid(), "Index needs to be valid to get neighbor");

    int32_t i = this->i + offset_x;
    int32_t j = this->j + offset_y;
    int32_t k = this->k + offset_z;

    if( i<0 || i>=(int32_t)bx || j<0 || j>=(int32_t)by || k<0 || k>=(int32_t)bz )
    {
      // Index is outside of block : find neighbor?
      if constexpr (std::is_same_v<SearchMode, SearchMode_local>)
      { 
        // Local only convert, no neighbor search
        if( search_mode.error_mode() == SearchMode_local::ASSERT )
        {
          DYABLO_ASSERT_KOKKOS_DEBUG(i>=0, "i out of block bounds"); DYABLO_ASSERT_KOKKOS_DEBUG(i<(int32_t)bx, "i out of block bounds");
          DYABLO_ASSERT_KOKKOS_DEBUG(j>=0, "j out of block bounds"); DYABLO_ASSERT_KOKKOS_DEBUG(j<(int32_t)by, "j out of block bounds");
          DYABLO_ASSERT_KOKKOS_DEBUG(k>=0, "k out of block bounds"); DYABLO_ASSERT_KOKKOS_DEBUG(k<(int32_t)bz, "k out of block bounds");
        }
        return CELLINDEX_INVALID;
      }
      else if constexpr (  std::is_same_v<SearchMode, SearchMode_neighbor> 
                        || std::is_same_v<SearchMode, SearchMode_intermediates> )
      { 
        // Neighbor search
        const LightOctree& lmesh = search_mode.getLightOctree();

        int16_t oct_offset_x = i/bx;
        int16_t oct_offset_y = j/by;
        int16_t oct_offset_z = k/bz;

        const LightOctree::OctantIndex& iOct = this->iOct;
        if( lmesh.isBoundary( iOct, oct_offset_x, oct_offset_y, oct_offset_z ) )
        {
          return CellIndex{iOct,i+bx,j+by,k+bz,bx,by,bz, CellIndex::BOUNDARY};;
        }

        if constexpr ( std::is_same_v<SearchMode, SearchMode_intermediates>  )
        {
          #warning TODO include level_diff in OctantIndex
          LightOctree::OctantIndex iOct_n = lmesh.findNeighbor_intermediate(iOct, oct_offset_x, oct_offset_y, oct_offset_z);
          // Compute position of cell in neighbor when neighbor is same size;
          uint32_t i_same = i - oct_offset_x * bx;
          uint32_t j_same = j - oct_offset_y * by;
          uint32_t k_same = k - oct_offset_z * bz;
          DYABLO_ASSERT_KOKKOS_DEBUG(i_same<bx, "internal error : i out of block bounds");
          DYABLO_ASSERT_KOKKOS_DEBUG(j_same<by, "internal error : j out of block bounds");
          DYABLO_ASSERT_KOKKOS_DEBUG(k_same<bz, "internal error : k out of block bounds");

          int level_diff = lmesh.getLevel(iOct) - lmesh.getLevel(iOct_n);
          if( level_diff == 1 ) // Neighbor is larger 
          {
            DYABLO_ASSERT_KOKKOS_DEBUG(search_mode.bigger_neighbor_mode() != SearchMode_intermediates::ASSERT, "Could not find intermediate neighbor : not refined enough");
            if( search_mode.bigger_neighbor_mode() == SearchMode_intermediates::INVALID 
             || search_mode.bigger_neighbor_mode() == SearchMode_intermediates::ASSERT )
            {
              return CELLINDEX_INVALID;
            }
            else // Return bigger cell
            {
              // Compute suboctant where target cell is located in larger neighbor
              auto coord_n = lmesh.get_logical_coords(iOct);

              auto is_odd = [](int x) {
                return (int)(x%2 != 0);
              };

              int suboctant_offset_x = is_odd( coord_n[IX] + oct_offset_x );
              int suboctant_offset_y = is_odd( coord_n[IY] + oct_offset_y );
              int suboctant_offset_z = is_odd( coord_n[IZ] + oct_offset_z );

              // Offset to select suboctant and /2 for position in larger octant 
              uint32_t i_larger = (i_same+suboctant_offset_x*bx)/2;
              uint32_t j_larger = (j_same+suboctant_offset_y*by)/2;
              uint32_t k_larger = (k_same+suboctant_offset_z*bz)/2;

              DYABLO_ASSERT_KOKKOS_DEBUG(i_larger<bx, "internal error : i out of block bounds");
              DYABLO_ASSERT_KOKKOS_DEBUG(j_larger<by, "internal error : j out of block bounds");
              DYABLO_ASSERT_KOKKOS_DEBUG(k_larger<bz, "internal error : k out of block bounds");

              CellIndex res{
                iOct_n, 
                i_larger, j_larger, k_larger,
                bx, by, bz,
                CellIndex::BIGGER
              }; 

              return res;
            }
          }
          else
          {
            return CellIndex{
              iOct_n,
              i_same, j_same, k_same,
              bx, by, bz,
              CellIndex::SAME_SIZE
            };
          }
        }
        else //constexpr if( std::is_same_v<SearchMode, SearchMode_neighbor>  )
        {
          DYABLO_ASSERT_KOKKOS_DEBUG( !lmesh.isBoundary(iOct, oct_offset_x, oct_offset_y, oct_offset_z), "Should not be boundary here" );

          LightOctree::OctantIndex iOct_neighbor = lmesh.findNeighbor(iOct, oct_offset_x, oct_offset_y, oct_offset_z);

          int level_diff = lmesh.getLevel(iOct) - lmesh.getLevel(iOct_neighbor);
          
          // Compute position of cell in neighbor when neighbor is same size;
          uint32_t i_same = i - oct_offset_x * bx;
          uint32_t j_same = j - oct_offset_y * by;
          uint32_t k_same = k - oct_offset_z * bz;

          if( level_diff == 0 )
          { // Neighbor is same size
            DYABLO_ASSERT_KOKKOS_DEBUG(i_same<bx, "internal error : i out of block bounds");
            DYABLO_ASSERT_KOKKOS_DEBUG(j_same<by, "internal error : j out of block bounds");
            DYABLO_ASSERT_KOKKOS_DEBUG(k_same<bz, "internal error : k out of block bounds");
            return CellIndex{
              iOct_neighbor,
              i_same, j_same, k_same,
              bx, by, bz,
              CellIndex::SAME_SIZE
            };
          }
          else if(level_diff == 1)
          { // Neighbor is larger  
            // Compute suboctant where target cell is located in larger neighbor
            auto coord_n = lmesh.get_logical_coords(iOct);

            auto is_odd = [](int x) {
              return (int)(x%2 != 0);
            };

            int suboctant_offset_x = is_odd( coord_n[IX] + oct_offset_x );
            int suboctant_offset_y = is_odd( coord_n[IY] + oct_offset_y );
            int suboctant_offset_z = is_odd( coord_n[IZ] + oct_offset_z );

            // Offset to select suboctant and /2 for position in larger octant 
            uint32_t i_larger = (i_same+suboctant_offset_x*bx)/2;
            uint32_t j_larger = (j_same+suboctant_offset_y*by)/2;
            uint32_t k_larger = (k_same+suboctant_offset_z*bz)/2;

            DYABLO_ASSERT_KOKKOS_DEBUG(i_larger<bx, "internal error : i out of block bounds");
            DYABLO_ASSERT_KOKKOS_DEBUG(j_larger<by, "internal error : j out of block bounds");
            DYABLO_ASSERT_KOKKOS_DEBUG(k_larger<bz, "internal error : k out of block bounds");

            CellIndex res{
              iOct_neighbor, 
              i_larger, j_larger, k_larger,
              bx, by, bz,
              CellIndex::BIGGER
            }; 

            return res;    
          }
          else if(level_diff == -1)
          { // Neighbor is smaller : compute CellIndex of "first neighbor" (neighbor cell closest to origin)

            // Compute cell position in neighbor meta-bloc of size {2*bx, 2*by, 2*bz}
            // Pick same-size cell origin for smaller_neighbor_mode() == ORIGIN
            uint32_t i_smaller = i*2 - oct_offset_x * bx * 2; 
            uint32_t j_smaller = j*2 - oct_offset_y * by * 2;
            uint32_t k_smaller = k*2 - oct_offset_z * bz * 2;

            if( search_mode.smaller_neighbor_mode() == SearchMode_neighbor::CLOSEST )
            {
              // Pick the smallest index {i_smaller, j_smaller, k_smaller} contiguous to current cell
              i_smaller += (int)(i<0);
              j_smaller += (int)(j<0);
              k_smaller += (int)(k<0);
            }

            DYABLO_ASSERT_KOKKOS_DEBUG(i_smaller<2*bx, "internal error : i out of block bounds");
            DYABLO_ASSERT_KOKKOS_DEBUG(j_smaller<2*by, "internal error : j out of block bounds");
            DYABLO_ASSERT_KOKKOS_DEBUG(k_smaller<2*bz, "internal error : k out of block bounds");

            // Compute position of suboctant containing "first neighbor" among the 8 suboctants
            int suboctant_x = i_smaller >= bx;    
            int suboctant_y = j_smaller >= by;    
            int suboctant_z = k_smaller >= bz;

            // Shift cell index to appropriate suboctant
            i_smaller -= bx * suboctant_x;
            j_smaller -= by * suboctant_y;
            k_smaller -= bz * suboctant_z;

            DYABLO_ASSERT_KOKKOS_DEBUG(i_smaller<bx, "internal error : i out of block bounds");
            DYABLO_ASSERT_KOKKOS_DEBUG(j_smaller<by, "internal error : j out of block bounds");
            DYABLO_ASSERT_KOKKOS_DEBUG(k_smaller<bz, "internal error : k out of block bounds");

            LightOctree::OctantIndex suboctant;
            
            // This is needed in case cells are scattered accross multiple suboctants
            // {
            //   [[maybe_unused]] bool found = false;

            //   LightOctree_tools::foreach_neighbor_octant( lmesh, iOct, iOct_neighbor, oct_offset,
            //     [&]( const LightOctree::OctantIndex& iOct_neighbor_i )
            //   {
            //     auto neighbor_suboct_coord = lmesh.get_logical_coords(iOct_neighbor_i);
            //     // Compute position of suboctant in bigger neighbor octant
            //     int this_suboctant_x = neighbor_suboct_coord[IX]%2;
            //     int this_suboctant_y = neighbor_suboct_coord[IY]%2;
            //     int this_suboctant_z = neighbor_suboct_coord[IZ]%2;

            //     // Match suboctant with suboctant containing first neighbor cell
            //     if( suboctant_x == this_suboctant_x && suboctant_y == this_suboctant_y && suboctant_z == this_suboctant_z )
            //     {
            //       suboctant = iOct_neighbor_i;
            //       found = true;
            //     }
            //   });

            //   DYABLO_ASSERT_KOKKOS_DEBUG( found, "smaller neighbor : corresponding suboctant not found" );
            // }
            #warning TODO optimize for even blocks : all subcells will be in same octant
            {
              DYABLO_ASSERT_KOKKOS_DEBUG( bx%2 == 0, "gathered subcells optimization enabled : block size must be even" );
              DYABLO_ASSERT_KOKKOS_DEBUG( by%2 == 0, "gathered subcells optimization enabled : block size must be even" );
              DYABLO_ASSERT_KOKKOS_DEBUG( bz%2 == 0, "gathered subcells optimization enabled : block size must be even" );
              suboctant = iOct_neighbor;
            }
            

            return CellIndex{
              suboctant, 
              (uint32_t)i_smaller, (uint32_t)j_smaller, (uint32_t)k_smaller,
              bx, by, bz,
              CellIndex::SMALLER};
          }
          else
          {
            DYABLO_ASSERT_KOKKOS_DEBUG(false, "Level-diff doesn't respect 2:1 balance");
            return CELLINDEX_INVALID;
          }
        }
      }
      else
      {
        static_assert( !std::is_same_v<SearchMode, SearchMode>, "Unsupported search mode" );
      }
    }
    else 
    {
      // Index is inside block
      // non-local cells keep their non-local status, but not their level difference
      CellIndex::Status cell_status = this->is_local() ?
                                        CellIndex::LOCAL_TO_BLOCK
                                      : CellIndex::SAME_SIZE;
      return CellIndex{this->iOct, (uint32_t)i, (uint32_t)j, (uint32_t)k, (uint32_t)bx, (uint32_t)by, (uint32_t)bz, cell_status};
    }
  }

  KOKKOS_INLINE_FUNCTION
  CellIndex operator+( const offset_t& offset ) const
  {
    return getNeighbor(offset, SearchMode_local(SearchMode_local::INVALID));
  }

  /**
   * Compute child cell index
   * 
   * @param lmesh LightOctree used to perform neighbor search
   * 
   * Current CellIndex needs to be inside an intermediate octant since leaves don't have children
   * //TODO Use a SearchMode to assert/return invalid CellIndex?
   * This will return the origin cell among the 4 or 8 subcells at 1 level below the current cell
   * The result can be an intermediate cell or a leaf
   * Search siblings using getNeighbor with offset in [0,1]^3
   **/
  KOKKOS_INLINE_FUNCTION
  CellIndex getChildren(const LightOctree& lmesh) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( this->is_valid(), "Index needs to be valid to get children");

    int8_t quadrant_x = (2*i) / bx; // = floor( i / (bx/2.0) )
    int8_t quadrant_y = (2*j) / by;
    int8_t quadrant_z = (2*k) / bz;

    LightOctree::OctantIndex iOct_c = lmesh.findChild(this->iOct, {quadrant_x, quadrant_y, quadrant_z});

    uint32_t i_c = (2*i) - bx*quadrant_x;
    uint32_t j_c = (2*j) - by*quadrant_y;
    uint32_t k_c = (2*k) - bz*quadrant_z;

    return CellIndex{iOct_c, i_c,j_c,k_c, bx,by,bz, CellIndex::SMALLER};
  }

  KOKKOS_INLINE_FUNCTION
  CellIndex getParent(const LightOctree& lmesh) const
  {
    const auto lc = lmesh.get_logical_coords(this->iOct);

    Kokkos::Array<uint32_t, 3> logical_coords;
    logical_coords[IX] = (lc[IX] >> 1);
    logical_coords[IY] = (lc[IY] >> 1);
    logical_coords[IZ] = (lc[IZ] >> 1);

    const int8_t quadrant_x = lc[IX] % 2;
    const int8_t quadrant_y = lc[IY] % 2;
    const int8_t quadrant_z = lc[IZ] % 2;

    const LightOctree::OctantIndex iOct_p = lmesh.findParent(this->iOct);

    const uint32_t i_p = ( i + bx * quadrant_x ) >> 1;
    const uint32_t j_p = ( j + by * quadrant_y ) >> 1;
    const uint32_t k_p = ( k + bz * quadrant_z ) >> 1;

    return CellIndex{iOct_p, i_p, j_p, k_p, bx,by,bz, CellIndex::BIGGER};
  }


  KOKKOS_INLINE_FUNCTION
  bool operator==(const CellIndex &c2) const {
  return (iOct.iOct == c2.iOct.iOct 
       && iOct.isGhost == c2.iOct.isGhost
       && iOct.isIntermediate == c2.iOct.isIntermediate
       && i == c2.i
       && j == c2.j
       && k == c2.k
       && status == c2.status);
  }
};
struct CellArray_shape
{
  uint32_t bx=0, by=0, bz=0;
  uint32_t nbFields=0;
  uint32_t nbOcts=0, nbGhosts=0, nbIntermediateOcts=0, nbIntermediateGhosts=0;

  /**
   * Convert cell index used for another array into an index compatible with current shape. 
   * What happens when *in* is outside of current block depends on search_mode
   * - SearchMode_local : does not look for neighbor octs
   * - SearchMode_neighbor : search cell in neighbor octants
   * (Read SearchMode_* doc for more detail on how to configure them)
   **/
  template<typename SearchMode>
  KOKKOS_INLINE_FUNCTION
  CellIndex convert_index(const CellIndex& in, const SearchMode& search_mode) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( in.is_valid(), "Index needs to be valid for conversion");

    if( in.bx == bx && in.by == by && in.bz == bz )
      return in;

    int32_t gx = ((int32_t)bx-(int32_t)in.bx)/2;
    int32_t gy = ((int32_t)by-(int32_t)in.by)/2;
    int32_t gz = ((int32_t)bz-(int32_t)in.bz)/2;
    int32_t i = in.i + gx;
    int32_t j = in.j + gy;
    int32_t k = in.k + gz;
    int32_t bx = this->bx;
    int32_t by = this->by;
    int32_t bz = this->bz;

    if( i<0 || i>=bx || j<0 || j>=by || k<0 || k>=bz )
    { 
      // Neighbor search
      // Closest position inside block
      uint32_t i_in = (uint32_t)i;
      uint32_t j_in = (uint32_t)j;
      uint32_t k_in = (uint32_t)k;
      // Offset from _in position
      int8_t di = 0;
      int8_t dj = 0;
      int8_t dk = 0;
      if(i<0)    { di=i   ; i_in=0;    }
      if(i>=bx)  { di=i-bx+1; i_in=bx-1; }
      if(j<0)    { dj=j   ; j_in=0;    }
      if(j>=by)  { dj=j-by+1; j_in=by-1; }
      if(k<0)    { dk=k   ; k_in=0;    }
      if(k>=bz)  { dk=k-bz+1; k_in=bz-1; }

      CellIndex iCell_in{in.iOct, i_in, j_in, k_in, (uint32_t)bx, (uint32_t)by, (uint32_t)bz};
      CellIndex::offset_t offset = {di,dj,dk};
      CellIndex iCell_out = iCell_in.getNeighbor( offset, search_mode );
      return iCell_out;
    }
    else
    {
      // Index is inside block
      // non-local cells keep their non-local status, but not their level difference
      CellIndex::Status cell_status =  in.is_local()?
                                        CellIndex::LOCAL_TO_BLOCK
                                      : CellIndex::SAME_SIZE;
      return CellIndex{in.iOct, (uint32_t)i, (uint32_t)j, (uint32_t)k, (uint32_t)bx, (uint32_t)by, (uint32_t)bz, cell_status};
    }
  }
};

/**
 * Abstraction for an array containing cell values
 * Cells are indexed by abstracted CellIndexes and 
 * values in cells are accessed through the at() accessor.
 * @tparam has_ghosts_ ghost cells are present
 * @tparam has_intermediate_ intermediate cells are present
 **/
template< bool has_ghosts_ >
class CellArray_base
{
public:
  using Shape_t = CellArray_shape;
  using View_t = Kokkos::View<real_t***, Kokkos::LayoutLeft>;
  static constexpr bool has_ghosts = has_ghosts_;

  View_t U, Ughost, Uintermediate;    
  Shape_t shape;

protected: 
  KOKKOS_INLINE_FUNCTION
  CellArray_base( const Shape_t& s)
    : shape(s)
  {}

public:

  KOKKOS_INLINE_FUNCTION
  CellArray_base() = default;
  
  
  /**
   * Construct a CellArray from a shape and allocate views
   * @param label to label the Kokkos views
   * @param shape size of the blocks and number of octants
   * Note: when has_ghosts is false, number of ghosts must be 0
   **/
  template< typename T >
  CellArray_base( const T& label, const Shape_t& s)
  : shape(s)
  {
    DYABLO_ASSERT_HOST_RELEASE( has_ghosts || shape.nbGhosts==0, "CellArray_base : ghosts disabled but nbGhosts>0"  );
    
    uint32_t nbCellsPerOct = shape.bx*shape.by*shape.bz;
    U = View_t(label, nbCellsPerOct, s.nbFields, s.nbOcts);
    if constexpr( has_ghosts )
    { 
      Ughost = View_t(label, nbCellsPerOct, s.nbFields, s.nbGhosts);
    }
  }
  
  KOKKOS_INLINE_FUNCTION
  int nbFields() const
  {
    return shape.nbFields;
  }

  // TODO : unify nbfields/nbFields
  KOKKOS_INLINE_FUNCTION
  int nbfields() const
  {
    return shape.nbFields;
  }

  KOKKOS_INLINE_FUNCTION
  Shape_t getShape() const
  {
    return shape;
  }

  KOKKOS_INLINE_FUNCTION
  operator Shape_t() const
  {
    return getShape();
  }

  KOKKOS_INLINE_FUNCTION
  real_t& at_ivar( const CellIndex& iCell, uint32_t ivar ) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG(ivar < shape.nbFields, "at_ivar : out of range");
    DYABLO_ASSERT_KOKKOS_DEBUG(shape.bx == iCell.bx, "bx mismatch icell vs array");
    DYABLO_ASSERT_KOKKOS_DEBUG(shape.by == iCell.by, "by mismatch icell vs array");
    DYABLO_ASSERT_KOKKOS_DEBUG(shape.bz == iCell.bz, "bz mismatch icell vs array");

    uint32_t i = iCell.i + iCell.j*iCell.bx + iCell.k*iCell.bx*iCell.by;
    if constexpr( has_ghosts )
    {
      if( iCell.iOct.isGhost )
        return Ughost(i, ivar, iCell.iOct.iOct);
    }
    DYABLO_ASSERT_KOKKOS_DEBUG( !iCell.iOct.isGhost, "Accessing ghost in non-ghosted array" );
    return U(i, ivar, iCell.iOct.iOct%shape.nbOcts);
  }

  KOKKOS_INLINE_FUNCTION
  real_t& at( const CellIndex& iCell, int ivar ) const
  {
    return at_ivar(iCell, ivar);
  }
};

using CellArray_global = CellArray_base<false>;
using CellArray_global_ghosted = CellArray_base<true>;

} // namespace AMRBlockForeachCell_CellArray_impl

} // namespace dyablo