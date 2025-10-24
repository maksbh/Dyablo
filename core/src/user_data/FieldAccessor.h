#pragma once

#include "user_data/UserData.h"

namespace dyablo {

class GhostCommunicator_full_blocks;

namespace UserData_Impl{

struct UserData_FieldAccessor_FieldInfo
{
  std::string name; /// Name as in VarIndex.h
  VarIndex id; /// id to use to access with at()
};

template< bool has_intermediates >
class UserData_FieldAccessor_impl
{
friend GhostCommunicator_full_blocks;
friend UserData_Fields_Pdata;
public:
    static constexpr int MAX_FIELD_COUNT = 20;
    using FieldInfo = UserData_FieldAccessor_FieldInfo;
    using FieldView_t = UserData::FieldView_t;

    UserData_FieldAccessor_impl() = default;
    UserData_FieldAccessor_impl(const UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl(UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl& operator=(const UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl& operator=(UserData_FieldAccessor_impl& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return ivar_to_arrayindex.size();
    }

    UserData_FieldAccessor_impl(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info);

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ForeachCell::CellIndex& iCell, const VarIndex& varindex ) const
    {
        if constexpr ( has_intermediates )
        {
            
            if( iCell.iOct.isIntermediate )
            {
                ForeachCell::CellIndex iCell_intermediate = iCell;
                iCell_intermediate.iOct.isIntermediate=false;
                return fields_intermediates.at_ivar( iCell_intermediate, get_index_from_varindex_intermediates(varindex) );
            }
        }
        DYABLO_ASSERT_KOKKOS_DEBUG( !iCell.iOct.isIntermediate, "Accessing intermediates in array without intermediates" );
        return fields.at_ivar( iCell, get_index_from_varindex(varindex) );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachCell::CellIndex& iCell, int ivar ) const
    {
        if constexpr ( has_intermediates )
        {
            if( iCell.iOct.isIntermediate )
                return fields_intermediates.at_ivar( iCell, get_index_from_ivar_device_intermediates(ivar) );
        }
        DYABLO_ASSERT_KOKKOS_DEBUG( !iCell.iOct.isIntermediate, "Accessing intermediates in array without intermediates" );
        return fields.at_ivar( iCell, get_index_from_ivar_device(ivar) );
    }

    KOKKOS_INLINE_FUNCTION
    FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG(nbFields() > 0, "Cannot getShape() of an empty UserData_fields" );
        return fields.getShape();
    }

private:
    Kokkos::View<int*> var_to_arrayindex; // Index conversion for at()
    Kokkos::View<int*> ivar_to_arrayindex; // Index conversion for at_ivar()
    Kokkos::View<int*>::host_mirror_type ivar_to_arrayindex_host;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return var_to_arrayindex(var);
    }

    Kokkos::View<int*> var_to_arrayindex_intermediates; // Index conversion for at()
    Kokkos::View<int*> ivar_to_arrayindex_intermediates; // Index conversion for at_ivar()
    Kokkos::View<int*>::host_mirror_type ivar_to_arrayindex_host_intermediates;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex_intermediates(VarIndex var) const
    {
        return var_to_arrayindex_intermediates(var);
    }
protected:
    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return ivar_to_arrayindex(ivar);
    }
    int get_index_from_ivar_host(int ivar) const
    {
        return ivar_to_arrayindex_host(ivar);
    }
    FieldView_t fields;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device_intermediates(int ivar) const
    {
        return ivar_to_arrayindex_intermediates(ivar);
    }
    int get_index_from_ivar_host_intermediates(int ivar) const
    {
        return ivar_to_arrayindex_host_intermediates(ivar);
    }
    FieldView_t fields_intermediates;
};

class UserData_FieldAccessor : public UserData_FieldAccessor_impl<false>{
public:
    using UserData_FieldAccessor_impl::UserData_FieldAccessor_impl;
};

class UserData_FieldAccessor_intermediates : public UserData_FieldAccessor_impl<true>{
public:
    using UserData_FieldAccessor_impl::UserData_FieldAccessor_impl;
};

} //namespace UserData_Impl
} //namespace dyablo