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

void FieldAccessor_init(const UserData_Fields_Pdata& user_data, const std::vector<UserData_FieldAccessor_FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        int& _nbFields,
                        int* var_to_arrayindex,
                        int* ivar_to_arrayindex,
                        int* var_to_arrayindex_intermediates,
                        int* ivar_to_arrayindex_intermediates,
                        UserData::FieldView_t& fields,
                        UserData::FieldView_t& fields_intermediates
                      );

template< bool has_intermediates, int _MAX_FIELD_COUNT >
class UserData_FieldAccessor_impl
{
friend GhostCommunicator_full_blocks;
friend UserData_Fields_Pdata;
public:
    static constexpr int MAX_FIELD_COUNT = _MAX_FIELD_COUNT;
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
        return _nbFields;
    }

    UserData_FieldAccessor_impl(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info)
    {
        UserData_Impl::FieldAccessor_init(
            user_data, fields_info,
            MAX_FIELD_COUNT, has_intermediates,
            this->_nbFields,
            this->var_to_arrayindex.data(), this->ivar_to_arrayindex.data(),
            this->var_to_arrayindex_intermediates.data(), this->ivar_to_arrayindex_intermediates.data(),
            this->fields, this->fields_intermediates
        );
    }

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
        auto iter_space = fields.getShape();
        auto iter_space_inter = fields_intermediates.getShape();
        iter_space.nbIntermediateOcts = iter_space_inter.nbOcts;
        iter_space.nbIntermediateGhosts = iter_space_inter.nbGhosts;
        return iter_space;
    }

private:
    int _nbFields;

    Kokkos::Array<int, MAX_FIELD_COUNT> var_to_arrayindex; // Index conversion for at()
    Kokkos::Array<int, MAX_FIELD_COUNT> ivar_to_arrayindex; // Index conversion for at_ivar()
    
    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return var_to_arrayindex[var];
    }

    Kokkos::Array<int, MAX_FIELD_COUNT> var_to_arrayindex_intermediates; // Index conversion for at()
    Kokkos::Array<int, MAX_FIELD_COUNT> ivar_to_arrayindex_intermediates; // Index conversion for at_ivar()

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex_intermediates(VarIndex var) const
    {
        return var_to_arrayindex_intermediates[var];
    }
protected:
    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return ivar_to_arrayindex[ivar];
    }
    int get_index_from_ivar_host(int ivar) const
    {
        return ivar_to_arrayindex[ivar];
    }
    FieldView_t fields;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device_intermediates(int ivar) const
    {
        return ivar_to_arrayindex_intermediates[ivar];
    }
    int get_index_from_ivar_host_intermediates(int ivar) const
    {
        return ivar_to_arrayindex_intermediates[ivar];
    }
    FieldView_t fields_intermediates;
};

} //namespace UserData_Impl
} //namespace dyablo