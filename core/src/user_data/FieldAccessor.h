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

class UserData_FieldAccessor
{
friend GhostCommunicator_full_blocks;
friend UserData_Fields_Pdata;
public:
    static constexpr int MAX_FIELD_COUNT = id2index_t::MAX_INDEX_COUNT;
    using FieldInfo = UserData_FieldAccessor_FieldInfo;
    using FieldView_t = UserData::FieldView_t;

    UserData_FieldAccessor() = default;
    UserData_FieldAccessor(const UserData_FieldAccessor& ) = default;
    UserData_FieldAccessor(UserData_FieldAccessor& ) = default;
    UserData_FieldAccessor& operator=(const UserData_FieldAccessor& ) = default;
    UserData_FieldAccessor& operator=(UserData_FieldAccessor& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return ivar_to_arrayindex.size();
    }

    UserData_FieldAccessor(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info);

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ForeachCell::CellIndex& iCell, const VarIndex& varindex ) const
    {
        return fields.at_ivar( iCell, get_index_from_varindex(varindex) );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachCell::CellIndex& iCell, int ivar ) const
    {
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
};

} //namespace UserData_Impl
} //namespace dyablo