#include "FieldAccessor.h"

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

namespace dyablo {

namespace UserData_Impl{

struct UserData_Fields_Pdata
{
public:
    using FieldView_t = ForeachCell::CellArray_global_ghosted;

    UserData_Fields_Pdata( const UserData_Fields_Pdata& ) = default;
    UserData_Fields_Pdata( UserData_Fields_Pdata&& ) = default;

    UserData_Fields_Pdata( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_cell(foreach_cell)
    {}

    /***
     * @brief Return a CellArray_global_ghosted::Shape_t instance 
     * with the same size as all fields in current UserData_fields
     * UserData_fields must have at least on active field
     ***/
    const FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_HOST_RELEASE( field_index.size() > 0, "Cannot getShape() of an empty UserData_fields" );
        return fields.getShape();
    }

    void extend_fields( )
    {
        int allocated_field_count = fields.nbfields();
        auto fields_new = foreach_cell.allocate_ghosted_array( "UserData_fields", FieldManager(this->max_field_count) );
        if( allocated_field_count != 0 )
        {
            Kokkos::deep_copy( 
                Kokkos::subview(fields_new.U, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                fields.U
            );
            Kokkos::deep_copy( 
                Kokkos::subview(fields_new.Ughost, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                fields.Ughost
            );
        }
        fields = fields_new;
    }

    /**
     * Add new fields with unique identifiers 
     * names should not be already present
     **/
    void new_fields( const std::set<std::string>& names)
    {
        if( this->nbFields() != 0 )
        {
            DYABLO_ASSERT_HOST_RELEASE( fields.U.extent(2) == foreach_cell.get_amr_mesh().getNumOctants(), "UserData_fields internal error : mismatch between allocated size and octant count" );
            DYABLO_ASSERT_HOST_RELEASE( fields.Ughost.extent(2) == foreach_cell.get_amr_mesh().getNumGhosts(), "UserData_fields internal error : mismatch between allocated size and ghost octant count" );
        }
        
        int needed_field_count = nbFields() + names.size();
        this->max_field_count = std::max( this->max_field_count, needed_field_count );
        int allocated_field_count = fields.nbfields();
        if( needed_field_count > allocated_field_count )
        {   // Not enough fields : resize to add fields
            std::cout << "Reallocate : add fields " << allocated_field_count << " -> " << max_field_count << std::endl;
            extend_fields();
        }

        for( const std::string& name : names )
        {
            if( this->has_field(name) )
                throw std::runtime_error(std::string("UserData_fields::new_fields() - field already exists : ") + name);
            /// Find first free ivar in `fields` view
            auto first_free = [&]() -> int
            {
                for(int i=0; i<fields.nbfields(); i++)
                {
                    bool free = true;
                    for( auto& p : field_index )
                    {
                        if(p.second.index == i)
                            free = false;
                    }
                    if( free ) return i;
                }
                DYABLO_ASSERT_HOST_RELEASE(false, "UserData_fields internal error : not enough fields allocated");
                return -1;
            };
            
            int index = first_free();
            field_index[name].index = index;
            const auto& U = fields.U;
            Kokkos::parallel_for( "zero_new_field", U.extent(0)*U.extent(2),
                KOKKOS_LAMBDA( uint32_t i )
            {
                uint32_t iCell = i%U.extent(0);
                uint32_t iOct  = i/U.extent(0);
                U(iCell, index, iOct) = 0;
            });
            const auto& Ughost = fields.Ughost;
            Kokkos::parallel_for( "zero_new_field_ghost", Ughost.extent(0)*Ughost.extent(2),
                KOKKOS_LAMBDA( uint32_t i )
            {
                uint32_t iCell = i%Ughost.extent(0);
                uint32_t iOct  = i/Ughost.extent(0);
                Ughost(iCell, index, iOct) = 0;
            });
        }
    }

    /// Check if field exists
    bool has_field(const std::string& name) const
    {
        return field_index.end() != field_index.find(name);
    }

    std::set<std::string> getEnabledFields() const
    {
        std::set<std::string> res;
        for( const auto& p : field_index )
        {
            res.insert( p.first );
        }
        return res;
    }   

    // Get View associated with field name
    const FieldView_t getField(const std::string& name) const
    {
        if( !this->has_field(name)  )
            throw std::runtime_error(std::string("UserData_fields::getField() - field doesn't exist : ") + name);
        
        int index = field_index.at(name).index;

        auto field = foreach_cell.allocate_ghosted_array( std::string("field_")+name, FieldManager(1) );
        Kokkos::deep_copy( 
            field.U,
            Kokkos::subview(fields.U, Kokkos::ALL(), std::pair(index, index+1) , Kokkos::ALL() )
        );

        return field;
    }

    /// Change field name from src to dest. If dest already exist it is replaced
    void move_field( const std::string& dest, const std::string& src )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_field(src), "UserData_fields::move_field() - field doesn't exist : " << src);

        field_index[ dest ] = field_index.at( src );
        field_index.erase( src );
    }

    void delete_field( const std::string& name )
    {
        field_index.erase( name );
    }

    /// Get the number of active fields in UserData_fields
    int nbFields() const
    {
        return field_index.size();
    }

    void exchange_loadbalance( const ViewCommunicator& ghost_comm )
    {
      {  
        UserData::FieldAccessor fields_old = this->backup_and_realloc();
        int nb_fields = this->nbFields();
        auto old_fields_View = Kokkos::subview( fields_old.fields.U, 
                                                Kokkos::ALL(),
                                                std::make_pair(0, nb_fields),
                                                Kokkos::ALL()  );
        auto new_fields_View = Kokkos::subview(this->fields.U, 
                                                Kokkos::ALL(),
                                                std::make_pair(0, nb_fields),
                                                Kokkos::ALL()  );
        
        ghost_comm.exchange_ghosts<2>(old_fields_View, new_fields_View );
      }// This block is important to deallocate fields_old before extend_fields()

      this->extend_fields();
    }

    UserData::FieldAccessor getAccessor( const std::vector<UserData::FieldAccessor_FieldInfo>& fields_info ) const
    {
      return UserData::FieldAccessor(*this, fields_info);
    }

    UserData::FieldAccessor backup_and_realloc()
    {
      using FieldAccessor = UserData::FieldAccessor;
      std::vector<FieldAccessor::FieldInfo> all_fields;
      int i=0;
      for( const std::string& field : this->getEnabledFields() )
        all_fields.push_back({field, i++});
      FieldAccessor fields_old = this->getAccessor( all_fields );

      this->fields = foreach_cell.allocate_ghosted_array( "UserData_fields", FieldManager(this->field_index.size()) );
      // Reorder fields to reduce fragmentation
      std::map<std::string, field_index_t> field_index_new;
      {
        int new_index = 0;
        for( const auto& [field_name, old_index] : this->field_index )
        {
          field_index_new[field_name].index = new_index;
          new_index++;
        }
      }
      this->field_index = field_index_new;

      return fields_old;
    }

//private:
    ForeachCell& foreach_cell;
    FieldView_t fields;
    struct field_index_t
    {
        int index;
    };
    std::map<std::string, field_index_t> field_index;
    int max_field_count = 0;
};

} //namespace UserData_Impl

namespace {

using Pdata = UserData_Impl::UserData_Fields_Pdata;
using FieldView_t = UserData::FieldView_t;

}

UserData::Fields::Fields(ConfigMap& configMap, ForeachCell& foreach_cell)
  : pdata( std::make_unique<Pdata>(configMap, foreach_cell) )
{}

UserData::Fields::~Fields()
{}

const FieldView_t::Shape_t UserData::getShape() const
{
  return this->fields.pdata->getShape();
}

void UserData::new_fields( const std::set<std::string>& names)
{
  this->fields.pdata->new_fields(names);
}

bool UserData::has_field(const std::string& name) const
{
  return this->fields.pdata->has_field(name);
}

std::set<std::string> UserData::getEnabledFields() const
{
  return this->fields.pdata->getEnabledFields();
}

const FieldView_t UserData::getField(const std::string& name) const
{
  return this->fields.pdata->getField(name);
}

void UserData::move_field( const std::string& dest, const std::string& src )
{
  this->fields.pdata->move_field(dest, src);
}

void UserData::delete_field( const std::string& name )
{
  this->fields.pdata->delete_field(name);
}

void UserData::exchange_loadbalance( const ViewCommunicator& ghost_comm )
{
  this->fields.pdata->exchange_loadbalance(ghost_comm);
}

int UserData::nbFields() const
{
  return this->fields.pdata->nbFields();
}

UserData::FieldAccessor UserData::backup_and_realloc()
{
  return this->fields.pdata->backup_and_realloc();
}

void UserData::extend_fields()
{
  this->fields.pdata->extend_fields();
}

UserData::FieldAccessor UserData::getAccessor( const std::vector<UserData::FieldAccessor_FieldInfo>& fields_info ) const
{
  return this->fields.pdata->getAccessor(fields_info);
}

namespace UserData_Impl {

UserData_FieldAccessor::UserData_FieldAccessor(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info)
: fields(user_data.fields)
{
    DYABLO_ASSERT_HOST_RELEASE( fields_info.size() > 0, "fields_info cannot be empty" );

    auto unknown_field_error = [&](std::string field_name)
    {
        std::stringstream s;
        s << "Could not find field '" << field_name << "' in UserData" << std::endl;
        s << "Available fields are :" << std::endl;
        for( auto& p : user_data.field_index )
        {
            s << " - '" << p.first << "'" << std::endl;
        }
        return s.str();
    };

    int max_varindex = 0;
    for( const FieldInfo& info : fields_info )
    {
        max_varindex = std::max( max_varindex, info.id );
    }

    this->var_to_arrayindex = Kokkos::View<int*>( "varindex_to_viewindex", max_varindex+1 );
    this->ivar_to_arrayindex = Kokkos::View<int*>( "ivar_to_viewindex", fields_info.size() );
    auto var_to_arrayindex_host = Kokkos::create_mirror_view( this->var_to_arrayindex );
    this->ivar_to_arrayindex_host = Kokkos::create_mirror_view( this->ivar_to_arrayindex );
    for(size_t i=0; i<var_to_arrayindex_host.size(); i++)
        var_to_arrayindex_host(i) = -1;

    int i=0; 
    for( const FieldInfo& info : fields_info )
    {
        DYABLO_ASSERT_HOST_RELEASE( user_data.has_field(info.name), 
                                    unknown_field_error(info.name) );
        int index = user_data.field_index.at(info.name).index;
        var_to_arrayindex_host(info.id) = index;
        this->ivar_to_arrayindex_host(i) = index;
        i++;
    }
    Kokkos::deep_copy( this->var_to_arrayindex, var_to_arrayindex_host );
    Kokkos::deep_copy( this->ivar_to_arrayindex, this->ivar_to_arrayindex_host );
}

} // namespace UserData_Impl
} // namespace dyablo

