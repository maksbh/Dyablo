#include "ParticleAccessor.h"

namespace dyablo {

namespace UserData_Impl{
namespace{

class ParticleContainer
{
public:
    ParticleContainer( const ParticleContainer& ) = default;
    ParticleContainer( ParticleContainer&& ) = default;
    ParticleContainer( const std::string& name, const ForeachParticle& foreach_particle, uint32_t num_particles )
      : name(name), foreach_particle(foreach_particle), particles(name, num_particles, FieldManager(0))
    {}
    int nbAttributes() const
    {
        return attribute_index.size();
    }
    void new_ParticleAttribute( const std::string& attribute_name )
    {
        int nb_new_attributes = 1;
        int needed_attr_count = nbAttributes() + nb_new_attributes;
        this->max_particle_count = std::max( this->max_particle_count, needed_attr_count );
        int allocated_attr_count = nbAttributes();
        if( needed_attr_count > allocated_attr_count )
        {
            ParticleData particles_new( particles, FieldManager(needed_attr_count) );
            if( allocated_attr_count != 0 )
            {
                Kokkos::deep_copy( 
                    Kokkos::subview( particles_new.particle_data, Kokkos::ALL(), std::pair(0,allocated_attr_count) ),
                    particles.particle_data
                );
            }
            this->particles = particles_new;
        }

        for( const std::string& name : {attribute_name} )
        {
            DYABLO_ASSERT_HOST_RELEASE( !has_ParticleAttribute(name), "new_ParticleAttribute() - attribute already exist : " << name );
        
            auto first_free = [&]() -> int
            {
                for(int i=0; i<particles.nbfields(); i++)
                {
                    bool free = true;
                    for( auto& p : attribute_index )
                    {
                        if(p.second == i)
                            free = false;
                    }
                    if( free ) return i;
                }
                DYABLO_ASSERT_HOST_RELEASE(false, "new_ParticleAttribute internal error : not enough fields allocated");
                return -1;
            };

            int index = first_free();
            attribute_index[name] = index;
            
            const auto& particles = this->particles;

            foreach_particle.foreach_particle( "zero_attribute", particles,
                KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
            {
                particles.at_ivar(iPart, index) = 0;
            });
        }
        
    }
    bool has_ParticleAttribute( const std::string& name ) const
    {
        return attribute_index.end() != attribute_index.find(name); 
    }
    std::set<std::string> getEnabledParticleAttributes() const
    {
        std::set<std::string> res;
        for( const auto& p : attribute_index )
        {
            res.insert( p.first );
        }
        return res;
    }
    ParticleArray getParticleArray() const
    {
        return particles;
    }
    UserData::ParticleAttribute_t getParticleAttribute( const std::string& attribute_name ) const
    {
      ParticleData res( particles, FieldManager(1) );

      int index = attribute_index.at(attribute_name);

      const auto& particles = this->particles;
      foreach_particle.foreach_particle( "copy_attr", particles,
          KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
      {
          res.at_ivar(iPart, 0) = particles.at_ivar( iPart, index );
      });

      return res;
    }
    void move_ParticleAttribute( const std::string& attr_dest, const std::string& attr_src )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleAttribute(attr_src), "move_ParticleAttribute() - source attribute doesn't exist : " << attr_src);

        attribute_index[ attr_dest ] = attribute_index.at(attr_src);
        attribute_index.erase( attr_src );
    }
    void delete_ParticleAttribute( const std::string& attribute_name)
    {
        attribute_index.erase( attribute_name );
    }
    void distributeParticles()
    {
        ViewCommunicator part_comm = foreach_particle.get_distribute_communicator( particles );
        uint32_t nbParticles_new = part_comm.getNumGhosts();

        ParticleData particles_new( this->name, nbParticles_new, FieldManager(particles.nbfields()) );

        part_comm.exchange_ghosts<0>( particles.particle_data, particles_new.particle_data );
        part_comm.exchange_ghosts<0>( particles.particle_position, particles_new.particle_position );

        this->particles = particles_new;
    }

//private:
    std::string name;
    ForeachParticle foreach_particle;
    ParticleData particles;
    std::map<std::string, int> attribute_index;
    int max_particle_count = 0;
};

} // namespace

struct UserData_Particles_Pdata
{
public: 
    UserData_Particles_Pdata( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_particle( foreach_cell.get_amr_mesh(), configMap )
    {}    

    bool has_ParticleArray( const std::string& array_name )
    {
      return particle_containers.count(array_name) == 1;
    }

    ParticleContainer& getParticleContainer( const std::string& array_name )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(array_name), "Particle array does not exist : " << array_name );
        return particle_containers.at(array_name);
    }

    const ParticleContainer& getParticleContainer( const std::string& array_name ) const
    {
      return const_cast<UserData_Particles_Pdata*>(this)->getParticleContainer(array_name);
    }

    std::set<std::string> getEnabledParticleArrays() const
    {
        std::set<std::string> res;
        for( const auto& p : particle_containers )
        {
            res.insert( p.first );
        }
        return res;
    } 

//private:
    ForeachParticle foreach_particle;
    std::map<std::string, ParticleContainer> particle_containers;
};

} //namespace UserData_Impl

namespace {

using Pdata = UserData_Impl::UserData_Particles_Pdata;
using ParticleContainer = UserData_Impl::ParticleContainer;

}

UserData::Particles::Particles(ConfigMap& configMap, ForeachCell& foreach_cell)
  : pdata( std::make_unique<Pdata>(configMap, foreach_cell) )
{/*empty*/}

UserData::Particles::~Particles()
{/*empty*/}

void UserData::new_ParticleArray( const std::string& name, uint32_t num_particles )
{
  auto& pdata = *(this->particles.pdata);

  DYABLO_ASSERT_HOST_RELEASE( !this->has_ParticleArray(name), "new_ParticleArray() - particle array already exists : " << name );
  pdata.particle_containers.emplace( name, ParticleContainer(name, pdata.foreach_particle, num_particles) );
}

void UserData::new_ParticleAttribute( const std::string& array_name, const std::string& attribute_name )
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  DYABLO_ASSERT_HOST_RELEASE( !array.has_ParticleAttribute(attribute_name), "UserData_particles::new_ParticleAttribute() - particle attribute already exists : " << attribute_name );
  array.new_ParticleAttribute( attribute_name );
}

bool UserData::has_ParticleArray(const std::string& name) const
{
  return this->particles.pdata->has_ParticleArray(name);
}

std::set<std::string> UserData::getEnabledParticleArrays() const
{
  return this->particles.pdata->getEnabledParticleArrays();
}

bool UserData::has_ParticleAttribute(const std::string& array_name, const std::string& attribute_name ) const
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.has_ParticleAttribute( attribute_name );
}

UserData::ParticleArray_t UserData::getParticleArray( const std::string& array_name ) const
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.getParticleArray();
}

std::set<std::string> UserData::getEnabledParticleAttributes( const std::string& array_name ) const
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.getEnabledParticleAttributes();
}

UserData::ParticleAttribute_t UserData::getParticleAttribute( const std::string& array_name, const std::string& attribute_name ) const
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.getParticleAttribute(attribute_name);
}
  
void UserData::move_ParticleAttribute( const std::string& array_name, const std::string& attr_dest, const std::string& attr_src )
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.move_ParticleAttribute(attr_dest, attr_src); 
}
  
void UserData::delete_ParticleAttribute(const std::string& array_name, const std::string& attribute_name)
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.delete_ParticleAttribute(attribute_name);
}

void UserData::distributeParticles( const std::string& array_name )
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.distributeParticles();
}
  
void UserData::distributeAllParticles()
{
  for(auto& pair : this->particles.pdata->particle_containers)
    pair.second.distributeParticles();
}

UserData::ParticleAccessor UserData::getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const
{
    return ParticleAccessor( *this->particles.pdata, array_name, attribute_info );
}

namespace UserData_Impl {

UserData_ParticleAccessor::UserData_ParticleAccessor(const UserData_Particles_Pdata& user_data, const std::string& array_name, const std::vector<UserData::ParticleAccessor_AttributeInfo>& attr_info)
 : particles(user_data.getParticleContainer( array_name ).particles)
{
  auto& particles = user_data.getParticleContainer( array_name );

  DYABLO_ASSERT_HOST_RELEASE( attr_info.size() > 0, "fields_info cannot be empty" );

  auto unknown_attr_error = [&](std::string attr_name)
  {
      std::stringstream s;
      s << "Could not find particle attribute '" << attr_name << "' in ParticleArray" << std::endl;
      s << "Available attributes are :" << std::endl;
      for( const std::string& particle_attr : particles.getEnabledParticleAttributes() )
      {
          s << " - '" << particle_attr << "'" << std::endl;
      }
      return s.str();
  };

  int max_varindex = 0;
  for( const AttributeInfo& info : attr_info )
  {
      max_varindex = std::max( max_varindex, info.id );
  }

  this->var_to_arrayindex = Kokkos::View<int*>( "varindex_to_viewindex", max_varindex+1 );
  this->ivar_to_arrayindex = Kokkos::View<int*>( "ivar_to_viewindex", attr_info.size() );
  auto var_to_arrayindex_host = Kokkos::create_mirror_view( this->var_to_arrayindex );
  auto ivar_to_arrayindex_host = Kokkos::create_mirror_view( this->ivar_to_arrayindex );
  for(size_t i=0; i<var_to_arrayindex_host.size(); i++)
      var_to_arrayindex_host(i) = -1;

  int i=0; 
  for( const AttributeInfo& info : attr_info )
  {
      DYABLO_ASSERT_HOST_RELEASE( particles.has_ParticleAttribute(info.name),
                                  unknown_attr_error(info.name));
      int index = particles.attribute_index.at(info.name);
      var_to_arrayindex_host(info.id) = index;
      ivar_to_arrayindex_host(i) = index;
      i++;
  }
  Kokkos::deep_copy( this->var_to_arrayindex, var_to_arrayindex_host );
  Kokkos::deep_copy( this->ivar_to_arrayindex, ivar_to_arrayindex_host );
}

} // namespace UserData_Impl
} // namespace dyablo