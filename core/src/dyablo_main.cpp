#include <string>

#include "DyabloSession.hpp"
#include "utils/config/ConfigMap.h"
#include "DyabloTimeLoop.h"

int main(int argc, char *argv[])
{
  using namespace dyablo;
  DyabloSession mpi_session(argc, argv);

  if( argc < 2 )
  {
    std::cout << "Error : no input file" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  ./dyablo [--kokkos-***=*] input_file.ini" << std::endl;
    return EXIT_FAILURE;
  }

  /*
   * read parameter file and initialize a ConfigMap object
   */
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap = ConfigMap::broadcast_parameters(input_file);
  if( configMap.hasValue("units","time") )
  { // Set code units
    auto unit_time = configMap.getValue<Units::Time>("units", "time");
    auto unit_length = configMap.getValue<Units::Length>("units", "length");
    Units::Mass unit_mass = Units::kg();
    DYABLO_ASSERT_HOST_RELEASE( !(configMap.hasValue("units","mass") && configMap.hasValue("units","density")), "Parsing units in .ini : cannot set code density and mass et the same time" );
    if( configMap.hasValue("units","density") )
    {
      auto unit_density = configMap.getValue<Units::Density>("units", "density");
      unit_mass = unit_density * unit_length.pow<3>();
    }
    else
    {
      unit_mass = configMap.getValue<Units::Mass>("units", "mass");
    }
    
    Units::code_units_init( Units::UnitSystem(
        unit_time,
        unit_length,
        unit_mass
      ));
  }
  DyabloTimeLoop simulation( configMap );

  simulation.run();

  return EXIT_SUCCESS;

} // end main
