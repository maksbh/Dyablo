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
    auto unit_time = configMap.getValue<real_t>("units", "time", 1.0) * Units::second();
    auto unit_length = configMap.getValue<real_t>("units", "length", 1.0) * Units::meter();
    real_t unit_mass_SI = 1;
    if( configMap.hasValue("units","density") )
    {
      real_t unit_density = configMap.getValue<real_t>("units", "density");
      unit_mass_SI = unit_density * unit_length.pow<3>().convert_to( Units::meter().pow<3>() );
    }
    auto unit_mass = configMap.getValue<real_t>("units", "mass", unit_mass_SI) * Units::kg();
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
