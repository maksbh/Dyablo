#######################
### MISC ##############
#######################
add_library(units "")

target_sources(units
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/Units.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CodeUnits.cpp
)

target_link_libraries( units PUBLIC Kokkos::kokkos )

target_include_directories(units
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils/units
  ${CMAKE_SOURCE_DIR}/src
  )

add_library(dyablo::units ALIAS units)
