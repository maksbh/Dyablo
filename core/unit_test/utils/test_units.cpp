#include "gtest/gtest.h"

#include "utils/units/UnitParser.hpp"

TEST( test_units, convert )
{
    using namespace dyablo::Units;

    EXPECT_DOUBLE_EQ( 1.0, (1000 * meter()).convert_to(km()) );
}

TEST( test_units, formula )
{
    using namespace dyablo::Units;

    EXPECT_DOUBLE_EQ( 201 , (2 * meter() + 10 * milli() * meter()).convert_to(cm()) );
}

TEST( test_units, string_conversion )
{
    using namespace dyablo::Units;

    EXPECT_NO_THROW( parse_unit<decltype(km()/s())>("1 km/s") );
    EXPECT_NO_THROW( parse_unit<decltype(J())>("1 J") );
    EXPECT_NO_THROW( parse_unit<Unit<>>("10") );

    EXPECT_ANY_THROW( parse_unit<decltype(km()/s())>("1 km") );
    EXPECT_ANY_THROW( parse_unit<Unit<>>("1 qslfkh") );
    EXPECT_ANY_THROW( parse_unit<Unit<>>("banane") );
    EXPECT_ANY_THROW( parse_unit<Unit<1>>(" ") );
    EXPECT_ANY_THROW( parse_unit<Unit<1>>("10") );
}

TEST( test_units, string_conversion_coherent )
{
    using namespace dyablo::Units;

#define CHECK_COHERENT_UNIT( name ) {                           \
        using U_t = decltype(name());                           \
        U_t u(1);                                               \
        EXPECT_NO_THROW(u = parse_unit<U_t>("11.1 "#name));        \
        EXPECT_DOUBLE_EQ( 11.1, u.convert_to(name()) );            \
    }

    CHECK_COHERENT_UNIT(Mpc);
    CHECK_COHERENT_UNIT(second);
    CHECK_COHERENT_UNIT(meter);
    CHECK_COHERENT_UNIT(kilogram);
    CHECK_COHERENT_UNIT(Ampere);
    CHECK_COHERENT_UNIT(Kelvin);
    CHECK_COHERENT_UNIT(mol);
    CHECK_COHERENT_UNIT(candela);
    CHECK_COHERENT_UNIT(one);
    CHECK_COHERENT_UNIT(Giga);
    CHECK_COHERENT_UNIT(Mega);
    CHECK_COHERENT_UNIT(Kilo);
    CHECK_COHERENT_UNIT(centi);
    CHECK_COHERENT_UNIT(milli);
    CHECK_COHERENT_UNIT(kg);
    CHECK_COHERENT_UNIT(m);
    CHECK_COHERENT_UNIT(s);
    CHECK_COHERENT_UNIT(km);
    CHECK_COHERENT_UNIT(K);
    CHECK_COHERENT_UNIT(Pa);
    CHECK_COHERENT_UNIT(cd);
    CHECK_COHERENT_UNIT(Joule);
    CHECK_COHERENT_UNIT(Newton);
    CHECK_COHERENT_UNIT(J);
    CHECK_COHERENT_UNIT(N);
    CHECK_COHERENT_UNIT(gram);
    CHECK_COHERENT_UNIT(g);
    CHECK_COHERENT_UNIT(centimeter);
    CHECK_COHERENT_UNIT(cm);
    //CHECK_COHERENT_UNIT(cm_per_s);
    CHECK_COHERENT_UNIT(erg);
    CHECK_COHERENT_UNIT(parsec);
    CHECK_COHERENT_UNIT(pc);
    CHECK_COHERENT_UNIT(kpc);
    CHECK_COHERENT_UNIT(Mpc);
    CHECK_COHERENT_UNIT(Gpc);
    CHECK_COHERENT_UNIT(astronomical_unit);
    CHECK_COHERENT_UNIT(au);
    CHECK_COHERENT_UNIT(solar_mass);
    //CHECK_COHERENT_UNIT(Msun);
    CHECK_COHERENT_UNIT(year);
    CHECK_COHERENT_UNIT(yr);
    CHECK_COHERENT_UNIT(kyr);
    CHECK_COHERENT_UNIT(Myr);
    CHECK_COHERENT_UNIT(Gyr);
    CHECK_COHERENT_UNIT(Dalton);
    CHECK_COHERENT_UNIT(atomic_mass_unit);
    CHECK_COHERENT_UNIT(amu);
    CHECK_COHERENT_UNIT(electronvolt);
    CHECK_COHERENT_UNIT(eV);
    
}