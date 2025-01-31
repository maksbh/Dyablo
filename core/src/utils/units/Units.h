#pragma once

namespace dyablo {

namespace Units{

/**
 * Represents a unit with it's dimensionnality as template parameters and 
 * it's value in arbitrary common units. 
 * 
 * @tparams Dims... dimensions for dimensionnal analysis (the names are probably explicit enough)
 * 
 * Notes:
 * - 'Units' represent physical units as well as quantities and may be used in 
 *   simple formulas for compile-time dimensional analysis 
 * - (Using SI as arbitrary common unit is only internal implementation details, 
 *    besides SI really is the only true real unit system)
 */
template<int Time=0, int Length=0, int Mass=0, int Current=0, int Temperature=0, int mol=0, int LuminousIntensity=0>
class Unit
{
public:
    real_t value_SI; // Value expressed in SI unit

public:
    /**
     * Construct a unit from it's value in SI units
     * /!\ please only create new units from multiplying/dividing existing units
     * This constructor is explicit to force correct dimensionnal analysis when using floats
     **/
    constexpr explicit Unit(real_t value_SI)
    : value_SI(value_SI)
    {}

    /**
     * Get the value of current 'Unit' (actually a quantity) expressed in unit `unit`
     * @tparam Dims... must be the same as current unit dims, this is only here for the error message
     * Note : this is the only way to convert from `Unit` to 
     *        `real_t` that ensures correct dimensional analysis
     **/
    template<int... Dims>
    constexpr real_t convert_to( const Unit<Dims...>& unit ) const
    {
        static_assert( std::is_same_v<Unit, Unit<Dims...>>, "Unit conversion error : dimension mismatch" );
        return value_SI / unit.value_SI;
    }

    /// Quantities of same dimensionnality can be added/substracted to help with dimensionnal analysis
    template<int... Dims>
    constexpr Unit operator+( const Unit<Dims...>& u2 ) const
    {
        static_assert( std::is_same_v<Unit, Unit<Dims...>>, "Unit addition error : dimension mismatch" );
        return Unit(value_SI + u2.value_SI);
    }

    template<int... Dims>
    constexpr Unit operator-( const Unit<Dims...> u2 ) const
    {
        static_assert( std::is_same_v<Unit, Unit<Dims...>>, "Unit substraction error : dimension mismatch" );
        return Unit(value_SI - u2.value_SI);
    }
};

/// Multiply Units to create derived units
template< int... Dims1, int... Dims2 >
constexpr Unit<Dims1+Dims2...> operator*(const Unit<Dims1...>& u1, const Unit<Dims2...>& u2 )
{
    return Unit<Dims1+Dims2...>( u1.value_SI * u2.value_SI );
}

/// Divide Units to create derived units
template< int... Dims1, int... Dims2 >
constexpr Unit<Dims1-Dims2...> operator/(const Unit<Dims1...>& u1, const Unit<Dims2...>& u2 )
{
    return Unit<Dims1-Dims2...>( u1.value_SI / u2.value_SI );
}

/**
 * Helper template function to perform controlled conversions from other types to units
 * Version for Units : Units are already units
 **/ 
template< int... Dims > 
constexpr Unit<Dims...> to_unit( const Unit<Dims...>& u )
{
    return u;
}
/**
 * Helper template function to perform controlled conversions from other types to units
 * Version for floats : Automatically convert floats to dimensionless Units (in operators only)
 **/ 
constexpr Unit<> to_unit( real_t val )
{
    return Unit<>(val);
}

template<typename T>
struct Unit_traits
{
    constexpr static bool is_unit = false;
};

template< int... Dims >
struct Unit_traits<Unit<Dims...>>
{
    constexpr static bool is_unit = true;
};

/// Is type a unit or not?
template<typename T>
constexpr bool is_unit = Unit_traits<T>::is_unit;

//----------------------------
// Operators for units with automatic type conversion using to_unit()
//----------------------------

// Don't overload operators when units are not involved
template<typename... Ts>
using enable_if_has_unit = std::enable_if_t<((is_unit<Ts> || ...)), int>;

/// Units can be multiplied with types compatible with to_unit() on both sides
template< typename T1, typename T2, enable_if_has_unit<T1, T2> = 0 >
constexpr auto operator*(const T1& u1, const T2& u2 )
{
    return to_unit( u1 ) * to_unit( u2 );
}

/// Units can be divided with types compatible with to_unit() on both sides
template< typename T1, typename T2, enable_if_has_unit<T1, T2> = 0  >
constexpr auto operator/(const T1& u1, const T2& u2 )
{
    return to_unit( u1 ) / to_unit( u2 );
}

/// Units can be added with types compatible with to_unit() on both sides
template< typename T1, typename T2, enable_if_has_unit<T1, T2> = 0  >
constexpr auto operator+(const T1& u1, const T2& u2 )
{
    return to_unit( u1 ).operator+(to_unit( u2 ));
}

/// Units can be substracted with types compatible with to_unit() on both sides
template< typename T1, typename T2, enable_if_has_unit<T1, T2> = 0  >
constexpr auto operator-(const T1& u1, const T2& u2 )
{
    return to_unit( u1 ).operator-(to_unit( u2 ));
}


// SI units = 1
constexpr Unit<1>              second   (1);
constexpr Unit<0,1>            meter    (1);
constexpr Unit<0,0,1>          kilogram (1);
constexpr Unit<0,0,0,1>        Ampere   (1);
constexpr Unit<0,0,0,0,1>      Kelvin   (1);
constexpr Unit<0,0,0,0,0,1>    mol      (1);
constexpr Unit<0,0,0,0,0,0,1>  candela  (1);

// Unit multiplicators
constexpr Unit<> one(1);
constexpr Unit<> Giga = 1e9 * one;
constexpr Unit<> Mega = 1e6 * one;
constexpr Unit<> Kilo = 1e3 * one;
constexpr Unit<> centi = 1e-2 * one;
constexpr Unit<> milli = 1e-3 * one;

// MKS units
constexpr auto kg = kilogram;
constexpr auto m = meter;
constexpr auto s = second;
constexpr auto km = Kilo*meter;
constexpr auto K = Kelvin;
constexpr auto Pa = kg / m / s / s;
constexpr auto cd = candela;
constexpr auto Joule = kg * m*m /s/s;
constexpr auto Newton = kg * m /s/s;
constexpr auto J = Joule;
constexpr auto N = Newton;

// CGS units
constexpr auto gram = milli * kilogram;
constexpr auto g = gram;
constexpr auto centimeter = centi * meter;
constexpr auto cm = centimeter;
constexpr auto cm_per_s = cm / s;
constexpr auto erg = g * cm*cm /s/s;

// Common astro units
constexpr auto parsec = 3.085677580962325e+16 * meter;
constexpr auto pc = parsec; 
constexpr auto kpc = Kilo * parsec;
constexpr auto Mpc = Mega * parsec;
constexpr auto Gpc = Giga * parsec;

constexpr auto astronomical_unit = 149597870750.76672 * meter;
constexpr auto au = astronomical_unit;
constexpr auto solar_mass = 1.98841586e+30 * kilogram;
constexpr auto Msun = solar_mass;

constexpr auto year = 31557600 * second;
constexpr auto yr = year;
constexpr auto kyr = Kilo * year;
constexpr auto Myr = Mega * year;
constexpr auto Gyr = Giga * year;

constexpr auto atomic_mass_unit = 1.660538921e-27 * kilogram;
constexpr auto amu = atomic_mass_unit;
constexpr auto electronvolt = 1.60217656e-19 * Joule;
constexpr auto eV = electronvolt;

// Convenient multiples
constexpr Unit m2 = m * m;
constexpr Unit m3 = m * m * m;
constexpr Unit cm2 = cm * cm;
constexpr Unit cm3 = cm * cm * cm;
constexpr Unit s2 = s * s;

// Constants
constexpr auto KBOLTZ = 1.3806e-23 * Joule / Kelvin;
constexpr auto PROTON_MASS = 1.67262158e-27 * kilogram;
constexpr auto MHE_OVER_MH = 4.002 * one;
constexpr auto HELIUM_MASS = MHE_OVER_MH * PROTON_MASS;
constexpr auto NEWTON_G = 6.67408e-11 * Newton * (meter*meter) / (kilogram*kilogram);
constexpr auto SOLAR_MASS = solar_mass;
constexpr auto SPEEDOFLIGHT = 299792458 * meter / second;
constexpr auto H0 = 70.3 * km / s / Mpc;
constexpr auto YHE = 0.24 * one; // Helium Mass fraction
constexpr auto yHE = (YHE/(1.-YHE)/MHE_OVER_MH); // Helium number fraction

}


} //namespace dyablo