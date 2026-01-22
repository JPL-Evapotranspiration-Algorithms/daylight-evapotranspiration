# Scientific evapotranspiration upscaling module
# Author: [Your Name]
# References:
# - Henderson-Sellers, B. (1984). QJRMS, 110(466), 1186-1190.
# - Shuttleworth, W.J. (1993). Agricultural and Forest Meteorology, 61(1-2), 13-35.
# - Monteith, J.L. (1965). Symposia of the Society for Experimental Biology, 19, 205-234.
# - Verma, S.B., et al. (1982). Agricultural Meteorology, 26, 1-10.
# - Anderson, M.C., et al. (2007). Remote Sensing of Environment, 112(1), 213-229.

import datetime
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from dateutil import parser

import rasters as rt
from rasters import Raster, SpatialGeometry, wrap_geometry
from sun_angles import SHA_deg_from_DOY_lat, daylight_from_SHA, sunrise_from_SHA, calculate_daylight, calculate_sunrise
from solar_apparent_time import calculate_solar_hour_of_day, calculate_solar_day_of_year
from verma_net_radiation import daylight_Rn_integration_verma

# Latent heat of vaporization for water at 20°C in Joules per kilogram
# Value from Henderson-Sellers (1984)
LAMBDA_JKG_WATER_20C = 2450000.0


def celcius_to_kelvin(T_C: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    Convert temperature from Celsius to Kelvin.

    Parameters
    ----------
    T_C : Raster or np.ndarray
        Temperature in degrees Celsius.

    Returns
    -------
    Raster or np.ndarray
        Temperature in Kelvin.

    Notes
    -----
    Standard conversion: T_K = T_C + 273.15
    See IUPAC Green Book, 3rd Edition.
    """
    # Add 273.15 to Celsius to get Kelvin
    return T_C + 273.15


def lambda_Jkg_from_Ta_K(Ta_K: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    Calculate the latent heat of vaporization of water as a function of air temperature (Kelvin).

    Parameters
    ----------
    Ta_K : Raster or np.ndarray
        Air temperature in Kelvin.

    Returns
    -------
    Raster or np.ndarray
        Latent heat of vaporization in J/kg.

    Notes
    -----
    Formula from Henderson-Sellers (1984):
        lambda = (2.501 - 0.002361 * (Ta_K - 273.15)) * 1e6
    """
    # Empirical formula for lambda as a function of temperature
    return (2.501 - 0.002361 * (Ta_K - 273.15)) * 1e6


def lambda_Jkg_from_Ta_C(Ta_C: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    Calculate the latent heat of vaporization of water as a function of air temperature (Celsius).

    Parameters
    ----------
    Ta_C : Raster or np.ndarray
        Air temperature in Celsius.

    Returns
    -------
    Raster or np.ndarray
        Latent heat of vaporization in J/kg.

    Notes
    -----
    Converts Celsius to Kelvin, then applies lambda_Jkg_from_Ta_K.
    """
    Ta_K = celcius_to_kelvin(Ta_C)
    lambda_Jkg = lambda_Jkg_from_Ta_K(Ta_K)
    return lambda_Jkg


def calculate_evaporative_fraction(
        LE: Union[Raster, np.ndarray],
        Rn: Union[Raster, np.ndarray],
        G: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    Calculate the evaporative fraction (EF), the ratio of latent heat flux to available energy.

    Parameters
    ----------
    LE : Raster or np.ndarray
        Latent heat flux (W/m^2).
    Rn : Raster or np.ndarray
        Net radiation (W/m^2).
    G : Raster or np.ndarray
        Soil heat flux (W/m^2).

    Returns
    -------
    Raster or np.ndarray
        Evaporative fraction (unitless).

    Notes
    -----
    EF = LE / (Rn - G)
    If denominator is zero, returns zero to avoid division by zero.
    Shuttleworth (1993).
    """
    # EF is the fraction of available energy used for evaporation
    # Suppress divide-by-zero warning since we handle it explicitly with rt.where
    with np.errstate(divide='ignore', invalid='ignore'):
        return rt.where((LE == 0) | ((Rn - G) == 0), 0, LE / (Rn - G))


def daylight_ET_from_daylight_LE(
    LE_daylight_Wm2: Union[Raster, np.ndarray, float], 
    daylight_hours: Union[Raster, np.ndarray, float] = None,
    DOY: Union[Raster, np.ndarray, int] = None,
    lat: Union[Raster, np.ndarray, float] = None,
    datetime_UTC: datetime = None,
    geometry: SpatialGeometry = None,
    lambda_Jkg: Union[Raster, np.ndarray, float] = LAMBDA_JKG_WATER_20C
) -> Union[Raster, np.ndarray]:
    """
    Calculate daylight evapotranspiration (ET) from daylight latent heat flux (LE).

    Parameters
    ----------
    LE_daylight_Wm2 : Raster, np.ndarray, or float
        Daylight latent heat flux (W/m^2).
    daylight_hours : Raster, np.ndarray, or float, optional
        Length of daylight in hours. If None, calculated from date/location.
    DOY : int, optional
        Day of year.
    lat : float, optional
        Latitude.
    datetime_UTC : datetime, optional
        UTC datetime.
    geometry : SpatialGeometry, optional
        Geometry object with spatial info.
    lambda_Jkg : float, optional
        Latent heat of vaporization (J/kg). Default is for 20°C.

    Returns
    -------
    Raster or np.ndarray
        Daylight evapotranspiration in kilograms (kg).

    Notes
    -----
    ET = (LE_daylight * daylight_seconds) / lambda
    If input is a dict, supports upscaling using evaporative fraction (EF) logic.
    Monteith (1965), Shuttleworth (1993).
    """
    # Calculate daylight hours if not provided
    if daylight_hours is None:
        daylight_hours = calculate_daylight(DOY=DOY, lat=lat, datetime_UTC=datetime_UTC, geometry=geometry)

    # Convert daylight hours to seconds
    daylight_seconds = daylight_hours * 3600.0

    # If input is dict, use EF logic for upscaling
    if isinstance(LE_daylight_Wm2, dict):
        # Support for dict input (e.g., PTJPLSM model output)
        LE_daylight_Wm2 = LE_daylight_Wm2.get('LE_daylight_Wm2', None)
        Rn_daylight_Wm2 = LE_daylight_Wm2.get('Rn_daylight_Wm2', None)
        Rn_Wm2 = LE_daylight_Wm2.get('Rn_Wm2', None)
        G_Wm2 = LE_daylight_Wm2.get('G_Wm2', None)
        LE_Wm2 = LE_daylight_Wm2.get('LE_Wm2', None)

        if None not in (LE_Wm2, Rn_Wm2, G_Wm2, Rn_daylight_Wm2):
            # Calculate EF and upscale
            EF = rt.where((LE_Wm2 == 0) | ((Rn_Wm2 - G_Wm2) == 0), 0, LE_Wm2 / (Rn_Wm2 - G_Wm2))
            LE_daylight_Wm2 = EF * Rn_daylight_Wm2
            ET_daylight_kg = rt.clip(LE_daylight_Wm2 * daylight_seconds / LAMBDA_JKG_WATER_20C, 0.0, None)
            return ET_daylight_kg
        elif LE_daylight_Wm2 is not None:
            ET_daylight_kg = rt.clip(LE_daylight_Wm2 * daylight_seconds / LAMBDA_JKG_WATER_20C, 0.0, None)
            return ET_daylight_kg

    # Standard calculation
    ET_daylight_kg = rt.clip(LE_daylight_Wm2 * daylight_seconds / LAMBDA_JKG_WATER_20C, 0.0, None)
    return ET_daylight_kg


def daylight_ET_from_instantaneous_LE(
        LE_instantaneous_Wm2: Union[Raster, np.ndarray, float],
        Rn_instantaneous_Wm2: Union[Raster, np.ndarray, float],
        G_instantaneous_Wm2: Union[Raster, np.ndarray, float],
        day_of_year: Union[Raster, np.ndarray, int] = None,
        lat: Union[Raster, np.ndarray, float] = None,
        hour_of_day: Union[Raster, np.ndarray, float] = None,
        sunrise_hour: Union[Raster, np.ndarray, float] = None,
        daylight_hours: Union[Raster, np.ndarray, float] = None,
        time_UTC: Union[datetime, str, np.ndarray, list] = None,
        geometry: SpatialGeometry = None,
        lambda_Jkg: Union[Raster, np.ndarray, float] = LAMBDA_JKG_WATER_20C
    ) -> Union[Raster, np.ndarray]:
    """
    Upscale instantaneous latent heat flux (LE) to daylight evapotranspiration (ET).

    Parameters
    ----------
    LE_instantaneous_Wm2 : Raster, np.ndarray, or float
        Instantaneous latent heat flux (W/m^2).
    Rn_instantaneous_Wm2 : Raster, np.ndarray, or float
        Instantaneous net radiation (W/m^2).
    G_instantaneous_Wm2 : Raster, np.ndarray, or float
        Instantaneous soil heat flux (W/m^2).
    day_of_year : int, optional
        Day of year.
    lat : float, optional
        Latitude.
    hour_of_day : float, optional
        Hour of day (decimal hours).
    sunrise_hour : float, optional
        Sunrise hour (decimal hours).
    daylight_hours : float, optional
        Daylight duration in hours.
    time_UTC : datetime, str, np.ndarray, or list, optional
        UTC time.
    geometry : SpatialGeometry, optional
        Geometry object with spatial info.
    lambda_Jkg : float, optional
        Latent heat of vaporization (J/kg). Default is for 20°C.

    Returns
    -------
    Raster or np.ndarray
        Daylight evapotranspiration in kilograms (kg).

    Notes
    -----
    1. Calculates evaporative fraction (EF) from instantaneous fluxes.
    2. Integrates net radiation over daylight using Verma et al. (1982) method.
    3. Computes daylight LE and converts to ET.
    Common in remote sensing upscaling (Anderson et al., 2007).
    """
    geometry = wrap_geometry(geometry)

    # Use geometry latitude if lat not provided
    if lat is None and geometry is not None:
        lat = geometry.lat
    
    if geometry is not None:
        lon = geometry.lon
    else:
        lon = None
    
    # Calculate evaporative fraction (EF)
    EF = calculate_evaporative_fraction(
        LE=LE_instantaneous_Wm2,
        Rn=Rn_instantaneous_Wm2,
        G=G_instantaneous_Wm2
    )

    # Calculate daylight hours if not provided
    if daylight_hours is None:
        daylight_hours = calculate_daylight(
            day_of_year=day_of_year,
            lat=lat,
            time_UTC=time_UTC,
            geometry=geometry
        )

    # Calculate sunrise hour if not provided
    if sunrise_hour is None:
        sunrise_hour = calculate_sunrise(
            day_of_year=day_of_year,
            lat=lat,
            time_UTC=time_UTC,
            geometry=geometry
        )

    if hour_of_day is None:
        hour_of_day = calculate_solar_hour_of_day(
            time_UTC=time_UTC,
            geometry=geometry,
            lat=lat,
            lon=lon
        )
    
    if day_of_year is None:
        day_of_year = calculate_solar_day_of_year(
            time_UTC=time_UTC,
            geometry=geometry,
            lat=lat,
            lon=lon
        )


    # Integrate net radiation over daylight period
    Rn_daylight_Wm2 = daylight_Rn_integration_verma(
        Rn_Wm2=Rn_instantaneous_Wm2,
        hour_of_day=hour_of_day,
        day_of_year=day_of_year,
        lat=lat,
        sunrise_hour=sunrise_hour,
        daylight_hours=daylight_hours
    )

    # Calculate daylight latent heat flux
    LE_daylight_Wm2 = EF * Rn_daylight_Wm2

    # Convert daylight LE to ET
    daylight_seconds = daylight_hours * 3600.0
    ET_daylight_kg = rt.clip(LE_daylight_Wm2 * daylight_seconds / LAMBDA_JKG_WATER_20C, 0.0, None)

    # Return a dict with Rn, LE, and ET outputs
    return {
        "Rn_daylight_Wm2": Rn_daylight_Wm2,
        "LE_daylight_Wm2": LE_daylight_Wm2,
        "ET_daylight_kg": ET_daylight_kg
    }
