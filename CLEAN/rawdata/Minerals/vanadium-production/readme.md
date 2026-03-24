# Vanadium production - Data package

This data package contains the data that powers the chart ["Vanadium production"](https://ourworldindata.org/explorers/minerals?Mineral=Vanadium&Metric=Production&Type=Mine&Share+of+global=false&country=OWID_WRL~AUS~CHL~CHN~USA) on the Our World in Data website. It was downloaded on January 08, 2026.

### Active Filters

A filtered subset of the full data was downloaded. The following filters were applied:

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For normal countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The final column is the data column, which is the time series that powers the chart. If the CSV data is downloaded using the "full data" option, then the column corresponds to the time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data column is transformed depending on the chart type and thus the association with the time series might not be as straightforward.

## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

## Detailed information about the data


## Vanadium production
Production of [vanadium](#dod:vanadium), in tonnes. Based on mined, rather than [refined](#dod:refined-production), production.
Last updated: December 15, 2025  
Next update: December 2026  
Date range: 1901–2024  
Unit: tonnes  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
USGS - Mineral Commodity Summaries (2025); USGS - Historical Statistics for Mineral and Material Commodities (2024) – with major processing by Our World in Data

#### Full citation
USGS - Mineral Commodity Summaries (2025); USGS - Historical Statistics for Mineral and Material Commodities (2024) – with major processing by Our World in Data. “Vanadium production” [dataset]. United States Geological Survey, “Mineral Commodity Summaries”; United States Geological Survey, “Historical Statistics for Mineral and Material Commodities” [original data].
Source: USGS - Mineral Commodity Summaries (2025), USGS - Historical Statistics for Mineral and Material Commodities (2024) – with major processing by Our World In Data

### How is this data described by its producer - USGS - Mineral Commodity Summaries (2025), USGS - Historical Statistics for Mineral and Material Commodities (2024)?
Notes found in original USGS historical data:
- Note on global production: World production data are for mine production of vanadium. Data are from the MR and MYB for 1912-22, 1925, 1927-31, 1934-43, 1945-47, and 1998 to the most recent year, the CDS for 1960-77, and the MCS for 1978-84 and 1990-97. Data were not available for 1901-11, 1923-24, and 1948-59. World production was interpolated to two significant figures for 1926, 1932-33, 1944, and 1985-89. World production data for 1927-31 and 1997-99 do not contain U.S. production.

Notes found in original USGS data:
- Production in 2020.
- Production in 2021.
- Production in 2023.
- Estimated production in 2024.
- Estimated 2023.
- Also estimated in 2022.

Notes found in original BGS data:
- This table includes vanadium in slag products but excludes vanadium recovered as a byproduct of the refining and burning of heavy oils.
- This table includes vanadium in slag products but excludes vanadium recovered as a by-product of the refining and burning of heavy oils.
- Estimated V content of slag exported only.
- Tonnes of vanadium pentoxide.
- Figure includes six months product for 1975.
- Includes V content of slag.
- Recoverable vanadium.

### Sources

#### United States Geological Survey – Mineral Commodity Summaries
Retrieved on: 2025-12-15  
Retrieved from: https://doi.org/10.5066/P13XCP3R  

#### United States Geological Survey – Historical Statistics for Mineral and Material Commodities
Retrieved on: 2024-07-15  
Retrieved from: https://www.usgs.gov/centers/national-minerals-information-center/historical-statistics-mineral-and-material-commodities  

#### Notes on our processing step for this indicator
- The majority of the data is sourced from USGS, supplemented by BGS data where available. Where both overlap, USGS data is prioritized.
- As BGS does not provide global data, we calculated the world total by summing the data from individual countries, using this as a cross-check against USGS global figures.
- Due to the inherent uncertainties in the data for certain minerals and countries, we allowed a maximum deviation of 10% between the global totals reported by USGS and the calculated ones for BGS. If the deviation exceeded this threshold, we excluded the BGS data.
- The calculated global total from BGS data was used only on exceptional occasions, after ensuring that the resulting aggregate was sufficiently complete.
- Both BGS and USGS datasets include numerous notes and footnotes. We have retained most of these, making only minor edits or deletions where necessary to maintain clarity.


    