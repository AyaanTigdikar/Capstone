# Lithium production - Data package

This data package contains the data that powers the chart ["Lithium production"](https://ourworldindata.org/explorers/minerals?Mineral=Lithium&Metric=Production&Type=Mine&Share+of+global=false&country=OWID_WRL~AUS~CHL~CHN~USA) on the Our World in Data website. It was downloaded on January 08, 2026.

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


## Lithium production
Based on mined, rather than [refined](#dod:refined-production), production. Measured in tonnes.
Last updated: December 15, 2025  
Next update: December 2026  
Date range: 2000–2024  
Unit: tonnes  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
USGS - Mineral Commodity Summaries (2025); USGS - Historical Statistics for Mineral and Material Commodities (2024) – with major processing by Our World in Data

#### Full citation
USGS - Mineral Commodity Summaries (2025); USGS - Historical Statistics for Mineral and Material Commodities (2024) – with major processing by Our World in Data. “Lithium production” [dataset]. United States Geological Survey, “Mineral Commodity Summaries”; United States Geological Survey, “Historical Statistics for Mineral and Material Commodities” [original data].
Source: USGS - Mineral Commodity Summaries (2025), USGS - Historical Statistics for Mineral and Material Commodities (2024) – with major processing by Our World In Data

### How is this data described by its producer - USGS - Mineral Commodity Summaries (2025), USGS - Historical Statistics for Mineral and Material Commodities (2024)?
Notes found in original USGS historical data:
- Note on United States production: Production data for lithium refers to lithium contained in material produced or shipped from mines and brine operations in the United States. Production data for 1940–54 include both gross tons of lithium minerals and compound production and Li2O content of these products. Li2O contains 46.46 percent lithium; this information was used to determine lithium content for those years. Because production data for 1940–54 included dilithium sodium phosphate, the average lithium content of domestic production varied from 2.50 percent to 5.20 percent for the period. Most lithium ores average about 2.00 percent and, and dilithium sodium phosphate contains about 10.5 percent lithium.
- Note on global production: World production data are in metric tons of gross product of lithium minerals and brine. Since 1967, lithium production was reported as ore and ore concentrate from mines and lithium carbonate from brine deposits. Since 2000, lithium production was reported as ore concentrate from mines and lithium carbonate from brine deposits, as lithium content, and as lithium carbonate equivalent. World production data for 1966–67 do not include data from Rhodesia (Zimbabwe) and some other African countries. Zimbabwe was by far the largest producer at the time. After 1954, world production does not include U.S. production. Data were not available for 1900–24.

Notes found in original USGS data:
- Production in 2020, excluding U.S. production.
- Production in 2021, excluding U.S. production.
- Production in 2022, excluding U.S. production. Estimated for Brazil, Canada, China, Portugal, Zimbabwe, and other countries.
- Production in 2023.
- Estimated production in 2024.
- Production in 2021 is also estimated.
- Also estimated in 2022.
- Estimated 2023.
- Excludes U.S. production.

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


    