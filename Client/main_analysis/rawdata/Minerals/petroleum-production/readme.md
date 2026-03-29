# Petroleum production - Data package

This data package contains the data that powers the chart ["Petroleum production"](https://ourworldindata.org/explorers/minerals?Mineral=Petroleum&Metric=Reserves&Type=Mine&Share+of+global=false&country=OWID_WRL~AUS~CHL~CHN~USA) on the Our World in Data website. It was downloaded on January 08, 2026.

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


## Petroleum production
Last updated: December 15, 2025  
Next update: December 2026  
Date range: 1970–2023  
Unit: tonnes  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
BGS - World Mineral Statistics (2025) – with major processing by Our World in Data

#### Full citation
BGS - World Mineral Statistics (2025) – with major processing by Our World in Data. “Petroleum production” [dataset]. British Geological Survey, “World Mineral Statistics” [original data].
Source: BGS - World Mineral Statistics (2025) – with major processing by Our World In Data

### How is this data described by its producer - BGS - World Mineral Statistics (2025)?
Notes found in original BGS data:
- The figures shown in this table include natural gas liquids.
- The figures shown in this table exclude natural gasoline.
- The figures shown in this table include crude oil, shale oil, oil sands and natural gas liquids.
- Including oil from shale and coal.
- Including natural gas liquids.
- Entirely natural gas liquids.
- From oil shale.
- 1954 oil agreement area only.
- Including estimated production from occupied Sinai.
- Including shares of production from the Neutral Zone.
- Petroleum condensate.
- Sudan and South Sudan separated on 9 July 2011.
- In 1990 the Federal Republic of Germany (West Germany) was reunited with the German Democratic Republic (East Germany). Therefore, from 1991 these 2 countries are shown simply as "Germany".
- Yemen Arab Republic and People's Dem. Rep. Of Yemen were unified on 22 May 1990 as the Republic of Yemen.

### Source

#### British Geological Survey – World Mineral Statistics
Retrieved on: 2025-12-15  
Retrieved from: https://www.bgs.ac.uk/mineralsuk/statistics/world-mineral-statistics/  

#### Notes on our processing step for this indicator
- The majority of the data is sourced from USGS, supplemented by BGS data where available. Where both overlap, USGS data is prioritized.
- As BGS does not provide global data, we calculated the world total by summing the data from individual countries, using this as a cross-check against USGS global figures.
- Due to the inherent uncertainties in the data for certain minerals and countries, we allowed a maximum deviation of 10% between the global totals reported by USGS and the calculated ones for BGS. If the deviation exceeded this threshold, we excluded the BGS data.
- The calculated global total from BGS data was used only on exceptional occasions, after ensuring that the resulting aggregate was sufficiently complete.
- Both BGS and USGS datasets include numerous notes and footnotes. We have retained most of these, making only minor edits or deletions where necessary to maintain clarity.


    