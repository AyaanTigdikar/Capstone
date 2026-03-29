# Gold reserves - Data package

This data package contains the data that powers the chart ["Gold reserves"](https://ourworldindata.org/explorers/minerals?Mineral=Gold&Metric=Reserves&Type=Mine&Share+of+global=false&country=OWID_WRL~AUS~CHL~CHN~USA) on the Our World in Data website. It was downloaded on January 08, 2026.

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


## Gold reserves
Mineral [reserves](#dod:mineral-reserve) are resources that have been evaluated and can be mined economically with current technologies.
Last updated: December 15, 2025  
Next update: December 2026  
Date range: 2022–2024  
Unit: tonnes  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
USGS - Mineral Commodity Summaries (2025) – with major processing by Our World in Data

#### Full citation
USGS - Mineral Commodity Summaries (2025) – with major processing by Our World in Data. “Gold reserves” [dataset]. United States Geological Survey, “Mineral Commodity Summaries” [original data].
Source: USGS - Mineral Commodity Summaries (2025) – with major processing by Our World In Data

### How is this data described by its producer - USGS - Mineral Commodity Summaries (2025)?
Notes found in original USGS data:
- For Australia, Joint Ore Reserves Committee-compliant or equivalent reserves were 4,200 tons. Revised based on company and Government reports.
- Hat part of the reserve base that could be economically extracted or produced at the time of determination. The term “reserves” need not signify that extraction facilities are in place and operative. Reserves include only recoverable materials; thus, terms such as “extractable reserves” and “recoverable reserves” are redundant and are not a part of this classification system. Additional information is available in Appendix C of the Mineral Commodity Summaries, included in this data release, for resource and reserve definitions and information concerning data sources.
- For Australia, Joint Ore Reserves Committee-compliant or equivalent reserves were 4,600 tons. Reserves for Australia, China, Peru, Russia, and Tanzania were revised based on company and Government reports.
- For Australia, Joint Ore Reserves Committee-compliant or equivalent reserves were 4,600 tons.
- Reserves of the mineral commodity in the subject country.
- Reserves for Australia, China, Peru, Russia, and Tanzania were revised based on company and Government reports.

### Source

#### United States Geological Survey – Mineral Commodity Summaries
Retrieved on: 2025-12-15  
Retrieved from: https://doi.org/10.5066/P13XCP3R  

#### Notes on our processing step for this indicator
- The majority of the data is sourced from USGS, supplemented by BGS data where available. Where both overlap, USGS data is prioritized.
- As BGS does not provide global data, we calculated the world total by summing the data from individual countries, using this as a cross-check against USGS global figures.
- Due to the inherent uncertainties in the data for certain minerals and countries, we allowed a maximum deviation of 10% between the global totals reported by USGS and the calculated ones for BGS. If the deviation exceeded this threshold, we excluded the BGS data.
- The calculated global total from BGS data was used only on exceptional occasions, after ensuring that the resulting aggregate was sufficiently complete.
- Both BGS and USGS datasets include numerous notes and footnotes. We have retained most of these, making only minor edits or deletions where necessary to maintain clarity.


    