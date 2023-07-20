# Charting the Heavens

This repository explains how to make a map of the Western constellations (those approved by the IAU) using open-source data from [Stellarium](https://stellarium.org/) and the [HYG Database](http://www.astronexus.com/hyg) (Hipparcos, Yale Bright Star, and Gliese Catalogs). The only software used is `Python 3.9.7`. If you have comments or suggestions, please let me know! The required dependencies are: `astropy`, `numpy`, `matplotlib`, `astroquery`, `pandas`, and `adjustText`. 

![Andromeda](https://github.com/astronoray/constellation_figures/assets/83091595/a46c506d-f4c8-40a9-b211-3c931f2fe6c3)
![Gemini](https://github.com/astronoray/constellation_figures/assets/83091595/535c2bea-781a-4eab-bbff-e5d9d50cd80f)


## A Few Introductory Notes

This project started a long time ago, and I've been working on it off and on for several years. I was heavily inspired by several different people online, and I highly suggest you go check them out. I give all due credit to them for letting me mooch off their hard work. Most of the impetus came from two different people: [Eleanor Lutz's](https://github.com/eleanorlutz) [western_constellations_atlas_of_space](https://github.com/eleanorlutz/western_constellations_atlas_of_space) here on GitHub and [Péter Pápics' Leuven Star Atlas](https://papics.eu/2017/06/the-leuven-star-atlas-making-a-publication-quality-stellar-atlas-from-scratch/) (on an indefinite hiatus). They really inspired me to complete this project and made it attainable. I cannot thank them enough!

Additionally, these figures that I've produced were put into a sort of ["Observers's Guide"](Constellations.pdf) with information about the constellations' history and mythology, viewing times and latitudes, and data for a few of the main stars and deep space objects. Let me quickly say thank you to Ian Ridpath's [Star Tales](http://www.ianridpath.com/startales/startales3.html) from which most of the mythos in my guide has been taken from. He did a lot of work documenting the history and naming conventions of each of the 88 constellations, and we should all be indebted to him. 

## Gathering and Processing Data

#### Stars

To map these stars, I pulled together several databases. My goal was to have each constellation plotted individually, similar to how Sky & Telescope have done on the Wikipedia pages for each constellation. Therefore, I needed the stars that join the asterisms to be colored and labeled and all other stars to be simply plotted. For the former, I manually went through each constellation and found the stars that form the asterism, collecting their right ascension (RA), declination (Dec), magnitudes, colors, and a bunch of other information. I also made sure to include any special LaTeX commands I might need for the Greek Bayer designations. That datatable is `Constellation_Stars.csv`. 

The other stars, including the ones inside the constellation but not joining the asterism together, came from David Nash's [HYG Database version 3](http://www.astronexus.com/hyg), which combines all stars from the Hipparcos, Yale Bright Star, and Gliese catalogs. Again, this dataset contains a bunch of information for each star. Just as Eleanor Lutz, I restricted the original dataset in my `Python` file to only include visible stars (magnitude <= 6.5). 

These charts plots the size of each star based on its magnitude. Visible stars were selected to have magnitudes <= 6.5. Star colors are based on their B-V colors, hopefully reproducing their appearance on the sky to the naked eye. The B-V colors are mapped to hexadecimal colors thanks to the tireless work of [Mitchell Charity](http://www.vendian.org/mncharity/dir3/starcolor/details.html). 

#### Constellations & Asterisms

I downloaded the classical Western asterisms through Stellarium, an open-source planetarium software. In the [Stellarium Git repository](https://github.com/Stellarium/stellarium), these asterisms are found in the `skycultures/western/` folder as `constellationship.fab`. This data is provided as a list of star ID numbers that should be connected to form each of the lines in the asterisms. To plot the asterisms on a map, I matched each star ID number to the same entry in the HYG Database, and converted the ID number into an RA and Dec location values for each asterism vertex. 

The official IAU constellation boundaries are available from the Royal Observatory Edinburgh Catalogue of Constellation Boundary Data. I download this using `astroquery`, targeting the catalog labeled `VI/49/bound_20`. I also created a list of constellations that bound each constellation. This allows me to plot those constellations in dark gray along with their stars. Annoying exceptions were made for the constellation of Serpens, the only constellation split into two spatially distinct parts. 

#### Deep Sky Objects

Originally I had intended to include the approximate positions and sizes of some deep sky objects, hopefully at least one per constellation. I did gather the data using the lists available over at [The Sky Live](https://theskylive.com/) with data collected from [SIMBAD](http://simbad.cds.unistra.fr/simbad/) and [NED](http://ned.ipac.caltech.edu/). However, I didn't like how they appeared on the maps and they distracted from the constellations and stars. The code is still included in `constellations.py`, but it is commented out. 

#### Map Projection

I used a Lambert conformal projection to plot each constellation. Directions, angles, and shapes are maintained, but distances are not the further away from the equator you travel. Other map projections are available through `Basemap`. The wrapping of RAs at 0 hours and 24 hours means that a few constellations (Andromeda, Cassiopeia, Cepheus, Octans, Pegasus, Phoenix, Pisces, Sculptor, Tucana, and Ursa Minor) had to have exceptions to their plotting that handled that wrapping. 

## The Final Product

These constellations are plotted and compiled into an "Observer's Guide" (uncreatively named [`Constellations.pdf`](Constellations.pdf), with the constellations separated by season. For each constellation, it's name, genitive, and abbreviation are given as well as the constellation's chart. A brief history and mythology of the constellation is given, and some observing facts, such as latitudes visible and if there are any notable metoer showers. Then two checklists are reported: the first giving the stars of the asterism and the second giving some deep sky objects for the constellation. A checkbox is next to each entry in both tables so the keen observer can check off the objects as they observe them. Names (along with proper IPA pronunciations), Bayer IDs, RAs, Decs, magnitudes, color, distances, and stellar spectral type are given for the stars, while the name, ID, RA, Dec, magnitude, object type, angular size, and distance are given for the deep sky objects. Finally a few empty lines for notes are available as well. 


