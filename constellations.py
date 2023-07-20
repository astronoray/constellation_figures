#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 23:01:02 2021

@author: RayG
"""

import os
os.chdir('/Volumes/RayPass/Constellations')

from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
from astroquery.vizier import Vizier
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PatchCollection
import pandas as pd
from mpl_toolkits.basemap import pyproj
from adjustText import adjust_text

def lon2ra_fmt(deg):
    deg = np.floor(deg)
    ra = 0
    if deg > 0:
        ra = deg/15
    elif deg < 0:
        ra = (deg + 360)/15
    else:
        ra = 0
    return (r'%d$^{\mathrm{h}}$') % (ra)

def lat2dec_fmt(deg):
    deg = np.floor(deg)
    dec = 0
    if deg > 0:
        dec = deg
        return (r'$+$%d$\degree$') % (dec)
    elif deg < 0:
        dec = np.abs(deg)
        return (r'$-$%d$\degree$') % (dec)
    else:
        dec = deg
        return (r'%d$\degree$') % (dec)

def lon2ra(deg):
    ra = 0
    if deg > 0:
        ra = deg/15
    elif deg < 0:
        ra = (deg + 360)/15
    else:
        ra = 0
    return ra

# colors from
# http://www.vendian.org/mncharity/dir3/starcolor/details.html
def star_colors(bmv):
    colors = np.zeros(0)
    for i in bmv:    
        if i == None:
            colors = np.append(colors, '#000000')
        elif i <= -.4:
            colors = np.append(colors, '#9bb2ff')
        elif i > -.4 and i <= -.35:
            colors = np.append(colors, '#9eb5ff')
        elif i > -.35 and i <= -.3:
            colors = np.append(colors, '#a3b9ff')
        elif i > -.3 and i <= -.25:
            colors = np.append(colors, '#aabfff')
        elif i > -.25 and i <= -.2:
            colors = np.append(colors, '#b2c5ff')
        elif i > -.2 and i <= -.15:
            colors = np.append(colors, '#bbccff')
        elif i > -.15 and i <= -.1:
            colors = np.append(colors, '#c4d2ff')
        elif i > -.1 and i <= -.05:
            colors = np.append(colors, '#ccd8ff')
        elif i > -.05 and i <= 0:
            colors = np.append(colors, '#d3ddff')
        elif i > 0 and i <= 0.05:
            colors = np.append(colors, '#dae2ff')
        elif i > 0.05 and i <= 0.1:
            colors = np.append(colors, '#dfe5ff')
        elif i > 0.1 and i <= 0.15:
            colors = np.append(colors, '#e4e9ff')
        elif i > 0.15 and i <= 0.2:
            colors = np.append(colors, '#e9ecff')
        elif i > 0.2 and i <= 0.25:
            colors = np.append(colors, '#eeefff')
        elif i > 0.25 and i <= 0.3:
            colors = np.append(colors, '#f3f2ff')
        elif i > 0.3 and i <= .35:
            colors = np.append(colors, '#f8f6ff')
        elif i > 0.35 and i <= 0.4:
            colors = np.append(colors, '#fef9ff')
        elif i > 0.4 and i <= 0.45:
            colors = np.append(colors, '#fff9fb')
        elif i > 0.45 and i <= 0.5:
            colors = np.append(colors, '#fff7f5')
        elif i > 0.5 and i <= 0.55:
            colors = np.append(colors, '#fff5ef')
        elif i > 0.55 and i <= 0.6:
            colors = np.append(colors, '#fff3ea')
        elif i > 0.6 and i <= 0.65:
            colors = np.append(colors, '#fff1e5')
        elif i > 0.65 and i <= 0.7:
            colors = np.append(colors, '#ffefe0')
        elif i > 0.7 and i <= 0.75:
            colors = np.append(colors, '#ffeddb')
        elif i > 0.75 and i <= 0.8:
            colors = np.append(colors, '#ffedb6')
        elif i > 0.8 and i <= 0.85:
            colors = np.append(colors, '#ffe9d2')
        elif i > 0.85 and i <= 0.9:
            colors = np.append(colors, '#ffe8ce')
        elif i > 0.9 and i <= 0.95:
            colors = np.append(colors, '#ffe6ca')
        elif i > 0.95 and i <= 1:
            colors = np.append(colors, '#ffe5c6')
        elif i > 1.0 and i <= 1.05:
            colors = np.append(colors, '#ffe3c3')
        elif i > 1.05 and i <= 1.1:
            colors = np.append(colors, '#ffe2bf')
        elif i > 1.1 and i <= 1.15:
            colors = np.append(colors, '#ffe0bb')
        elif i > 1.15 and i <= 1.2:
            colors = np.append(colors, '#ffdfb8')
        elif i > 1.2 and i <= 1.25:
            colors = np.append(colors, '#ffddb4')
        elif i > 1.25 and i <= 1.3:
            colors = np.append(colors, '#ffdbb0')
        elif i > 1.3 and i <= 1.35:
            colors = np.append(colors, '#ffdaad')
        elif i > 1.35 and i <= 1.4:
            colors = np.append(colors, '#ffd8a9')
        elif i > 1.4 and i <= 1.45:
            colors = np.append(colors, '#ffd6a5')
        elif i > 1.45 and i <= 1.5:
            colors = np.append(colors, '#ffd5a1')
        elif i > 1.5 and i <= 1.55:
            colors = np.append(colors, '#ffd29c')
        elif i > 1.55 and i <= 1.6:
            colors = np.append(colors, '#ffd096')
        elif i > 1.6 and i <= 1.65:
            colors = np.append(colors, '#ffcc8f')
        elif i > 1.65 and i <= 1.7:
            colors = np.append(colors, '#ffc885')
        elif i > 1.7 and i <= 1.75:
            colors = np.append(colors, '#ffc178')
        elif i > 1.75 and i <= 1.8:
            colors = np.append(colors, '#ffb765')
        elif i > 1.8 and i <= 1.85:
            colors = np.append(colors, '#ffa94b')
        elif i > 1.85 and i <= 1.9:
            colors = np.append(colors, '#ff9523')
        elif i > 1.9 and i <= 1.95:
            colors = np.append(colors, '#ff7b00')
        elif i > 1.95 and i <= 2:
            colors = np.append(colors, '#ff5200')
        else:
            colors = np.append(colors, '#ff5200')
    return colors

def star_sizes(mag):
    sizes = np.zeros(0)
    for i in mag:
        if np.isnan(i) == True:
            sizes = np.append(sizes,1.875*(6.5-6.4))
        else:
            sizes = np.append(sizes,1.875*(6.5-i))
    return sizes

class Basemap(Basemap):
    def ellipse(self, x0, y0, a, b, n, ax=None, **kwargs):
        """
        Draws a polygon centered at ``x0, y0``. The polygon approximates an
        ellipse on the surface of the Earth with semi-major-axis ``a`` and 
        semi-minor axis ``b`` degrees longitude and latitude, made up of 
        ``n`` vertices.
    
        For a description of the properties of ellipsis, please refer to [1].
    
        The polygon is based upon code written do plot Tissot's indicatrix
        found on the matplotlib mailing list at [2].
    
        Extra keyword ``ax`` can be used to override the default axis instance.
    
        Other \**kwargs passed on to matplotlib.patches.Polygon
    
        RETURNS
            poly : a maptplotlib.patches.Polygon object.
    
        REFERENCES
            [1] : http://en.wikipedia.org/wiki/Ellipse
    
    
        """
        ax = kwargs.pop('ax', None) or self._check_ax()
        g = pyproj.Geod(a=self.rmajor, b=self.rminor)
        # Gets forward and back azimuths, plus distances between initial
        # points (x0, y0)
        azf, azb, dist = g.inv([x0, x0], [y0, y0], [x0+a, x0], [y0, y0+b])
        tsid = dist[0] * dist[1] # a * b
    
        # Initializes list of segments, calculates \del azimuth, and goes on 
        # for every vertex
        seg = [self(x0+a, y0)]
        AZ = np.linspace(azf[0], 360. + azf[0], n)
        for i, az in enumerate(AZ):
            # Skips segments along equator (Geod can't handle equatorial arcs).
            if np.allclose(0., y0) and (np.allclose(90., az) or
                np.allclose(270., az)):
                continue
    
            # In polar coordinates, with the origin at the center of the 
            # ellipse and with the angular coordinate ``az`` measured from the
            # major axis, the ellipse's equation  is [1]:
            #
            #                           a * b
            # r(az) = ------------------------------------------
            #         ((b * cos(az))**2 + (a * sin(az))**2)**0.5
            #
            # Azymuth angle in radial coordinates and corrected for reference
            # angle.
            azr = 2. * np.pi / 360. * (az + 90.)
            A = dist[0] * np.sin(azr)
            B = dist[1] * np.cos(azr)
            r = tsid / (B**2. + A**2.)**0.5
            lon, lat, azb = g.fwd(x0, y0, az, r)
            x, y = self(lon, lat)
    
            # Add segment if it is in the map projection region.
            if x < 1e20 and y < 1e20:
                seg.append((x, y))
    
        poly = Polygon(seg, **kwargs)
        ax.add_patch(poly)
    
        # Set axes limits to fit map region.
        self.set_axes_limits(ax=ax)
    
        return poly


#%%

data = QTable.read('Constellation_Stars.csv')
data.rename_column('\ufeffI','Technical Name')
data.rename_column('RA [deg]','RA')
data['RA'].unit = u.deg
data.rename_column('Dec [deg]','Dec')
data['Dec'].unit = u.deg
data.rename_column('U App Mag','U')
data.rename_column('B App Mag','B')
data.rename_column('V App Mag','V')
data.rename_column('J App Mag','J')
data.rename_column('H App Mag','H')
data.rename_column('K App Mag','K')
data.rename_column('Prop M. RA [mas/yr]','pmra')
data['pmra'].unit = u.marcsec/u.year
data.rename_column('Prop M. Dec [mas/yr]','pmdec')
data['pmdec'].unit = u.marcsec/u.year
data.rename_column('Radial Velocity [km/s]','rv')
data['rv'].unit = u.km/u.s
data.rename_column('RV Err [km/s]','rv_err')
data['rv_err'].unit = u.km/u.s
data.rename_column('Dist [pc]','d')
data['d'].unit = u.pc
data.rename_column('Dist Err [pc]','d_err')
data['d_err'].unit = u.pc
data.rename_column('Mass [solar]','mass')
data['mass'].unit = u.solMass
data.rename_column('Mass Err [solar]','mass_err')
data['mass_err'].unit = u.solMass
data.rename_column('Rad [solar]','rad')
data['rad'].unit = u.solRad
data.rename_column('Rad Err [solar]','rad_err')
data['rad_err'].unit = u.solRad
data.rename_column('Lum [solar]','lum')
data['lum'].unit = u.solLum
data.rename_column('Lum Err [solar]','lum_err')
data['lum_err'].unit = u.solLum
data.rename_column('Temp [K]','temp')
data['temp'].unit = u.K
data.rename_column('Temp Err [K]','temp_err')
data['temp_err'].unit = u.K

data_coords = data[np.logical_or(data['RA'] != 0, data['Dec'] != 0)]

#%%

dso = QTable.read('DSOs.csv')
dso['RA'].unit = u.deg
dso['Dec'].unit = u.deg
dso['R1'].unit = u.arcmin
dso['R2'].unit = u.arcmin
dso.rename_column('PM RA','pmra')
dso.rename_column('PM Dec','pmdec')
dso['pmra'].unit = u.marcsec/u.year
dso['pmdec'].unit = u.marcsec/u.year
dso.rename_column('RV Err','rv_err')
dso.rename_column('RV','rv')
dso['rv'].unit = u.km/u.s
dso['rv_err'].unit = u.km/u.s
dso.rename_column('Dist','dist')
dso.rename_column('Dist Err','dist_err')
dso['dist'].unit = u.pc
dso['dist_err'].unit = u.pc

dso = dso[dso['ObjType'] != 'Neb']

#%%

v = Vizier(columns=['RAJ2000','DEJ2000','cst'])
v.ROW_LIMIT = -1
iau_borders = v.get_catalogs(catalog='VI/49/bound_20')[0]

cst_fullname = sorted(list(set(data['Constellation'])))
with open('iau_abbrev.txt','r') as f:
    cst_abbrev = f.readlines()
    cst_abbrev = [line.rstrip() for line in cst_abbrev]
    
hgv_data = pd.read_csv('hygdata_v3.csv')
hgv_data['con'] = hgv_data['con'].str.upper()
want = hgv_data['mag'] <= 6.5
hgv_data = hgv_data[want]

# following code from
# https://github.com/eleanorlutz/western_constellations_atlas_of_space/blob/main/6_plot_maps.ipynb
con_lines = pd.read_csv('constellationship.fab', header=None)
con_lines['constellation'] = con_lines[0].str.split().str.get(0)
con_lines['constellation'] = con_lines['constellation'].str.upper()
con_lines['num_pairs'] = con_lines[0].str.split().str.get(1)
con_lines['stars'] = con_lines[0].str.split().str[2:]
con_lines.drop(0, axis=1, inplace=True)

stars = [float(y) for x in con_lines['stars'].tolist() for y in x]
stars = sorted(set(stars))

ras, decs, = [], []
for star in stars: 
    temp = hgv_data[hgv_data['hip']==star]
    assert len(temp) == 1
    ras.append(temp['ra'].tolist()[0])
    decs.append(temp['dec'].tolist()[0])

star_df = pd.DataFrame(data={'star_ID':stars, 'ra':ras, 'dec':decs})
con_lines['ra'] = ''
con_lines['dec'] = ''

for index, row in con_lines.iterrows(): 
    ras, decs = [], []
    for star in row['stars']: 
        temp = hgv_data[hgv_data['hip']==float(star)]
        assert len(temp) == 1
        ras.append(temp['ra'].tolist()[0])
        decs.append(temp['dec'].tolist()[0])
    con_lines.at[index, 'ra'] = ras
    con_lines.at[index, 'dec'] = decs
    
#%%

with open('iau_neighbors.txt','r') as f:
    cst_neigh = f.readlines()
    cst_neigh = [line.replace(',','').split() for line in cst_neigh]

fig1 = plt.figure(figsize=(6,6), constrained_layout=True)
ax1 = plt.subplot(111)

# Andromeda, Cassiopeia, Cepheus, Octans, Pegasus, Phoenix, Pisces, Sculptor
# Tucana, Ursa Minor
main_cst = 'Vulpecula'
cst_index = cst_fullname.index(main_cst)
main_abbrev = cst_abbrev[cst_index]
main_borders = iau_borders[iau_borders['cst'] == main_abbrev]
main_data = data_coords[data_coords['Constellation'] == main_cst]
if main_abbrev == 'SER1' or main_abbrev == 'SER2':
    extra_stars = hgv_data[hgv_data['con'] == 'SER']
else:
    extra_stars = hgv_data[hgv_data['con'] == main_abbrev]
main_lines = con_lines[con_lines['constellation'] == main_abbrev]
main_dso = dso[dso['Constellation'] == main_cst]

lon_0 = (lon2ra(np.median(main_borders['RAJ2000'])) % 24)*15
ra_big = np.zeros(0)
ra_small = np.zeros(0)
ra_all = np.zeros(0)
near_zero = ['Andromeda','Cassiopeia','Cepheus','Octans','Pegasus',
             'Phoenix','Pisces','Sculptor','Tucana','Ursa Minor']
if main_cst in near_zero:
    want = np.logical_and(main_borders['RAJ2000'] < 360,main_borders['RAJ2000'] > 12*15)
    for i in range(0,len(main_borders[want]['RAJ2000'])):
        ra_big = np.append(ra_big,main_borders[want][i]['RAJ2000'] - 360)
        ra_all = np.append(ra_all,main_borders[want][i]['RAJ2000'] - 360)
    for i in range(0,len(main_borders[~want]['RAJ2000'])):
        ra_small = np.append(ra_small,main_borders[~want][i]['RAJ2000'])
        ra_all = np.append(ra_all,main_borders[~want][i]['RAJ2000'])
    lon_0 = np.median(ra_all)
    ra_width = np.abs(np.min(ra_big) - np.max(ra_small))
else:
    ra_max = np.max(main_borders['RAJ2000']) - lon_0
    ra_min = np.min(main_borders['RAJ2000']) - lon_0
    if lon_0 > 180:
        ra_width = np.abs(ra_max - ra_min)
        lon_0 = lon_0 - 360
    if ra_max > 180:
        ra_width = np.abs(np.abs(ra_max-360) - ra_min)
    else:
        ra_width = np.abs(ra_max - ra_min)
lat_0 = np.median(main_borders['DEJ2000'])
dec_width = np.abs(np.max(main_borders['DEJ2000']) - np.min(main_borders['DEJ2000']))

if main_cst == 'Cepheus':
    lat_0 = 75
    lon_0 = -30
elif main_cst == 'Pisces':
    lat_0 = 15
elif main_cst == 'Ursa Minor':
    lat_0 = 75
    lon_0 = 15*15
    ra_width = 4*15
elif main_cst == 'Cetus':
    lon_0 = 20
elif main_cst == 'Camelopardalis':
    lon_0 = 5*15
    lat_0 = 75


rsphere = 6370997
height = rsphere*np.radians(dec_width+20)
width = rsphere*np.abs(np.radians(ra_width+30)*np.cos(np.radians(lat_0)))

m = Basemap(projection='lcc',resolution=None,
            lat_0 = lat_0, lon_0 = lon_0,
            lat_1 = lat_0,
            width=-width, height=height, celestial=False)

for index, row in main_lines.iterrows():
    ras = [float(x)*360/24 for x in row['ra']]
    decs = [float(x) for x in row['dec']]

for i in range(len(ras)):
    m.plot(ras[i*2:(i+1)*2],decs[i*2:(i+1)*2],ls='-',lw=1,color='limegreen',latlon=True)


coord = SkyCoord(main_data['RA'],main_data['Dec'])
extra_coord = SkyCoord(extra_stars['ra']*u.hour,extra_stars['dec']*u.deg)
idx, d2d, d3d = coord.match_to_catalog_sky(extra_coord)
sep_constraint = d2d < 1*u.arcsec
idx = idx[sep_constraint]
inv_idx = np.zeros(0)
for i in range(0,len(extra_coord)):
    if i not in idx:
        inv_idx = np.append(inv_idx,i)
inv_idx = inv_idx.astype(int)
extra_coord = extra_coord[inv_idx]

extra_sizes = list(star_sizes(extra_stars['mag'].iloc[inv_idx]))
extra_colors = list(star_colors(extra_stars['ci'].iloc[inv_idx]))
for i in range(0,len(extra_coord)):
    m.plot(extra_coord.ra.value[i],extra_coord.dec.value[i],marker='o',color=extra_colors[i],
           markersize=extra_sizes[i],latlon=True,ls='None',markeredgecolor='k',markeredgewidth=0.5)
colors = list(star_colors(main_data['B-V']))
sizes = list(star_sizes(main_data['V']))
texts = np.zeros(0)
for i in range(0,len(main_data)):
    m.plot(coord.ra.value[i],coord.dec.value[i],marker='o',color=colors[i],
              markersize=sizes[i],ls='None',latlon=True,markeredgecolor='k',markeredgewidth=0.5)
    x, y = m(coord.ra.value[i],coord.dec.value[i])
    if main_data[i]['LaTeX'] != '--':
        texts = np.append(texts, [ax1.annotate(r'{:}'.format(main_data[i]['LaTeX']),(x,y),fontsize=10)])
adjust_text(texts)
m.drawparallels(np.arange(-90,100,10), linewidth=1, labels=[True,True,False,False], fmt=lat2dec_fmt, zorder=-1)
m.drawmeridians(np.arange(-180,180,15), linewidth=1, labels=[False,False,True,True], fmt=lon2ra_fmt, zorder=-1)

# want_gal = np.logical_or(main_dso['ObjType'] == 'SpiGal',main_dso['ObjType'] == 'EllGal')
# want_gal = np.logical_or(want_gal, main_dso['ObjType'] == 'IrrGal')
# t = np.linspace(0, 2*np.pi, 100)
# for i in main_dso[want_gal]:
#     Ell = np.array([i['R1'].to(u.deg).value*np.cos(t),
#                     i['R2'].to(u.deg).value*np.sin(t)])
#     R_rot = np.array([[np.cos(np.radians(i['Angle'])),
#                        -np.sin(np.radians(i['Angle']))],
#                       [np.sin(np.radians(i['Angle'])),
#                       np.cos(np.radians(i['Angle']))]])
#     Ell_rot = np.zeros((2,Ell.shape[1]))
#     for j in range(Ell.shape[1]):
#         Ell_rot[:,j] = np.dot(R_rot,Ell[:,j])
    
#     m.plot(i['RA'].value+Ell_rot[0,:], 
#            i['Dec'].value+Ell_rot[1,:],color='r',ls='-',latlon=True,lw=1)
    
# want_cluster = np.logical_or(main_dso['ObjType'] == 'OC',main_dso['ObjType'] == 'GC')
# t = np.linspace(0, 2*np.pi, 100)
# for i in main_dso[want_cluster]:
#     Ell = np.array([i['R1'].to(u.deg).value*np.cos(t),
#                     i['R2'].to(u.deg).value*np.sin(t)])
#     R_rot = np.array([[np.cos(np.radians(i['Angle'])),
#                        -np.sin(np.radians(i['Angle']))],
#                       [np.sin(np.radians(i['Angle'])),
#                       np.cos(np.radians(i['Angle']))]])
#     Ell_rot = np.zeros((2,Ell.shape[1]))
#     for j in range(Ell.shape[1]):
#         Ell_rot[:,j] = np.dot(R_rot,Ell[:,j])
    
#     m.plot(i['RA'].value+Ell_rot[0,:], 
#            i['Dec'].value+Ell_rot[1,:],color='purple',ls='-',latlon=True,lw=1)

patches = []
if cst_neigh[cst_index][0] == 'SER1' or cst_neigh[cst_index][0] == 'SER2':
    for i in cst_neigh[cst_index][1:]:
        if i not in ['SER1','SER2']:
            want = iau_borders['cst'] == i
            # m.plot(iau_borders[want]['RAJ2000'],iau_borders[want]['DEJ2000'],color='k',ls='-',lw=1,latlon=True)
            x, y = m(iau_borders[want]['RAJ2000'], iau_borders[want]['DEJ2000'])
            zone = [list(pair) for pair in zip(x,y)]
            patches.append(Polygon(zone))
            name = cst_fullname[cst_abbrev.index(i)]
            if i == 'SER1' or i == 'SER2':
                i = 'SER'
            extra_lines = con_lines[con_lines['constellation'] == i]
            for index, row in extra_lines.iterrows():
                ras = [float(x)*360/24 for x in row['ra']]
                decs = [float(x) for x in row['dec']]
            for n in range(len(ras)):
                m.plot(ras[n*2:(n+1)*2],decs[n*2:(n+1)*2],ls='-',lw=0.5,color='limegreen',latlon=True)
            extra_data = data_coords[data_coords['Constellation'] == name]
            coord = SkyCoord(extra_data['RA'],extra_data['Dec'])
            border_stars = hgv_data[hgv_data['con'] == i]
            border_coords = SkyCoord(border_stars['ra']*u.hour,border_stars['dec']*u.deg)
            idx, d2d, d3d = coord.match_to_catalog_sky(border_coords)
            sep_constraint = d2d < 1*u.arcsec
            idx = idx[sep_constraint]
            inv_idx = np.zeros(0)
            for i in range(0,len(border_coords)):
                if i not in idx:
                    inv_idx = np.append(inv_idx,i)
            inv_idx = inv_idx.astype(int)
            border_coords = border_coords[inv_idx]
            m.plot(coord.ra.value,coord.dec.value,marker='o',color='k',markersize=2,ls='None',latlon=True)
            m.plot(border_coords.ra.value,border_coords.dec.value,marker='.',color='k',markersize=1,ls='None',latlon=True)
        else:
            want = iau_borders['cst'] == i
            # m.plot(iau_borders[want]['RAJ2000'],iau_borders[want]['DEJ2000'],color='k',ls='-',lw=1,latlon=True)
            x, y = m(iau_borders[want]['RAJ2000'], iau_borders[want]['DEJ2000'])
            zone = [list(pair) for pair in zip(x,y)]
            patches.append(Polygon(zone))
            if i == 'SER1' or i == 'SER2':
                i = 'SER'
            extra_lines = con_lines[con_lines['constellation'] == i]
            for index, row in extra_lines.iterrows():
                ras = [float(x)*360/24 for x in row['ra']]
                decs = [float(x) for x in row['dec']]
            for n in range(len(ras)):
                m.plot(ras[n*2:(n+1)*2],decs[n*2:(n+1)*2],ls='-',lw=0.5,color='limegreen',latlon=True)
else:
    for i in cst_neigh[cst_index][1:]:
        want = iau_borders['cst'] == i
        # m.plot(iau_borders[want]['RAJ2000'],iau_borders[want]['DEJ2000'],color='k',ls='-',lw=1,latlon=True)
        x, y = m(iau_borders[want]['RAJ2000'], iau_borders[want]['DEJ2000'])
        zone = [list(pair) for pair in zip(x,y)]
        patches.append(Polygon(zone))
        name = cst_fullname[cst_abbrev.index(i)]
        if i == 'SER1' or i == 'SER2':
            i = 'SER'
        extra_lines = con_lines[con_lines['constellation'] == i]
        for index, row in extra_lines.iterrows():
            ras = [float(x)*360/24 for x in row['ra']]
            decs = [float(x) for x in row['dec']]
        for n in range(len(ras)):
            m.plot(ras[n*2:(n+1)*2],decs[n*2:(n+1)*2],ls='-',lw=0.5,color='limegreen',latlon=True)
        extra_data = data_coords[data_coords['Constellation'] == name]
        coord = SkyCoord(extra_data['RA'],extra_data['Dec'])
        border_stars = hgv_data[hgv_data['con'] == i]
        border_coords = SkyCoord(border_stars['ra']*u.hour,border_stars['dec']*u.deg)
        idx, d2d, d3d = coord.match_to_catalog_sky(border_coords)
        sep_constraint = d2d < 1*u.arcsec
        idx = idx[sep_constraint]
        inv_idx = np.zeros(0)
        for i in range(0,len(border_coords)):
            if i not in idx:
                inv_idx = np.append(inv_idx,i)
        inv_idx = inv_idx.astype(int)
        border_coords = border_coords[inv_idx]
        m.plot(coord.ra.value,coord.dec.value,marker='o',color='k',markersize=2,ls='None',latlon=True)
        m.plot(border_coords.ra.value,border_coords.dec.value,marker='.',color='k',markersize=1,ls='None',latlon=True)

        
ax1.add_collection(PatchCollection(patches, facecolor='silver', alpha=1, edgecolor='black', lw=1, zorder=-2))
plt.title(main_cst, pad=25)
plt.savefig('Constellation_Figures/{:}.pdf'.format(main_cst))

