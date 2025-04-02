import io
import math
import os
import pickle
import sys
from math import pi

import geopandas as gpd
try:
    import geoplot as gplt
except:
    pass
import mapclassify as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import scipy.io as scio
# import seaborn as sns
import shapely
# import skimage.io as sio
from mpl_toolkits import axes_grid1
# from numpy.core.getlimits import MachArLike
# from shapely.geometry import Point
# from pyecharts import options as opts
# from pyecharts.charts import Geo
# from pyecharts.globals import ChartType, SymbolType

from callUtility import (VreYearSplit, dirFlag, getBound, getCRF, getRamp,
                         getResDir, getWorkDir, line2list, makeDir)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

dir_flag = dirFlag()
work_dir = getWorkDir()
cell_area = 781  # km^2


def windsolarCellMap(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    if not os.path.exists(os.path.join(out_output_processed_path, "graphs")):
        os.makedirs(os.path.join(out_output_processed_path, "graphs"))

    # Read outputs
    wind_shp = {'new': [], 'polygon': [], 'ins_and_new': [], 'cof': [], 'density': []}
    offshore_shp = {'new': [], 'polygon': [], 'ins_and_new': [], 'cof': [], 'density': []}
    solar_shp = {'new': [], 'polygon': [], 'ins_and_new': [], 'cof': [], 'density': []}  # includes both UPV/DPV
    dpv_shp = {'new': [], 'polygon': [], 'ins_and_new': [], 'cof': [], 'density': []}

    fres_wind_info = open(os.path.join(out_output_processed_path, 'wind_info.csv'), 'r+')
    for line in fres_wind_info:
        line = line.replace('\n', '')
        line = line.split(',')
        if eval(line[1]) > 0:
            new_cap = (eval(line[1]) - eval(line[10])) * eval(line[4])
            IN_cap = eval(line[1]) * eval(line[4])
            cof = eval(line[1])

            density = 1000 * IN_cap / cell_area

            wind_shp['new'].append(new_cap)
            wind_shp['ins_and_new'].append(IN_cap)
            wind_shp['cof'].append(cof)
            wind_shp['density'].append(density)
            x = eval(line[2])  # lon
            y = eval(line[3])  # lat

            wind_shp['polygon'].append(
                shapely.geometry.Polygon(
                    [(x - 0.15625, y - 0.125),
                     (x + 0.15625, y - 0.125),
                     (x + 0.15625, y + 0.125),
                     (x - 0.15625, y + 0.125)]
                )
            )

            # Identify if this is a cell for onshore (1) or offshore (0)
            is_onshore = eval(line[9])
            if not is_onshore:
                offshore_shp['new'].append(new_cap)
                offshore_shp['ins_and_new'].append(IN_cap)
                offshore_shp['cof'].append(cof)
                offshore_shp['density'].append(density)
                x = eval(line[2])  # lon
                y = eval(line[3])  # lat
                offshore_shp['polygon'].append(
                    shapely.geometry.Polygon(
                        [(x - 0.15625, y - 0.125),
                         (x + 0.15625, y - 0.125),
                         (x + 0.15625, y + 0.125),
                         (x - 0.15625, y + 0.125)]
                    )
                )

    fres_wind_info.close()

    solar_count = {}
    count = 0
    fres_solar_info = open(os.path.join(out_output_processed_path, 'solar_info.csv'), 'r+')
    for line in fres_solar_info:
        line = line.replace('\n', '')
        line = line.split(',')
        if eval(line[1]) > 0:
            new_cap = (eval(line[1]) - eval(line[10])) * eval(line[4])

            IN_cap = eval(line[1]) * eval(line[4])

            cof = eval(line[1])

            density = 1000 * IN_cap / cell_area

            solar_shp['new'].append(new_cap)

            solar_shp['ins_and_new'].append(IN_cap)

            solar_shp['cof'].append(cof)
            solar_shp['density'].append(density)
            x = eval(line[2])
            y = eval(line[3])
            solar_count[(x, y)] = count
            count += 1
            solar_shp['polygon'].append(
                shapely.geometry.Polygon([
                    (x - 0.15625, y - 0.125),
                    (x + 0.15625, y - 0.125),
                    (x + 0.15625, y + 0.125),
                    (x - 0.15625, y + 0.125)
                ])
            )

            # Identify if this is a cell for onshore (1) or offshore (0)
            is_utility = eval(line[9])
            if not is_utility:
                dpv_shp['new'].append(new_cap)
                dpv_shp['ins_and_new'].append(IN_cap)
                dpv_shp['cof'].append(cof)
                dpv_shp['density'].append(density)
                x = eval(line[2])  # lon
                y = eval(line[3])  # lat
                dpv_shp['polygon'].append(
                    shapely.geometry.Polygon(
                        [(x - 0.15625, y - 0.125),
                         (x + 0.15625, y - 0.125),
                         (x + 0.15625, y + 0.125),
                         (x - 0.15625, y + 0.125)]
                    )
                )

            # if eval(line[9]) == 0:
            #     solar_shp['new'].append(new_cap)
            #
            #     solar_shp['ins_and_new'].append(IN_cap)
            #
            #     solar_shp['cof'].append(cof)
            #     solar_shp['density'].append(density)
            #     x = eval(line[2])
            #     y = eval(line[3])
            #     solar_count[(x, y)] = count
            #     count += 1
            #     solar_shp['polygon'].append(
            #         shapely.geometry.Polygon([
            #             (x - 0.15625, y - 0.125),
            #             (x + 0.15625, y - 0.125),
            #             (x + 0.15625, y + 0.125),
            #             (x - 0.15625, y + 0.125)
            #         ])
            #     )
            # elif eval(line[9]) == 1:
            #     dpv_shp['new'].append(new_cap)
            #
            #     dpv_shp['ins_and_new'].append(IN_cap)
            #
            #     density = 1000 * IN_cap / cell_area
            #
            #     dpv_shp['cof'].append(cof)
            #     dpv_shp['density'].append(density)
            #     x = eval(line[2])
            #     y = eval(line[3])
            #     dpv_shp['polygon'].append(
            #         shapely.geometry.Polygon([
            #             (x - 0.15625, y - 0.125),
            #             (x + 0.15625, y - 0.125),
            #             (x + 0.15625, y + 0.125),
            #             (x - 0.15625, y + 0.125)
            #         ])
            #     )
            #
            #     if (x, y) in solar_count:
            #         solar_shp['ins_and_new'][solar_count[(x, y)]] += IN_cap
            #         solar_shp['density'][solar_count[(x, y)]] += density
            #     else:
            #         solar_shp['new'].append(new_cap)
            #         solar_shp['ins_and_new'].append(IN_cap)
            #         solar_shp['cof'].append(cof)
            #         solar_shp['density'].append(density)
            #         solar_shp['polygon'].append(
            #             shapely.geometry.Polygon([
            #                 (x - 0.15625, y - 0.125),
            #                 (x + 0.15625, y - 0.125),
            #                 (x + 0.15625, y + 0.125),
            #                 (x - 0.15625, y + 0.125)
            #             ])
            #         )
    fres_solar_info.close()

    # print(max(solar_shp['ins_and_new']))
    solar_shp['polygon'].append(
        shapely.geometry.Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])
    )
    solar_shp['new'].append(0)
    solar_shp['ins_and_new'].append(15)
    solar_shp['cof'].append(0)
    solar_shp['density'].append(0)

    dpv_shp['polygon'].append(
        shapely.geometry.Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])
    )
    dpv_shp['new'].append(0)
    dpv_shp['ins_and_new'].append(15)
    dpv_shp['cof'].append(0)
    dpv_shp['density'].append(0)

    # Add an arbitrary value set for plotting
    wind_shp['polygon'].append(
        shapely.geometry.Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])
    )
    wind_shp['new'].append(0)
    wind_shp['ins_and_new'].append(5)
    wind_shp['cof'].append(0)
    wind_shp['density'].append(0)

    offshore_shp['polygon'].append(
        shapely.geometry.Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])
    )
    offshore_shp['new'].append(0)
    offshore_shp['ins_and_new'].append(5)
    offshore_shp['cof'].append(0)
    offshore_shp['density'].append(0)

    wind_gdf = gpd.GeoDataFrame(wind_shp, geometry=wind_shp['polygon'])
    offshore_gdf = gpd.GeoDataFrame(offshore_shp, geometry=offshore_shp["polygon"])
    solar_gdf = gpd.GeoDataFrame(solar_shp, geometry=solar_shp['polygon'])
    dpv_gdf = gpd.GeoDataFrame(dpv_shp, geometry=dpv_shp['polygon'])

    # re_gdf = {'wind': wind_gdf, 'solar': solar_gdf, 'dpv': dpv_gdf}
    re_gdf = {'wind': wind_gdf, 'solar': solar_gdf}
    # g_kind = {'Installed_and_New': 'ins_and_new', 'Share': 'cof', 'Density': 'density'}
    g_kind = {'Installed_and_New': 'ins_and_new'}

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_country.shp')
    # china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')
    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    bound = getBound()

    # Change the default font family
    font = {'family': 'Arial', 'weight': 'bold', 'size': 24}
    plt.rcParams.update({'font.family': font["family"]})

    for et in re_gdf:
        if et == 'wind':
            cmap_2 = 'cividis_r'
            cmap = 'PiYG_r'
        elif et == 'solar':
            cmap_2 = 'cividis_r'
            cmap = 'PiYG_r'
        for gk in g_kind:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()

            # Add colormaps
            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size="5%", pad=0.4)
            cax_2 = divider.append_axes('right', size="5%", pad=0.4)

            # Change the formatting of the divider (cax)
            cax.xaxis.set_label_position("top")
            cax.tick_params(labelcolor="w")
            cax.xaxis.labelpad = 10
            cax_2.xaxis.set_label_position("top")
            cax_2.xaxis.labelpad = 10
            cax_2.yaxis.labelpad = 15
            if et == 'wind':
                cax.xaxis.set_label_text("Onshore")
                cax_2.xaxis.set_label_text("Offshore")
            else:
                cax.xaxis.set_label_text("Utility-scale")
                cax_2.xaxis.set_label_text("Distributed")

            if gk == 'Share':
                ax = re_gdf[et].plot(column=g_kind[gk], ax=ax, legend=True, legend_kwds={'label': 'Share (0~1)'},
                                     cax=cax, cmap=cmap)
            elif gk == 'Density':
                ax = re_gdf[et].plot(column=g_kind[gk], ax=ax, legend=True,
                                     legend_kwds={'label': 'Density (MW / km$^2$)'}, cax=cax, cmap=cmap)
            else:  # Installed + new capacity
                ax = re_gdf[et].plot(column=g_kind[gk], ax=ax, legend=True, cax=cax, cmap=cmap)
                if et == "wind":
                    # Add a separate offshore wind layer
                    ax = offshore_gdf.plot(column=g_kind[gk], ax=ax, legend=True, cax=cax_2, cmap=cmap_2,
                                           legend_kwds={'label': 'Capacity (GW)'})
                else:
                    # Add a separate distributed solar layer
                    ax = dpv_gdf.plot(column=g_kind[gk], ax=ax, legend=True, cax=cax_2, cmap=cmap_2,
                                      legend_kwds={'label': 'Capacity (GW)'})

            ax = china_nine_lines.geometry.plot(ax=ax, edgecolor='#747678', facecolor='None', linewidth=0.5)
            ax = china.geometry.plot(ax=ax, edgecolor='#747678', facecolor='None', linewidth=0.4)
            ax = china_grids.geometry.plot(ax=ax, edgecolor='#747678', facecolor='None', alpha=0, linewidth=0.5)

            ax.axis('off')
            ax.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
            ax.set_ylim(bound.geometry[0].y, bound.geometry[1].y)

            # if et == 'wind':
            #     ax.set_title(f'Installed {et} capacities', font=font)
            # elif et == 'solar':
            #     ax.set_title(f'Installed {et} capacities', font=font)
            ax.set_title(f'{curr_year}', font=font)

            ax_child = fig.add_axes((0.58, 0.22, 0.2, 0.2))
            re_gdf[et].plot(column=g_kind[gk], ax=ax_child, legend=False, cmap=cmap)
            if et == "wind":
                offshore_gdf.plot(column=g_kind[gk], ax=ax_child, legend=False, cmap=cmap_2)
            else:
                dpv_gdf.plot(column=g_kind[gk], ax=ax_child, legend=False, cmap=cmap_2)

            ax_child = china_nine_lines.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', linewidth=0.3)
            ax_child = china.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', linewidth=0.2)
            ax_child = china_grids.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', alpha=0, linewidth=0.3)

            ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
            ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
            ax_child.set_xticks([])
            ax_child.set_yticks([])

            # Save plots
            fig.savefig(os.path.join(out_output_processed_path, "graphs",
                                     gk + '_' + et + '_' + 'CellMap.png'),
                        dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(out_output_processed_path, "graphs",
                                     gk + '_' + et + '_' + 'CellMap.svg'),
                        dpi=300, bbox_inches='tight')
            plt.close()


def windsolarSpurMap():
    dir_flag = dirFlag()
    work_dir = getWorkDir()

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin_py']

    with open(work_dir + "data_pkl" + dir_flag + 'wind_cell_2016.pkl', 'rb') as fin:
        wind_cell = pickle.load(fin)
    fin.close()

    with open(work_dir + "data_pkl" + dir_flag + 'solar_cell_2015.pkl', 'rb') as fin:
        solar_cell = pickle.load(fin)
    fin.close()

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china.shp')
    china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')

    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    wind_shp = {'id': [], 'lon': [], 'lat': [], 'spur_cost': [], 'polygon': [], 'spur_cost_kw': []}
    wind_added = []

    solar_shp = {'id': [], 'lon': [], 'lat': [], 'spur_cost': [], 'polygon': [], 'spur_cost_kw': []}

    spur_capex = 3.25 * 2.561  # $/kw-km
    spur_capex_fixed = 0  # $/kw

    bound = getBound()

    for pro in provins:
        for i in range(len(wind_cell['provin_cf_sort'][pro])):
            wind_shp['id'].append(wind_cell['provin_cf_sort'][pro][i][0])
            wind_shp['lat'].append(wind_cell['provin_cf_sort'][pro][i][7])
            wind_shp['lon'].append(wind_cell['provin_cf_sort'][pro][i][8])
            if wind_cell['provin_cf_sort'][pro][i][9] >= 0.1:
                wind_shp['spur_cost'].append(0.101)
            else:
                wind_shp['spur_cost'].append(wind_cell['provin_cf_sort'][pro][i][9])

            wind_shp['spur_cost_kw'].append(spur_capex * wind_cell['provin_cf_sort'][pro][i][11] + spur_capex_fixed)
            x = wind_cell['provin_cf_sort'][pro][i][7]
            y = wind_cell['provin_cf_sort'][pro][i][8]
            wind_added.append((x, y))
            wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                 (x + 0.15625, y - 0.125),
                                                                 (x + 0.15625, y + 0.125),
                                                                 (x - 0.15625, y + 0.125)]))
        for i in range(len(solar_cell['provin_cf_sort'][pro])):
            solar_shp['id'].append(solar_cell['provin_cf_sort'][pro][i][0])
            solar_shp['lat'].append(solar_cell['provin_cf_sort'][pro][i][7])
            solar_shp['lon'].append(solar_cell['provin_cf_sort'][pro][i][8])

            if solar_cell['provin_cf_sort'][pro][i][9] >= 0.1:
                solar_shp['spur_cost'].append(0.1)
            else:
                solar_shp['spur_cost'].append(solar_cell['provin_cf_sort'][pro][i][9])

            solar_shp['spur_cost_kw'].append(spur_capex * solar_cell['provin_cf_sort'][pro][i][11] + spur_capex_fixed)
            x = solar_cell['provin_cf_sort'][pro][i][7]
            y = solar_cell['provin_cf_sort'][pro][i][8]

            solar_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                  (x + 0.15625, y - 0.125),
                                                                  (x + 0.15625, y + 0.125),
                                                                  (x - 0.15625, y + 0.125)]))

    f_wind_missing = open(work_dir + 'data_csv' + dir_flag + 'vre_installations/inter_connect_China_windpower_onshore_provin_2016.csv',
                          'r+')
    for line in f_wind_missing:
        line = line.replace('\n', '')
        line = line.split(',')
        x = eval(line[4])
        y = eval(line[3])
        if (x, y) not in wind_added:
            wind_shp['id'].append(eval(line[0]))
            wind_shp['lat'].append(eval(line[3]))
            wind_shp['lon'].append(eval(line[4]))
            wind_shp['spur_cost'].append(0.1)
            wind_shp['spur_cost_kw'].append(100)
            wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                 (x + 0.15625, y - 0.125),
                                                                 (x + 0.15625, y + 0.125),
                                                                 (x - 0.15625, y + 0.125)]))

    f_wind_missing.close()

    wind_gdf = gpd.GeoDataFrame(wind_shp, geometry=wind_shp['polygon'])
    solar_gdf = gpd.GeoDataFrame(solar_shp, geometry=solar_shp['polygon'])
    gdf = {'Wind': wind_gdf, 'Solar': solar_gdf}

    g_kind = {'Spur Cost': 'spur_cost', 'Spur Price': 'spur_cost_kw'}
    label = {'Spur Cost': 'LCOE (yuan/kWh)', 'Spur Price': 'Price (yuan/kW)'}
    fig_file = {'Spur Cost': '_spur_cost_map.png', 'Spur Price': '_spur_price_map.png'}

    for et in gdf:
        for gk in g_kind:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()
            # ax = fig.add_axes((0,0,1,1))

            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size="4%", pad=0.05)

            gdf[et].plot(column=g_kind[gk], ax=ax, cmap='YlGn', legend=True, legend_kwds={'label': label[gk]}, cax=cax)
            ax = china_nine_lines.geometry.plot(ax=ax, color='black')
            ax = china_grids.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)
            ax = china.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.1)

            ax.axis('off')
            ax.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
            ax.set_ylim(bound.geometry[0].y, bound.geometry[1].y)
            ax.set_title(gk + ' of ' + et, fontweight='bold')

            ax_child = fig.add_axes((0.65, 0.19, 0.2, 0.2))
            wind_gdf.plot(column=g_kind[gk], ax=ax_child, cmap='YlGn', legend=False)
            ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black')
            ax_child = china_grids.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.2)
            ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.1)
            ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
            ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
            ax_child.set_xticks([])
            ax_child.set_yticks([])
            fig.savefig(work_dir + 'data_fig' + dir_flag + et + fig_file[gk], dpi=600)


def windsolarTrunkMap():
    dir_flag = dirFlag()
    work_dir = getWorkDir()

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin_py']

    with open(work_dir + "data_pkl" + dir_flag + 'wind_cell_2016.pkl', 'rb') as fin:
        wind_cell = pickle.load(fin)
    fin.close()

    with open(work_dir + "data_pkl" + dir_flag + 'solar_cell_2015.pkl', 'rb') as fin:
        solar_cell = pickle.load(fin)
    fin.close()

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china.shp')
    china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')

    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    wind_shp = {'id': [], 'lon': [], 'lat': [], 'trunk_cost': [], 'polygon': [], 'trunk_cost_kw': []}
    wind_added = []

    solar_shp = {'id': [], 'lon': [], 'lat': [], 'trunk_cost': [], 'polygon': [], 'trunk_cost_kw': []}
    bound = getBound()

    trunk_capex = 3.25 * 2.561  # $/kw-km
    trunk_capex_fixed = 0  # $/kw

    for pro in provins:
        for i in range(len(wind_cell['provin_cf_sort'][pro])):

            wind_shp['id'].append(wind_cell['provin_cf_sort'][pro][i][0])
            wind_shp['lat'].append(wind_cell['provin_cf_sort'][pro][i][7])
            wind_shp['lon'].append(wind_cell['provin_cf_sort'][pro][i][8])
            if wind_cell['provin_cf_sort'][pro][i][10] >= 0.2:
                wind_shp['trunk_cost'].append(0.202)
            else:
                wind_shp['trunk_cost'].append(wind_cell['provin_cf_sort'][pro][i][10])
            wind_shp['trunk_cost_kw'].append(trunk_capex * wind_cell['provin_cf_sort'][pro][i][12] + trunk_capex_fixed)
            x = wind_cell['provin_cf_sort'][pro][i][7]
            y = wind_cell['provin_cf_sort'][pro][i][8]
            wind_added.append((x, y))
            wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                 (x + 0.15625, y - 0.125),
                                                                 (x + 0.15625, y + 0.125),
                                                                 (x - 0.15625, y + 0.125)]))
        for i in range(len(solar_cell['provin_cf_sort'][pro])):

            solar_shp['id'].append(solar_cell['provin_cf_sort'][pro][i][0])
            solar_shp['lat'].append(solar_cell['provin_cf_sort'][pro][i][7])
            solar_shp['lon'].append(solar_cell['provin_cf_sort'][pro][i][8])
            if solar_cell['provin_cf_sort'][pro][i][10] >= 0.16:
                solar_shp['trunk_cost'].append(0.16)
            else:
                solar_shp['trunk_cost'].append(solar_cell['provin_cf_sort'][pro][i][10])
            solar_shp['trunk_cost_kw'].append(
                trunk_capex * solar_cell['provin_cf_sort'][pro][i][12] + trunk_capex_fixed)
            x = solar_cell['provin_cf_sort'][pro][i][7]
            y = solar_cell['provin_cf_sort'][pro][i][8]

            solar_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                  (x + 0.15625, y - 0.125),
                                                                  (x + 0.15625, y + 0.125),
                                                                  (x - 0.15625, y + 0.125)]))

    f_wind_missing = open(work_dir + 'data_csv' + dir_flag + 'vre_installations/inter_connect_China_windpower_onshore_provin_2016.csv',
                          'r+')
    for line in f_wind_missing:
        line = line.replace('\n', '')
        line = line.split(',')
        x = eval(line[4])
        y = eval(line[3])
        if (x, y) not in wind_added:
            wind_shp['id'].append(eval(line[0]))
            wind_shp['lat'].append(eval(line[3]))
            wind_shp['lon'].append(eval(line[4]))
            wind_shp['trunk_cost'].append(0.2)
            wind_shp['trunk_cost_kw'].append(2000)
            wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                 (x + 0.15625, y - 0.125),
                                                                 (x + 0.15625, y + 0.125),
                                                                 (x - 0.15625, y + 0.125)]))

    f_wind_missing.close()

    wind_gdf = gpd.GeoDataFrame(wind_shp, geometry=wind_shp['polygon'])
    solar_gdf = gpd.GeoDataFrame(solar_shp, geometry=solar_shp['polygon'])
    gdf = {'Wind': wind_gdf, 'Solar': solar_gdf}

    g_kind = {'Trunk Cost': 'trunk_cost', 'Trunk Price': 'trunk_cost_kw'}
    label = {'Trunk Cost': 'LCOE (yuan/kWh)', 'Trunk Price': 'Price (yuan/kW)'}
    fig_file = {'Trunk Cost': '_trunk_cost_map.png', 'Trunk Price': '_trunk_price_map.png'}

    for et in gdf:
        for gk in g_kind:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()
            # ax = fig.add_axes((0,0,1,1))

            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size="4%", pad=0.1)

            gdf[et].plot(column=g_kind[gk], ax=ax, cmap='YlGn', legend=True, legend_kwds={'label': label[gk]}, cax=cax)
            ax = china_nine_lines.geometry.plot(ax=ax, color='black')
            ax = china_grids.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)
            ax = china.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.1)

            ax.axis('off')
            ax.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
            ax.set_ylim(bound.geometry[0].y, bound.geometry[1].y)
            ax.set_title(gk + ' of ' + et, fontweight='bold')

            ax_child = fig.add_axes((0.65, 0.19, 0.2, 0.2))
            gdf[et].plot(column=g_kind[gk], ax=ax_child, cmap='YlGn', legend=False)
            ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black')
            ax_child = china_grids.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.2)
            ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.1)
            ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
            ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
            ax_child.set_xticks([])
            ax_child.set_yticks([])
            fig.savefig(work_dir + 'data_fig' + dir_flag + et + fig_file[gk], dpi=600)


def windsolarGSTcMap():
    dir_flag = dirFlag()
    work_dir = getWorkDir()

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin_py']

    with open(work_dir + "data_pkl" + dir_flag + 'wind_cell_2016.pkl', 'rb') as fin:
        wind_cell = pickle.load(fin)
    fin.close()

    with open(work_dir + "data_pkl" + dir_flag + 'solar_cell_2015.pkl', 'rb') as fin:
        solar_cell = pickle.load(fin)
    fin.close()

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china.shp')
    china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')

    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    wind_shp = {'id': [], 'lon': [], 'lat': [], 'GSTc': [], 'polygon': []}
    wind_added = []

    solar_shp = {'id': [], 'lon': [], 'lat': [], 'GSTc': [], 'polygon': []}

    bound = getBound()

    for pro in provins:
        for i in range(len(wind_cell['provin_cf_sort'][pro])):
            wind_shp['id'].append(wind_cell['provin_cf_sort'][pro][i][0])
            wind_shp['lat'].append(wind_cell['provin_cf_sort'][pro][i][7])
            wind_shp['lon'].append(wind_cell['provin_cf_sort'][pro][i][8])
            GSTc = wind_cell['provin_cf_sort'][pro][i][5] + wind_cell['provin_cf_sort'][pro][i][9] + \
                   wind_cell['provin_cf_sort'][pro][i][10]
            if GSTc >= 1.0:
                wind_shp['GSTc'].append(1.01)
            else:
                wind_shp['GSTc'].append(GSTc)
            x = wind_cell['provin_cf_sort'][pro][i][7]
            y = wind_cell['provin_cf_sort'][pro][i][8]
            wind_added.append((x, y))
            wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                 (x + 0.15625, y - 0.125),
                                                                 (x + 0.15625, y + 0.125),
                                                                 (x - 0.15625, y + 0.125)]))
        for i in range(len(solar_cell['provin_cf_sort'][pro])):
            solar_shp['id'].append(solar_cell['provin_cf_sort'][pro][i][0])
            solar_shp['lat'].append(solar_cell['provin_cf_sort'][pro][i][7])
            solar_shp['lon'].append(solar_cell['provin_cf_sort'][pro][i][8])
            GSTc = solar_cell['provin_cf_sort'][pro][i][5] + solar_cell['provin_cf_sort'][pro][i][9] + \
                   solar_cell['provin_cf_sort'][pro][i][10]

            if GSTc >= 0.6:
                solar_shp['GSTc'].append(0.61)
            else:
                solar_shp['GSTc'].append(GSTc)

            x = solar_cell['provin_cf_sort'][pro][i][7]
            y = solar_cell['provin_cf_sort'][pro][i][8]

            solar_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                  (x + 0.15625, y - 0.125),
                                                                  (x + 0.15625, y + 0.125),
                                                                  (x - 0.15625, y + 0.125)]))

    f_wind_missing = open(work_dir + 'data_csv' + dir_flag + 'vre_installations/inter_connect_China_windpower_onshore_provin_2016.csv',
                          'r+')
    for line in f_wind_missing:
        line = line.replace('\n', '')
        line = line.split(',')
        x = eval(line[4])
        y = eval(line[3])
        if (x, y) not in wind_added:
            wind_shp['id'].append(eval(line[0]))
            wind_shp['lat'].append(eval(line[3]))
            wind_shp['lon'].append(eval(line[4]))
            wind_shp['GSTc'].append(0.8)
            wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                 (x + 0.15625, y - 0.125),
                                                                 (x + 0.15625, y + 0.125),
                                                                 (x - 0.15625, y + 0.125)]))

    f_wind_missing.close()
    wind_gdf = gpd.GeoDataFrame(wind_shp, geometry=wind_shp['polygon'])
    solar_gdf = gpd.GeoDataFrame(solar_shp, geometry=solar_shp['polygon'])
    gdf = {'wind': wind_gdf, 'solar': solar_gdf}

    for et in gdf:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.gca()
        # ax = fig.add_axes((0,0,1,1))

        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', size="4%", pad=0.05)

        gdf[et].plot(column='GSTc', ax=ax, cmap='YlGn', legend=True, legend_kwds={'label': 'LCOE(yuan/kWh)'}, cax=cax)
        ax = china_nine_lines.geometry.plot(ax=ax, color='black')
        ax = china_grids.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)
        ax = china.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.1)

        ax.axis('off')
        ax.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
        ax.set_ylim(bound.geometry[0].y, bound.geometry[1].y)
        ax.set_title('LCOE of ' + et + ' GST', fontweight='bold')

        ax_child = fig.add_axes((0.65, 0.19, 0.2, 0.2))
        wind_gdf.plot(column='GSTc', ax=ax_child, cmap='YlGn', legend=False)
        ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black')
        ax_child = china_grids.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.2)
        ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.1)
        ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
        ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
        ax_child.set_xticks([])
        ax_child.set_yticks([])
        fig.savefig(work_dir + 'data_fig' + dir_flag + et + 'GSTc_map.png', dpi=500)


def windsolarCFMap():
    dir_flag = dirFlag()
    work_dir = getWorkDir()

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin_py']

    with open(work_dir + "data_pkl" + dir_flag + 'wind_cell_2016.pkl', 'rb') as fin:
        wind_cell = pickle.load(fin)
    fin.close()

    with open(work_dir + "data_pkl" + dir_flag + 'solar_cell_2015.pkl', 'rb') as fin:
        solar_cell = pickle.load(fin)
    fin.close()

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china.shp')
    china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')

    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    wind_shp = {'id': [], 'lon': [], 'lat': [], 'CF': [], 'polygon': []}
    wind_added = []

    solar_shp = {'id': [], 'lon': [], 'lat': [], 'CF': [], 'polygon': []}

    bound = getBound()

    for pro in provins:
        for i in range(len(wind_cell['provin_cf_sort'][pro])):
            wind_shp['id'].append(wind_cell['provin_cf_sort'][pro][i][0])
            wind_shp['lat'].append(wind_cell['provin_cf_sort'][pro][i][7])
            wind_shp['lon'].append(wind_cell['provin_cf_sort'][pro][i][8])
            CF = wind_cell['provin_cf_sort'][pro][i][15]
            if CF > 0.005:
                wind_shp['CF'].append(CF)
            else:
                wind_shp['CF'].append(0.005)
            x = wind_cell['provin_cf_sort'][pro][i][7]
            y = wind_cell['provin_cf_sort'][pro][i][8]
            wind_added.append((x, y))
            wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                 (x + 0.15625, y - 0.125),
                                                                 (x + 0.15625, y + 0.125),
                                                                 (x - 0.15625, y + 0.125)]))
        for i in range(len(solar_cell['provin_cf_sort'][pro])):
            solar_shp['id'].append(solar_cell['provin_cf_sort'][pro][i][0])
            solar_shp['lat'].append(solar_cell['provin_cf_sort'][pro][i][7])
            solar_shp['lon'].append(solar_cell['provin_cf_sort'][pro][i][8])
            CF = solar_cell['provin_cf_sort'][pro][i][15]

            if CF > 0.005:
                solar_shp['CF'].append(CF)
            else:
                solar_shp['CF'].append(0.005)

            x = solar_cell['provin_cf_sort'][pro][i][7]
            y = solar_cell['provin_cf_sort'][pro][i][8]

            solar_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                  (x + 0.15625, y - 0.125),
                                                                  (x + 0.15625, y + 0.125),
                                                                  (x - 0.15625, y + 0.125)]))

    f_wind_missing = open(work_dir + 'data_csv' + dir_flag + 'vre_installations/inter_connect_China_windpower_onshore_provin_2016.csv',
                          'r+')
    for line in f_wind_missing:
        line = line.replace('\n', '')
        line = line.split(',')
        x = eval(line[4])
        y = eval(line[3])
        if (x, y) not in wind_added:
            wind_shp['id'].append(eval(line[0]))
            wind_shp['lat'].append(eval(line[3]))
            wind_shp['lon'].append(eval(line[4]))
            wind_shp['CF'].append(0.005)
            wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                 (x + 0.15625, y - 0.125),
                                                                 (x + 0.15625, y + 0.125),
                                                                 (x - 0.15625, y + 0.125)]))

    f_wind_missing.close()
    wind_gdf = gpd.GeoDataFrame(wind_shp, geometry=wind_shp['polygon'])
    solar_gdf = gpd.GeoDataFrame(solar_shp, geometry=solar_shp['polygon'])
    gdf = {'wind': wind_gdf, 'solar': solar_gdf}

    for et in gdf:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.gca()
        # ax = fig.add_axes((0,0,1,1))

        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', size="4%", pad=0.05)

        gdf[et].plot(column='CF', ax=ax, cmap='YlGn', legend=True, legend_kwds={'label': 'Capacity Factor'}, cax=cax)
        ax = china_nine_lines.geometry.plot(ax=ax, color='black')
        ax = china_grids.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)
        ax = china.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.1)

        ax.axis('off')
        ax.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
        ax.set_ylim(bound.geometry[0].y, bound.geometry[1].y)
        ax.set_title(et + ' capacity factor', fontweight='bold')

        ax_child = fig.add_axes((0.65, 0.19, 0.2, 0.2))
        wind_gdf.plot(column='CF', ax=ax_child, cmap='YlGn', legend=False)
        ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black')
        ax_child = china_grids.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.2)
        ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.1)
        ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
        ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
        ax_child.set_xticks([])
        ax_child.set_yticks([])
        fig.savefig(work_dir + 'data_fig' + dir_flag + et + '_cf_map.png', dpi=500)


def windsolarIntedMap():
    dir_flag = dirFlag()
    work_dir = getWorkDir()

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin_py']

    with open(work_dir + "data_pkl" + dir_flag + 'wind_cell_2015.pkl', 'rb') as fin:
        wind_cell = pickle.load(fin)
    fin.close()

    with open(work_dir + "data_pkl" + dir_flag + 'solar_cell_2015.pkl', 'rb') as fin:
        solar_cell = pickle.load(fin)
    fin.close()

    bound = getBound()

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china.shp')
    china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')

    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    wind_shp = {'id': [], 'lon': [], 'lat': [], 'inted': [], 'polygon': []}

    solar_shp = {'id': [], 'lon': [], 'lat': [], 'inted': [], 'polygon': []}

    for pro in provins:
        if pro != 'Xizang':
            for i in range(len(wind_cell['provin_cf_sort'][pro])):
                # print(pro,len(wind_cell['provin_cf_sort'][pro]))
                if wind_cell['provin_cf_sort'][pro][i][13] > 0:
                    wind_shp['id'].append(wind_cell['provin_cf_sort'][pro][i][0])
                    wind_shp['lat'].append(wind_cell['provin_cf_sort'][pro][i][7])
                    wind_shp['lon'].append(wind_cell['provin_cf_sort'][pro][i][8])
                    inted = wind_cell['provin_cf_sort'][pro][i][13] * wind_cell['provin_cf_sort'][pro][i][3]
                    # if inted >= 1:
                    #    wind_shp['inted'].append(1.02)
                    # else:
                    wind_shp['inted'].append(inted)
                    x = wind_cell['provin_cf_sort'][pro][i][7]
                    y = wind_cell['provin_cf_sort'][pro][i][8]
                    wind_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                         (x + 0.15625, y - 0.125),
                                                                         (x + 0.15625, y + 0.125),
                                                                         (x - 0.15625, y + 0.125)]))
        for i in range(len(solar_cell['provin_cf_sort'][pro])):
            if solar_cell['provin_cf_sort'][pro][i][2] == 0:
                if solar_cell['provin_cf_sort'][pro][i][13] > 0:
                    solar_shp['id'].append(solar_cell['provin_cf_sort'][pro][i][0])
                    solar_shp['lat'].append(solar_cell['provin_cf_sort'][pro][i][7])
                    solar_shp['lon'].append(solar_cell['provin_cf_sort'][pro][i][8])
                    inted = solar_cell['provin_cf_sort'][pro][i][13] * solar_cell['provin_cf_sort'][pro][i][3]

                    # if inted >= 1:
                    #    solar_shp['inted'].append(1.02)
                    # else:
                    solar_shp['inted'].append(inted)
                    x = solar_cell['provin_cf_sort'][pro][i][7]
                    y = solar_cell['provin_cf_sort'][pro][i][8]

                    solar_shp['polygon'].append(shapely.geometry.Polygon([(x - 0.15625, y - 0.125),
                                                                          (x + 0.15625, y - 0.125),
                                                                          (x + 0.15625, y + 0.125),
                                                                          (x - 0.15625, y + 0.125)]))

    wind_gdf = gpd.GeoDataFrame(wind_shp, geometry=wind_shp['polygon'])
    solar_gdf = gpd.GeoDataFrame(solar_shp, geometry=solar_shp['polygon'])
    gdf = {'wind': wind_gdf, 'solar': solar_gdf}

    for et in gdf:
        fig = plt.figure(figsize=(12.6, 8.4))
        ax = plt.gca()
        cmap = 'plasma_r'

        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', size="4%", pad=0.1)

        gdf[et].plot(column='inted', ax=ax, legend=True, legend_kwds={'label': 'Capacity (GW)'}, cax=cax, cmap=cmap)

        ax.axis('off')
        ax.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
        ax.set_ylim(bound.geometry[0].y, bound.geometry[1].y)

        # ax.set_title('Integrated '+et,fontweight='bold')

        ax_child = fig.add_axes((0.65, 0.19, 0.2, 0.2))
        gdf[et].plot(column='inted', ax=ax_child, legend=False, cmap=cmap)
        ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black')
        ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.1)
        ax_child = china_grids.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.2)
        ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
        ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)

        ax_child.set_xticks([])
        ax_child.set_yticks([])
        fig.savefig(work_dir + 'data_fig' + dir_flag + et + '_inted_map.png', dpi=600)


def windsolarInstalledFactor(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    if not os.path.exists(os.path.join(out_output_processed_path, "graphs")):
        os.makedirs(os.path.join(out_output_processed_path, "graphs"))

    cell_if = {'solar': {}, 'wind': {}}
    for re in ['solar', 'wind']:
        f_re_info = open(os.path.join(out_output_processed_path, re + '_info.csv'), 'r+')
        for line in f_re_info:
            line = line.replace('\n', '')
            line = line.split(',')
            if line[8] not in cell_if[re]:
                cell_if[re][line[8]] = []

            cell_if[re][line[8]].append(eval(line[1]))
        f_re_info.close()

    pro_if = {'solar': {}, 'wind': {}}

    for re in ['solar', 'wind']:
        for pro in cell_if[re]:
            pro_if[re][pro] = sum(cell_if[re][pro]) / len(cell_if[re][pro])

    pro_if['wind']['Xizang'] = 0

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_country.shp')
    # china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')
    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    bound = getBound()

    china_nmsplit = gpd.read_file(
        work_dir + 'data_shp' + dir_flag + 'province' + dir_flag + 'china_provinces_nmsplit.shp')

    # print(china['FENAME'])
    solar_shp = {'geometry': [], 'cof': [], 'NAME': []}
    wind_shp = {'geometry': [], 'cof': [], 'NAME': []}

    shp = {'solar': solar_shp, 'wind': wind_shp}

    for re in ['solar', 'wind']:
        for i in range(len(china_nmsplit['OBJECTID'])):
            if china_nmsplit['NAME_PY'][i] != 'Hong Kong' and china_nmsplit['NAME_PY'][i] != 'Macau':
                # if china_nmsplit['NAME_PY'][i] == 'Xizang':
                #     shp[re]['geometry'].append(china['geometry'][26])
                # elif china_nmsplit['NAME_PY'][i] == 'Gansu':
                #     shp[re]['geometry'].append(china['geometry'][24])
                # elif china_nmsplit['NAME_PY'][i] == 'Qinghai':
                #     shp[re]['geometry'].append(china['geometry'][31])
                # elif china_nmsplit['NAME_PY'][i] == 'Xinjiang':
                #     shp[re]['geometry'].append(china['geometry'][14])
                # elif china_nmsplit['NAME_PY'][i] == 'Sichuan':
                #     shp[re]['geometry'].append(china['geometry'][6])
                # else:
                #     shp[re]['geometry'].append(china_nmsplit['geometry'][i])
                shp[re]['geometry'].append(china_nmsplit['geometry'][i])
                shp[re]['NAME'].append(china_nmsplit['NAME_PY'][i])
                shp[re]['cof'].append(pro_if[re][china_nmsplit['NAME_PY'][i]])

    wind_gdf = gpd.GeoDataFrame(wind_shp, geometry=wind_shp['geometry'], crs='EPSG:4326')
    solar_gdf = gpd.GeoDataFrame(solar_shp, geometry=solar_shp['geometry'], crs='EPSG:4326')

    re_gdf = {'wind': wind_gdf, 'solar': solar_gdf}
    g_kind = {'Installed Factor': 'cof'}

    albers_proj = '+proj=aea +lat_1=25 +lat_2=47 +lon_0=105'
    for et in re_gdf:
        for gk in g_kind:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()

            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size="5%", pad=0.1)

            ax = china_nine_lines.geometry.plot(ax=ax, edgecolor='#747678', facecolor='None', linewidth=0.5)
            ax = china.geometry.plot(ax=ax, edgecolor='#747678', facecolor='None', linewidth=0.4)
            ax = china_grids.geometry.plot(ax=ax, edgecolor='#747678', facecolor='None', alpha=0, linewidth=0.5)

            if et == "wind":
                cmap = "cividis_r"
            else:
                cmap = "cividis_r"
            # if et == "wind":
            #     cmap = "cividis_r"
            # else:
            #     cmap = "plasma_r"
            re_gdf[et].plot(column=g_kind[gk], ax=ax, legend=True, legend_kwds={'label': 'Usage of available land'},
                            cax=cax, cmap=cmap,
                            vmin=0, vmax=0.8)

            ax.axis('off')
            ax.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
            ax.set_ylim(bound.geometry[0].y, bound.geometry[1].y)
            font = {'family': 'Arial', 'weight': 'bold', 'size': 24}

            ax.set_title(f'{curr_year}', font=font)
            # ax.set_title('Land use share of ' + et + f' in {curr_year}', font=font)

            ax_child = fig.add_axes((0.65, 0.19, 0.2, 0.2))
            re_gdf[et].plot(column=g_kind[gk], ax=ax_child, legend=False,
                            cax=cax, cmap=cmap,
                            vmin=0, vmax=0.8)
            ax_child = china_nine_lines.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', linewidth=0.3)
            ax_child = china.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', linewidth=0.2)
            ax_child = china_grids.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', alpha=0, linewidth=0.3)

            ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
            ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
            ax_child.set_xticks([])
            ax_child.set_yticks([])

            # Save results
            fig.savefig(os.path.join(out_output_processed_path, "graphs", et + '_' + 'IF' + '.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
    # save csv results
    pd.DataFrame.from_dict(pro_if["solar"], orient='index').to_csv(
        os.path.join(out_output_processed_path, f"graphs/solar_IF.csv"), header=False)
    pd.DataFrame.from_dict(pro_if["wind"], orient='index').to_csv(
        os.path.join(out_output_processed_path, f"graphs/wind_IF.csv"), header=False)


def averageStoreLengthVision(vreYear, resTag):
    res_dir = getResDir(vreYear, resTag)

    ST = ['phs', 'bat']

    with open(res_dir + dir_flag + 'store_length' + dir_flag + 'aveCL.pkl', 'rb+') as fin:
        aveCL = pickle.load(fin)
    fin.close()

    for st in ST:
        for pro in aveCL[st]:
            aveCL[st][pro] = np.array(aveCL[st][pro])
            fig = plt.figure()

            ax = fig.add_subplot()
            '''ax.hist(x=aveCL[st][pro][aveCL[st][pro]!=0],
                    bins = 40,
                    density = 1,
                    color = 'lightblue',
                    edgecolor = 'dimgray',
                    label = 'histogram')'''
            sns.kdeplot(aveCL[st][pro][aveCL[st][pro] != 0], color=sns.xkcd_rgb['light blue'], label='kernel density',
                        fill=True)

            font = {'family': 'Arial', 'weight': 'normal', 'size': 12}
            ax.set_xlabel('Length (hours)', font=font)
            ax.set_ylabel('Frequency', font=font)
            font = {'family': 'Arial', 'weight': 'bold', 'size': 12}
            ax.set_title(pro + ' (' + st.upper() + ')', font=font)

            font = {'family': 'Arial', 'weight': 'normal', 'size': 10}
            # ax.legend(prop=font,frameon=False)
            # plt.tight_layout()
            plt.savefig(res_dir + dir_flag + 'store_length' + dir_flag + st + dir_flag + pro + '_aveCL.png', dpi=600)
            plt.close()


def storageCap(vreYear, resTag):  # provincial
    res_dir = getResDir(vreYear, resTag) + dir_flag

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin']

    phs_cap = []
    bat_cap = []
    unit_trans = 1
    pro_name = []

    fst_bat_cap = open(res_dir + 'es_cap' + dir_flag + 'es_bat_cap.csv', 'r+')

    for line in fst_bat_cap:
        line = line.replace('\n', '')
        line = line.split(',')
        bat_cap.append(round(unit_trans * eval(line[1]), 2))
        # print(line[0],line[1])

        if line[0] == 'EastInnerMongolia':
            pro_name.append('Mengdong')
        elif line[0] == 'WestInnerMongolia':
            pro_name.append('Mengxi')
        else:
            pro_name.append(line[0])

    fst_bat_cap.close()

    fst_phs_cap = open(res_dir + 'es_cap' + dir_flag + 'es_phs_cap.csv', 'r+')
    for line in fst_phs_cap:
        line = line.replace('\n', '')
        line = line.split(',')
        phs_cap.append(round(unit_trans * eval(line[1]), 2))
    fst_phs_cap.close()

    x_aixs = np.arange(len(pro_name))

    width = 0.425
    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot()
    ax.bar(x_aixs, phs_cap, width, color='lightblue', label='PHS')
    ax.bar(x_aixs + width, bat_cap, width, color='lightgreen', label='BAT')
    ax.set_title('Storage Capacity', fontweight='bold')
    ax.set_xlabel('Region', fontweight='bold')
    plt.legend(loc='upper center', ncol=2)
    ax.set_ylabel('Capacity (GW)')
    plt.xticks(x_aixs + width / 2, pro_name, rotation=60, fontsize=12)

    for a, b in zip(x_aixs, phs_cap):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    for a, b in zip(x_aixs + width, bat_cap):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(bottom=0.2)

    plt.savefig(res_dir + dir_flag + 'storage_cap.png', dpi=600)
    plt.close()

    fsto_cap = open(res_dir + dir_flag + 'storage_cap.csv', 'w+')
    fsto_cap.write('%s,%s\n' % ('phs', sum(phs_cap)))
    fsto_cap.write('%s,%s\n' % ('bat', sum(bat_cap)))
    fsto_cap.close()


def StorageProfile(vreYear, resTag):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    provins = []
    f_provins = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_provins)
    for line in f_provins:
        line = line.replace('\n', '')
        line = line.split(',')
        provins.append(line[1])
    f_provins.close()

    fhour_seed = open(res_dir + 'hour_seed.csv', 'r+')
    hour_seed = []
    for line in fhour_seed:
        line = line.replace('\n', '')
        line = eval(line)
        hour_seed.append(line)
    fhour_seed.close()

    with open(res_dir + 'scen_params.pkl', 'rb+') as fin:
        scen_params = pickle.load(fin)
    fin.close()

    ST = ['phs', 'bat']

    if 'with_caes' in scen_params['storage']:
        if scen_params['storage']['with_caes']:
            ST.append('caes')

    if 'with_vrb' in scen_params['storage']:
        if scen_params['storage']['with_vrb']:
            ST.append('vrb')

    if 'with_lds' in scen_params['storage']:
        if scen_params['storage']['with_lds']:
            ST.append('caes')

    store_cap = {}

    for st in ST:
        store_cap[st] = {'nation': 0}

    for st in ST:
        fsc = open(res_dir + dir_flag + 'es_cap' + dir_flag + 'es_' + st + '_cap.csv', 'r+')
        for line in fsc:
            line = line.replace('\n', '')
            line = line.split(',')
            store_cap[st][line[0]] = eval(line[1])
            store_cap[st]['nation'] += eval(line[1])
        fsc.close()

    l2_char = {}
    l3_char = {}
    solar_char = {}
    wind_char = {}

    char = {}
    dischar = {}

    for st in ST:
        l2_char[st] = {}
        l3_char[st] = {}
        solar_char[st] = {}
        wind_char[st] = {}

        char[st] = {'nation': np.zeros(len(hour_seed))}
        dischar[st] = {'nation': np.zeros(len(hour_seed))}

    for pro in provins:
        for st in ST:
            if pro not in l2_char[st]:
                l2_char[st][pro] = []
                l3_char[st][pro] = []
                solar_char[st][pro] = []
                wind_char[st][pro] = []
                dischar[st][pro] = []

            f_l2_char = open(res_dir + dir_flag + 'es_char' + dir_flag + st + dir_flag + 'l2' + dir_flag + pro + '.csv',
                             'r+')
            for line in f_l2_char:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    l2_char[st][pro].append(line[1])
            f_l2_char.close()

            f_l3_char = open(res_dir + dir_flag + 'es_char' + dir_flag + st + dir_flag + 'l3' + dir_flag + pro + '.csv',
                             'r+')
            for line in f_l3_char:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    l3_char[st][pro].append(line[1])
            f_l3_char.close()

            f_s_char = open(
                res_dir + dir_flag + 'es_char' + dir_flag + st + dir_flag + 'solar' + dir_flag + pro + '.csv', 'r+')
            for line in f_s_char:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    solar_char[st][pro].append(line[1])
            f_s_char.close()

            f_w_char = open(
                res_dir + dir_flag + 'es_char' + dir_flag + st + dir_flag + 'wind' + dir_flag + pro + '.csv', 'r+')
            for line in f_w_char:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    wind_char[st][pro].append(line[1])
            f_w_char.close()

            f_c_dischar = open(res_dir + dir_flag + 'es_inte' + dir_flag + st + dir_flag + pro + '.csv', 'r+')
            for line in f_c_dischar:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    dischar[st][pro].append(line[1])
            f_c_dischar.close()

            l2_char[st][pro] = np.array(l2_char[st][pro])
            l3_char[st][pro] = np.array(l3_char[st][pro])
            solar_char[st][pro] = np.array(solar_char[st][pro])
            wind_char[st][pro] = np.array(wind_char[st][pro])
            dischar[st][pro] = np.array(dischar[st][pro])

            char[st][pro] = l2_char[st][pro] + l3_char[st][pro] + solar_char[st][pro] + wind_char[st][pro]

            char[st]['nation'] += char[st][pro]

            dischar[st]['nation'] += dischar[st][pro]

    for st in ST:
        for pro in provins:
            for h in range(len(char[st][pro])):
                if char[st][pro][h] > dischar[st][pro][h]:
                    char[st][pro][h] -= dischar[st][pro][h]
                    dischar[st][pro][h] = 0
                elif char[st][pro][h] < dischar[st][pro][h]:
                    dischar[st][pro][h] -= char[st][pro][h]
                    char[st][pro][h] = 0
                else:
                    char[st][pro][h] = 0
                    dischar[st][pro][h] = 0

        for h in range(len(hour_seed)):
            if char[st]['nation'][h] > dischar[st]['nation'][h]:
                char[st]['nation'][h] -= dischar[st]['nation'][h]
                dischar[st]['nation'][h] = 0
            elif char[st]['nation'][h] < dischar[st]['nation'][h]:
                dischar[st]['nation'][h] -= char[st]['nation'][h]
                char[st]['nation'][h] = 0
            else:
                char[st]['nation'][h] = 0
                dischar[st]['nation'][h] = 0

    '''resv_st = {
        'bat':{},
        'lds':{}
    }

    rt_effi = {'bat':0.95}

    for st in ['bat']:
        for pro in provins:
            if pro not in resv_st[st]:
                resv_st[st][pro] = np.zeros(8760)
            frsv_conv = open(res_dir+'resv_'+st+dir_flag+pro+'.csv','r+')
            for line in frsv_conv:
                line = line.replace('\n','')
                line = line.split(',')
                resv_st[st][pro][eval(line[0])] += rt_effi[st] * eval(line[1])
            frsv_conv.close()'''

    width = 1
    x_axis = range(len(hour_seed))

    for pro in ['nation']:
        for st in ST:
            folder = makeDir(res_dir + dir_flag + 'storage_fig' + dir_flag + st) + dir_flag
            if store_cap[st][pro] != 0:
                fig = plt.figure(figsize=(12.8, 4.8))
                ax = fig.add_subplot()
                ax.bar(x_axis, -char[st][pro], width, label='Charge')
                ax.bar(x_axis, dischar[st][pro], width, label='Discharge')

                ax.axhline(store_cap[st][pro], ls=':', c='black', linewidth=0.3, label='Capacity')
                ax.axhline(-store_cap[st][pro], ls=':', c='black', linewidth=0.3)

                font = {'family': 'Arial', 'weight': 'normal', 'size': 10}
                ax.legend(loc='upper center', ncol=3, prop=font, frameon=False)
                ax.set_xlabel('Time (hour)', font=font)
                ax.set_ylabel('Power output (GW)', font=font)

                font['size'] = 15
                ax.set_title('Storage profile (' + st.upper() + ')', font=font)

                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                # fig.tight_layout()
                plt.axhline(y=0, color='black', linewidth=0.5)

                plt.ylim(-1.2 * store_cap[st][pro], 1.2 * store_cap[st][pro])
                plt.xlim(-50, len(hour_seed) + 50)
                font['size'] = 10
                plt.xticks([0, len(hour_seed) / 2, len(hour_seed)], [1, 4380, 8760], font=font)

                plt.savefig(folder + pro + '_' + st + '_sto_profile.png', dpi=500)
                plt.close()

    '''for st in ['bat']:
        folder = makeDir(res_dir+dir_flag+'st_resv_fig'+dir_flag+st)+dir_flag
        for pro in provins:
            if store_cap[st][pro] != 0:
                fig = plt.figure(figsize=(12.8,4.8))
                ax = fig.add_subplot()
                ax.bar(x_axis, char[st][pro],width,label='Charge')
                ax.bar(x_axis, dischar[st][pro],width,bottom=char[st][pro],label='Discharge')
                ax.bar(x_axis, resv_st[st][pro],width,bottom=char[st][pro]+dischar[st][pro],label='Reserve')
                
                ax.axhline(store_cap[st][pro],ls=':',c='black',linewidth=0.3,label='Capacity')
            
                ax.legend(loc='upper center',fontsize = 7,ncol=4)
                ax.set_xlabel('Time (hour)')
                ax.set_ylabel('Power output (GW)')
                ax.set_title('Storage Profile (' +st+ ') of ' + pro, fontweight = "bold")
                
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                fig.tight_layout()
                plt.axhline(y=0,color='black',linewidth=0.5)
                
                plt.ylim(0,1.2*store_cap[st][pro])
                plt.xlim(-50,len(hour_seed)+50)
                plt.xticks([0,len(hour_seed)/2,len(hour_seed)],[1,4380,8760])
                
                plt.savefig(folder+pro+'_'+st+'_st_resv.png',dpi=600)
                plt.close()'''


def windsolarCapCurt(vreYear, resTag):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin']

    f_wind = open(res_dir + 'wind_info.csv', 'r+')
    f_solar = open(res_dir + 'solar_info.csv', 'r+')

    cap_pro = {}
    inted_cap_pro = {}

    wind_count = 0
    solar_count = 0
    off_count = 0

    for line in f_wind:
        line = line.replace('\n', '')
        line = line.split(',')
        wind_count += eval(line[1]) * eval(line[4])

        if line[8] not in cap_pro.keys():
            cap_pro[line[8]] = [0, 0]

        if line[8] not in inted_cap_pro:
            inted_cap_pro[line[8]] = [0, 0]

        cap_pro[line[8]][0] += eval(line[1]) * eval(line[4])

        inted_cap_pro[line[8]][0] += eval(line[10]) * eval(line[4])

        if eval(line[9]) == 0:
            off_count += eval(line[1]) * eval(line[4])
    f_wind.close()

    # print(off_count)

    for line in f_solar:
        line = line.replace('\n', '')
        line = line.split(',')
        solar_count += eval(line[1]) * eval(line[4])
        if line[8] not in cap_pro.keys():
            cap_pro[line[8]] = [0, 0]

        if line[8] not in inted_cap_pro:
            inted_cap_pro[line[8]] = [0, 0]

        cap_pro[line[8]][1] += eval(line[1]) * eval(line[4])

        inted_cap_pro[line[8]][1] += eval(line[10]) * eval(line[4])
    f_solar.close()

    # print(sum(solar_cap_pro))
    # total
    new_solar_cap_pro = []
    new_wind_cap_pro = []

    inted_solar_cap_pro = []
    inted_wind_cap_pro = []

    pro_name = []

    for pro in cap_pro:
        if pro == 'EastInnerMongolia':
            pro_name.append('Mengdong')
        elif pro == 'WestInnerMongolia':
            pro_name.append('Mengxi')
        else:
            pro_name.append(pro)
        new_wind_cap_pro.append(cap_pro[pro][0] - inted_cap_pro[pro][0])
        new_solar_cap_pro.append(cap_pro[pro][1] - inted_cap_pro[pro][1])

        inted_wind_cap_pro.append(inted_cap_pro[pro][0])
        inted_solar_cap_pro.append(inted_cap_pro[pro][1])

    new_wind_cap_pro.append(0)
    new_solar_cap_pro.append(0)
    new_wind_cap_pro = np.array(new_wind_cap_pro)
    new_solar_cap_pro = np.array(new_solar_cap_pro)

    inted_wind_cap_pro.append(0)
    inted_solar_cap_pro.append(0)
    inted_wind_cap_pro = np.array(inted_wind_cap_pro)
    inted_solar_cap_pro = np.array(inted_solar_cap_pro)

    pro_name.append('Nation')

    # curtailment
    with open(res_dir + 'curtailment' + dir_flag + 'solar' + dir_flag + 'curtailed_solar_split.pkl', 'rb+') as fin:
        solar_curt_info = pickle.load(fin)
    fin.close()

    with open(res_dir + 'curtailment' + dir_flag + 'wind' + dir_flag + 'curtailed_wind_split.pkl', 'rb+') as fin:
        wind_curt_info = pickle.load(fin)
    fin.close()

    # print(wind_curt_info)

    solar_curt = []
    wind_curt = []

    for p in cap_pro:
        solar_curt.append(solar_curt_info[p])

    for p in cap_pro:
        wind_curt.append(wind_curt_info[p])

    solar_curt.append(solar_curt_info['nation'])
    wind_curt.append(wind_curt_info['nation'])

    x = np.arange(len(provins) + 1)

    width = 0.425

    fig = plt.figure(figsize=(36, 12))

    ax = fig.add_subplot()
    ax.bar(x, inted_wind_cap_pro, width, color='lightblue', label='Inted Wind', hatch="///")
    ax.bar(x + width, inted_solar_cap_pro, width, color='lightgreen', label='Inted Solar', hatch="///")

    ax.bar(x, new_wind_cap_pro, width, bottom=inted_wind_cap_pro, color='lightblue', label='Total Wind')
    ax.bar(x + width, new_solar_cap_pro, width, bottom=inted_solar_cap_pro, color='lightgreen', label='Total Solar')

    ax.set_title('VRE Capacity and Curtailment Rate', fontweight='bold', fontsize=18)
    ax.set_xlabel('Region', fontweight='bold')
    ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.47, 1), edgecolor='white')

    ax.set_ylabel('Capacity (GW)', fontweight='bold', fontsize=12)
    plt.xticks(x + width / 2, pro_name, rotation=70, fontweight='bold', fontsize=12)

    total_cap = {
        'wind': new_wind_cap_pro + inted_wind_cap_pro,
        'solar': new_solar_cap_pro + inted_solar_cap_pro
    }

    for a, b in zip(x, total_cap['wind']):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x + width, total_cap['solar']):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=10)

    ax_1 = ax.twinx()
    ax_1.scatter(x, wind_curt, color='blue', label='Wind Curtailment', marker='^')
    ax_1.scatter(x + width, solar_curt, color='green', label='Solar Curtailment', marker='o')

    ax_1.set_ylabel('Curtailment rate (%)', fontweight='bold', fontsize=12)
    ax_1.legend(loc='upper center', bbox_to_anchor=(0.55, 1), edgecolor='white')

    plt.subplots_adjust(bottom=0.2)

    plt.savefig(res_dir + dir_flag + 'ws_cap_curt_by_province' + '.png', dpi=500)
    plt.close()


def totEnergyStore(vreYear, st, resTag):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin_py']

    fhour_seed = open(res_dir + 'hour_seed.csv', 'r+')
    hour_seed = []
    for line in fhour_seed:
        line = line.replace('\n', '')
        line = eval(line)
        hour_seed.append(line)

    tot_store_re = {}
    tot_store_conv = {}

    for pro in provins:
        f_ts_re = open(res_dir + 'es_tot' + dir_flag + st + dir_flag + 're' + dir_flag + pro + '.csv', 'r+')
        if pro not in tot_store_re.keys():
            tot_store_re[pro] = []
        for line in f_ts_re:
            line = line.replace('\n', '')
            line = eval(line)
            if line[0] in hour_seed:
                tot_store_re[pro].append(line[1])
        f_ts_re.close()
        f_ts_conv = open(res_dir + 'es_tot' + dir_flag + st + dir_flag + 'conv' + dir_flag + pro + '.csv', 'r+')
        if pro not in tot_store_conv.keys():
            tot_store_conv[pro] = []
        for line in f_ts_conv:
            line = line.replace('\n', '')
            line = eval(line)
            if line[0] in hour_seed:
                tot_store_conv[pro].append(line[1])
        f_ts_conv.close()

        tot_store_conv[pro] = np.array(tot_store_conv[pro])
        tot_store_re[pro] = np.array(tot_store_re[pro])

    x_axis = range(len(tot_store_re['Anhui']))
    width = 1
    for pro in provins:
        fig = plt.figure(figsize=(12.8, 4.8))
        ax = fig.add_subplot()

        ax.bar(x_axis, tot_store_conv[pro] + tot_store_re[pro], width, color='lightblue')

        ax.set_xlabel('Time (hour)')
        ax.set_ylabel('GWh')
        ax.set_title('Energy capacity (' + st + ') in ' + pro, fontweight="bold")
        fig.tight_layout()
        ax.axhline(max(tot_store_conv[pro] + tot_store_re[pro]), ls=':', c='black', linewidth=0.3,
                   label='Maximum Energy Capacity')

        plt.ylim(0, 1.2 * max(tot_store_conv[pro] + tot_store_re[pro]))

        plt.xlim(-50, len(tot_store_re['Anhui']) + 50)

        plt.legend(loc='upper center')

        plt.xticks([0, len(hour_seed) / 2, len(hour_seed)], [1, 4380, 8760])
        plt.savefig(res_dir + 'es_tot' + dir_flag + st + dir_flag + 'energy_capacity_' + st + '_' + pro + '.png',
                    dpi=500)
        plt.close()


def LoadProfile(vreYear, resTag, demCof):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    provins = []
    f_provins = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_provins)
    for line in f_provins:
        line = line.replace('\n', '')
        line = line.split(',')
        provins.append(line[1])
    f_provins.close()

    fhour_seed = open(res_dir + 'hour_seed.csv', 'r+')
    hour_seed = []
    for line in fhour_seed:
        line = line.replace('\n', '')
        line = eval(line)
        hour_seed.append(line)
    fhour_seed.close()

    with open(res_dir + 'scen_params.pkl', 'rb+') as fin:
        scen_params = pickle.load(fin)
    fin.close()

    with open(work_dir + 'data_pkl' + dir_flag + 'province_demand_full_2060.pkl', 'rb+') as fin:
        demand = pickle.load(fin)
    fin.close()

    demand['Nation'] = np.zeros(8760)

    for pro in provins:
        demand[pro] = np.array(demand[pro])
        demand['Nation'] += demCof * demand[pro]

    with open(res_dir + 'load.pkl', 'rb') as fin:
        load = pickle.load(fin)
    fin.close()

    discharge_tot = load['discharge']['Nationwide']

    charge_tot = load['charge_tot']

    for h in range(len(charge_tot)):
        if charge_tot[h] >= discharge_tot[h]:
            es_to_hydro = discharge_tot[h] * load['charge_l2']['Nationwide'][h] / charge_tot[h]
            es_to_nuclear = discharge_tot[h] * load['charge_l3']['Nationwide'][h] / charge_tot[h]
            es_to_wind = discharge_tot[h] * load['charge_wind']['Nationwide'][h] / charge_tot[h]
            es_to_solar = discharge_tot[h] * load['charge_solar']['Nationwide'][h] / charge_tot[h]

            load['nuclear']['Nationwide'][h] += es_to_nuclear
            load['hydro']['Nationwide'][h] += es_to_hydro
            load['wind']['Nationwide'][h] += es_to_wind
            load['solar']['Nationwide'][h] += es_to_solar

            charge_tot[h] -= discharge_tot[h]
            discharge_tot[h] = 0
        else:
            discharge_tot[h] -= charge_tot[h]

            charge_tot[h] = 0

            load['bio']['Nationwide'][h] += load['charge_l3']['Nationwide'][h]
            load['hydro']['Nationwide'][h] += load['charge_l2']['Nationwide'][h]
            load['wind']['Nationwide'][h] += load['charge_wind']['Nationwide'][h]
            load['solar']['Nationwide'][h] += load['charge_solar']['Nationwide'][h]

    with open(res_dir + 'curtailment' + dir_flag + 'wind' + dir_flag + 'curtailed_wind.pkl', 'rb+') as fin:
        curtailed_wind = pickle.load(fin)
    fin.close()

    with open(res_dir + 'curtailment' + dir_flag + 'solar' + dir_flag + 'curtailed_solar.pkl', 'rb+') as fin:
        curtailed_solar = pickle.load(fin)
    fin.close()

    curtailment = np.zeros(8760)

    for pro in curtailed_wind['power_curt']:
        for h in range(len(curtailed_wind['power_curt'][pro])):
            curtailment[h] += curtailed_wind['power_curt'][pro][h]
            curtailment[h] += curtailed_solar['power_curt'][pro][h]

    load_nw = [load['nuclear']['Nationwide'] + load['trans_out']['Nationwide']['l3'],
               load['coal']['Nationwide'],
               load['hydro']['Nationwide'] + load['trans_out']['Nationwide']['l2'],
               load['bio']['Nationwide'],
               load['gas']['Nationwide'],
               load['wind']['Nationwide'] + load['trans_out']['Nationwide']['wind'],
               load['solar']['Nationwide'] + load['trans_out']['Nationwide']['solar'],
               discharge_tot,
               charge_tot,
               curtailment]

    x_axis = range(len(hour_seed))

    # nationwide

    load_color = ['red', 'very dark brown', 'sky blue', 'snot green', 'mint', 'bright blue', 'mango', 'greyish',
                  'white', 'white']

    load_label = ['nuclear', 'coal', 'hydro', 'bio', 'gas', 'wind', 'solar', 'storage discharge', 'storage charge', '']
    load_hatch = ['', '', '', '', '', '', '', '', '/////', '']

    hour_start = [0, 1800, 5400, 7128]
    season = {0: 'Winter', 1800: 'Spring', 5400: 'Summer', 7128: 'Autumn'}

    # hour_start = [750]
    # season = {750:'Winter'}

    draw_len = 168

    for h in hour_start:
        y_low = np.zeros(len(hour_seed))
        fig = plt.figure(figsize=(12.8, 4.6))
        ax = fig.add_subplot()

        for i in range(len(load_nw)):
            y_up = y_low + load_nw[i]

            if i == 5 or i == 6 or i == 7:
                for t in range(h, h + draw_len):
                    if y_up[t] > demand['Nation'][t]:
                        y_up[t] = demand['Nation'][t]

            plt.fill_between(x_axis[:draw_len], y_low[h:h + draw_len], y_up[h:h + draw_len], hatch=load_hatch[i],
                             facecolor=sns.xkcd_rgb[load_color[i]], label=load_label[i])

            y_low = y_up

            if i == 8:
                plt.plot(x_axis[:draw_len], y_up[h:h + draw_len], color='grey', linewidth=0.3, linestyle='--')
            if i == 9:
                plt.plot(x_axis[:draw_len], y_up[h:h + draw_len], color='grey', linewidth=0.5, linestyle='-.',
                         label='VRE curtailment')

        plt.plot(x_axis[:draw_len], demand['Nation'][h:h + draw_len], color='black', label='demand')

        font = {'family': 'Arial', 'weight': 'normal', 'size': 10}
        ax.set_xlabel('Time (hour)', font=font)
        ax.set_ylabel('Load (GW)', font=font)
        plt.ylim(0, 5500)
        plt.xlim(0, draw_len - 1)
        # plt.xticks([0,len(hour_seed)],[1,8760])

        font = {'family': 'Arial', 'weight': 'normal', 'size': 15}
        ax.set_title(season[h], font=font)

        font = {'family': 'Arial', 'weight': 'normal', 'size': 10}
        plt.legend(ncol=6, loc='upper center', prop=font, frameon=False)

        plt.xticks([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 167],
                   [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168])
        fig.tight_layout()
        folder = makeDir(res_dir + 'load_fig') + dir_flag
        plt.savefig(folder + 'Nationwide' + 'in' + season[h] + '.png', dpi=600)
        plt.close()


def lcoeNation(wind_year, solar_year, inte_target, nuclearBeta, hydroBeta, gasBeta, bioBeta):
    res_dir = getResDir(wind_year, solar_year, inte_target) + dir_flag

    hour_seed = []
    f_hour_seed = open(work_dir + 'data_csv' + dir_flag + 'simulation_meta/hour_seed.csv', 'r')
    for line in f_hour_seed:
        line = line.replace('\n', '')
        line = eval(line)
        hour_seed.append(line)
    f_hour_seed.close()

    with open(work_dir + 'data_pkl' + dir_flag + 'hour_pre.pkl', 'rb') as fin:
        hour_pre = pickle.load(fin)
    fin.close()

    # generation cost of integrated wind and solar
    gen_cost_ws = 0

    with open(work_dir + 'data_pkl' + dir_flag + 'wind_cell_2016.pkl', 'rb') as fin:
        wind_cell = pickle.load(fin)
    fin.close()

    with open(work_dir + 'data_pkl' + dir_flag + 'solar_cell_2015.pkl', 'rb') as fin:
        solar_cell = pickle.load(fin)
    fin.close()

    wind_cell = wind_cell['provin_cf_sort']
    solar_cell = solar_cell['provin_cf_sort']

    for pro in wind_cell:
        for c in wind_cell[pro]:
            gen_cost_ws += c[4] * c[5] * c[13]

    for pro in solar_cell:
        for c in solar_cell[pro]:
            gen_cost_ws += c[4] * c[5] * c[13]

    scale = 0.001
    files = ['nuclear.csv', 'coal.csv', 'hydro.csv', 'beccs.csv', 'gas.csv', 'ccs.csv']
    provin_conv = {}
    nation_conv = [0] * 6
    for file in files:
        fr = open(work_dir + 'data_csv' + dir_flag + file, 'r+')
        for line in fr:
            line = line.replace('\n', '')
            line = line.split(',')
            if line[0] not in provin_conv.keys():
                provin_conv[line[0]] = []

            if file == 'beccs.csv':
                scale = 1
            else:
                scale = 0.001
            provin_conv[line[0]].append(scale * eval(line[1]))
        fr.close()

    for pro in provin_conv:
        if len(provin_conv[pro]) == 5:
            provin_conv[pro].append(0)
        for i in range(len(files)):
            nation_conv[i] += provin_conv[pro][i]

    lcoe_nuclear = 0.14
    lcoe_hydro = 0
    lcoe_gas = 0.23

    lcoe_coalccs = 0.58  # yuan/kwh
    lcoe_beccs = 0.43

    with open(res_dir + 'load.pkl', 'rb') as fin:
        load = pickle.load(fin)
    fin.close()

    gen_cost_nuclear = lcoe_nuclear * sum(load['nuclear']['Nationwide'])

    gen_cost_hydro = lcoe_hydro * (sum(load['l2']['Nationwide']) + sum(load['charge_l2']['Nationwide']))

    gen_cost_bio = lcoe_beccs * (sum(load['l3']['Nationwide']) + sum(load['charge_l3']['Nationwide']))

    gen_cost_gas = lcoe_gas * sum(load['l4']['Nationwide'])

    gen_cost_coalccs = (lcoe_coalccs) * sum(load['coal_ccs'])

    gen_cost_beccs = (lcoe_beccs) * sum(load['bio']['Nationwide'])

    nuclear_ramp = getRamp(load['nuclear']['Nationwide'], hour_seed, hour_pre)

    coal_ramp = getRamp(load['coal_ccs'], hour_seed, hour_pre)

    nuclear_ramp_cost = 0.2 * nuclear_ramp[0]

    coal_ramp_cost = 0.05 * coal_ramp[0]

    ramp_cost = coal_ramp_cost + nuclear_ramp_cost

    gen_cost = gen_cost_ws + gen_cost_nuclear + gen_cost_hydro + gen_cost_bio + gen_cost_gas + gen_cost_beccs + gen_cost_coalccs

    fixed_cost = nuclearBeta * nation_conv[0] * (0.629 + 15000 * getCRF(4.2, 50))
    fixed_cost += hydroBeta * nation_conv[2] * (0.203 + 9900 * getCRF(4.2, 50))
    fixed_cost += bioBeta * nation_conv[3] * (0.712 + 9700 * getCRF(4.2, 30))
    fixed_cost += gasBeta * nation_conv[4] * (0.224 + 7300 * getCRF(4.2, 30))
    fixed_cost += nation_conv[5] * (0.519 + 12400 * getCRF(4.2, 20))

    f_gen_cost = open(res_dir + 'gen_cost.csv', 'w+')
    f_gen_cost.write('%s\n' % (gen_cost * 1000000))
    f_gen_cost.write('%s\n' % (fixed_cost * 1000000))
    f_gen_cost.write('%s\n' % (ramp_cost * 1000000))
    f_gen_cost.write('%s\n' % (gen_cost * 1000000 + fixed_cost * 1000000 + ramp_cost * 1000000))
    f_gen_cost.close()


def DemandProfile():
    f_dem = open(work_dir + 'data_csv' + dir_flag + 'demand_assumptions/province_demand_by_hour_2060' +
                 dir_flag + 'nation_dem_full.csv',
                 'r+')

    nation_dem = []

    for line in f_dem:
        line = line.replace('\n', '')
        line = eval(line)
        nation_dem.append(line[1])

    f_dem.close()

    x = range(8760)

    fig = plt.figure(figsize=(32, 8))

    ax = fig.add_subplot()

    ax.plot(x, nation_dem, color='green')

    ax.set_title('Demand', fontweight='bold')
    ax.set_ylabel('Load (GW)')
    ax.set_xlabel('Time (hour)')

    plt.savefig(work_dir + 'data_fig' + dir_flag + 'nation_demand.png', dpi=600)
    plt.close()


def VREValueDistribution(vreYear, resTag):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    wind_year = VreYearSplit(vreYear)[0]
    solar_year = VreYearSplit(vreYear)[1]

    with open(res_dir + 'wind_cell_value_' + wind_year + '.pkl', 'rb+') as fin:
        wind_value = pickle.load(fin)
    fin.close()

    with open(res_dir + 'solar_cell_value_' + solar_year + '.pkl', 'rb+') as fin:
        solar_value = pickle.load(fin)
    fin.close()

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china.shp')
    china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')

    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    wind_shp = {'value': [], 'polygon': []}

    solar_shp = {'value': [], 'polygon': []}

    bound = getBound()

    for pro in wind_value:
        for cell in range(len(wind_value[pro])):
            x = wind_value[pro][cell][0]
            y = wind_value[pro][cell][1]

            wind_shp['polygon'].append(
                shapely.geometry.Polygon(
                    [(x - 0.15625, y - 0.125),
                     (x + 0.15625, y - 0.125),
                     (x + 0.15625, y + 0.125),
                     (x - 0.15625, y + 0.125)]
                )
            )

            wind_shp['value'].append(wind_value[pro][cell][2])

    for pro in solar_value:
        for cell in range(len(solar_value[pro])):
            x = solar_value[pro][cell][0]
            y = solar_value[pro][cell][1]

            solar_shp['polygon'].append(
                shapely.geometry.Polygon(
                    [(x - 0.15625, y - 0.125),
                     (x + 0.15625, y - 0.125),
                     (x + 0.15625, y + 0.125),
                     (x - 0.15625, y + 0.125)]
                )
            )

            solar_shp['value'].append(solar_value[pro][cell][2])

    wind_gdf = gpd.GeoDataFrame(wind_shp, geometry=wind_shp['polygon'])
    solar_gdf = gpd.GeoDataFrame(solar_shp, geometry=solar_shp['polygon'])

    re_gdf = {'wind': wind_gdf, 'solar': solar_gdf}

    folder = makeDir(res_dir + 'vre_value_distri')

    cmap = 'autumn_r'

    for et in re_gdf:
        fig = plt.figure(figsize=(12.6, 8.4))
        ax = plt.gca()

        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', size="4%", pad=0.05)

        re_gdf[et].plot(column='value', ax=ax, legend=True, legend_kwds={'label': 'Value (revenue / cost)'}, cax=cax,
                        cmap=cmap)

        ax = china_nine_lines.geometry.plot(ax=ax, color='black')

        ax = china.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=0.25, linewidth=1.2)
        # gplt.polyplot(china,ax=ax)

        ax = china_grids.geometry.plot(ax=ax, color='white', edgecolor='white', alpha=0.0)

        ax.axis('off')
        ax.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
        ax.set_ylim(bound.geometry[0].y, bound.geometry[1].y)
        ax.set_title('Value of ' + et, fontweight='bold')

        ax_child = fig.add_axes((0.65, 0.19, 0.2, 0.2))
        re_gdf[et].plot(column='value', ax=ax_child, legend=False, cmap=cmap)
        ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black')

        ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.1)
        ax_child = china_grids.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0)

        ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
        ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
        ax_child.set_xticks([])
        ax_child.set_yticks([])

        fig.savefig(folder + dir_flag + et + '_value_distri.png', dpi=600)
        plt.close()


def GeneratorDistribution(vreYear, resTag):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    with open(work_dir + 'data_pkl' + dir_flag + 'province_demand_full_2060.pkl', 'rb+') as fin:
        demand = pickle.load(fin)
    fin.close()

    with open(work_dir + 'data_pkl' + dir_flag + 'conv_cap.pkl', 'rb+') as fin:
        conv_cap = pickle.load(fin)
    fin.close()

    demand_load_max = {}

    for pro in conv_cap:
        max_load = max(demand[pro])
        demand_load_max[pro] = max_load

    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins_name['provin']

    f_wind = open(res_dir + 'wind_info.csv', 'r+')
    f_solar = open(res_dir + 'solar_info.csv', 'r+')

    cap_pro = {}

    for line in f_wind:
        line = line.replace('\n', '')
        line = line.split(',')
        if line[8] not in cap_pro.keys():
            cap_pro[line[8]] = [0, 0]

        cap_pro[line[8]][0] += eval(line[1]) * eval(line[4])

    f_wind.close()

    for line in f_solar:
        line = line.replace('\n', '')
        line = line.split(',')
        if line[8] not in cap_pro.keys():
            cap_pro[line[8]] = [0, 0]

        cap_pro[line[8]][1] += eval(line[1]) * eval(line[4])

    f_solar.close()

    phs_cap = {}
    bat_cap = {}
    unit_trans = 1

    fst_bat_cap = open(res_dir + 'es_cap' + dir_flag + 'es_bat_cap.csv', 'r+')
    for line in fst_bat_cap:
        line = line.replace('\n', '')
        line = line.split(',')
        bat_cap[line[0]] = round(unit_trans * eval(line[1]), 2)

    fst_bat_cap.close()

    fst_phs_cap = open(res_dir + 'es_cap' + dir_flag + 'es_phs_cap.csv', 'r+')
    for line in fst_phs_cap:
        line = line.replace('\n', '')
        line = line.split(',')
        phs_cap[line[0]] = round(unit_trans * eval(line[1]), 2)
    fst_phs_cap.close()

    generator_pro = {'coal': [], 'nuclear': [], 'hydro': [], 'bio': [], 'gas': [], 'wind': [], 'solar': []}

    store_cap = {'BAT': [], 'PHS': []}

    pro_index = {}
    index = 0

    generator_pro['tot'] = np.zeros((32, 12))

    for pro in conv_cap:
        pro_index[index] = pro
        generator_pro['tot'][index][0] = conv_cap[pro]['coal']
        generator_pro['tot'][index][1] = conv_cap[pro]['nuclear']
        generator_pro['tot'][index][2] = conv_cap[pro]['hydro']
        generator_pro['tot'][index][3] = conv_cap[pro]['beccs']
        generator_pro['tot'][index][4] = conv_cap[pro]['gas']
        generator_pro['tot'][index][5] = cap_pro[pro][0]
        generator_pro['tot'][index][6] = cap_pro[pro][1]
        generator_pro['tot'][index][7] = phs_cap[pro]
        generator_pro['tot'][index][8] = bat_cap[pro]
        generator_pro['tot'][index][9] = index
        generator_pro['tot'][index][10] = sum(generator_pro['tot'][index][:7])
        generator_pro['tot'][index][11] = demand_load_max[pro]
        index += 1

    generator_pro['tot'] = generator_pro['tot'][np.argsort(-generator_pro['tot'][:, 10])]

    generator_pro['coal'] = generator_pro['tot'][:, 0]
    generator_pro['nuclear'] = generator_pro['tot'][:, 1]
    generator_pro['hydro'] = generator_pro['tot'][:, 2]
    generator_pro['bio'] = generator_pro['tot'][:, 3]
    generator_pro['gas'] = generator_pro['tot'][:, 4]
    generator_pro['wind'] = generator_pro['tot'][:, 5]
    generator_pro['solar'] = generator_pro['tot'][:, 6]
    store_cap['PHS'] = generator_pro['tot'][:, 7]
    store_cap['BAT'] = generator_pro['tot'][:, 8]
    peak_load = generator_pro['tot'][:, 11]

    provin_name = []

    for index in generator_pro['tot'][:, 9]:
        provin_name.append(pro_index[index])

    colors = {
        'coal': 'very dark brown',
        'nuclear': 'red',
        'hydro': 'sky blue',
        'bio': 'snot green',
        'gas': 'mint',
        'wind': 'bright blue',
        'solar': 'mango',
        'PHS': 'purple blue',
        'BAT': 'silver'}

    folder = makeDir(res_dir + 'generator_distribution' + dir_flag)

    label_keys = {
        'coal': 'Coal-CCS',
        'nuclear': 'Nuclear',
        'hydro': 'Hydro',
        'bio': 'BECCS',
        'gas': 'Gas-CCS',
        'wind': 'Wind',
        'solar': 'Solar'
    }

    for skip in range(2):

        x = range(int(len(conv_cap) / 2))

        x = np.array(x) * 2

        width = 1
        fig = plt.figure(figsize=(28, 8))
        ax = fig.add_subplot()
        # ax.scatter(x,demand_load_max,marker='*',color='black')

        bottom_value = np.zeros(int(len(conv_cap) / 2))

        for power in ['coal', 'nuclear', 'hydro', 'bio', 'gas', 'wind', 'solar']:
            ax.bar(x, generator_pro[power][skip * 16:(skip + 1) * 16], width, bottom=bottom_value,
                   label=label_keys[power], color=sns.xkcd_rgb[colors[power]], edgecolor='gray')

            bottom_value += generator_pro[power][skip * 16:(skip + 1) * 16]

        ax.scatter(x, peak_load[skip * 16:(skip + 1) * 16], marker='^', color='black', zorder=2, label='Peak load')

        plt.bar(x + 0.75, store_cap['PHS'][skip * 16:(skip + 1) * 16], width=0.5, color=sns.xkcd_rgb[colors['PHS']],
                label='PHS', edgecolor='gray')
        plt.bar(x + 0.75, store_cap['BAT'][skip * 16:(skip + 1) * 16], width=0.5,
                bottom=store_cap['PHS'][skip * 16:(skip + 1) * 16], color=sns.xkcd_rgb[colors['BAT']], label='BAT',
                edgecolor='gray')

        # ax.set_xlabel('Region', fontweight='bold')
        font = {'family': 'Arial', 'weight': 'normal', 'size': 18}
        ax.set_ylabel('GW', font=font)
        # ax.set_title('Installation assumptions of conventional power in 2060',fontweight='bold',fontsize=15)
        fig.tight_layout()

        font = {'family': 'Arial', 'weight': 'normal', 'size': 15}

        for a, b in zip(x, bottom_value):
            plt.text(a, b, '%.0f' % b, ha='center', va='bottom', font=font)
        # ax.set_yticks([])
        # plt.ylim(0,2000)

        plt.subplots_adjust(bottom=0.3)

        if skip == 1:
            font = {'family': 'Arial', 'weight': 'normal', 'size': 15}
            plt.legend(ncol=2, prop=font)

        plt.ylim(0, 550)

        font = {'family': 'Arial', 'weight': 'bold', 'size': 18}
        plt.xticks(x, provin_name[skip * 16:(skip + 1) * 16], rotation=90, font=font)

        # ax_1 = ax.twinx()

        # for a,b in zip(x, demand_load_max):
        #    plt.text(a, b,'%.1f' % b, ha='center',va='bottom',fontsize=8)

        plt.savefig(folder + str(skip) + '.png', dpi=400)

        plt.close()

    res_fig = np.zeros((3200 * 2, 11200, 4))

    for i in range(2):
        col = i // 1
        # row = i % 2

        fig = sio.imread(res_dir + 'generator_distribution' + dir_flag + str(i) + '.png')

        res_fig[col * 3200:(col + 1) * 3200, :, :] = fig.copy()

    res_fig = np.array(res_fig, dtype=np.uint8)

    sio.imsave(res_dir + 'generator_distribution' + dir_flag + 'gen_distri.png', res_fig)


def loadSheddingByProvince(vreYear, resTag):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    with open(res_dir + 'load_shedding' + dir_flag + 'by_province.pkl', 'rb+') as fin:
        load_shedding = pickle.load(fin)
    fin.close()

    load_shedding = sorted(load_shedding.items(), key=lambda item: item[1], reverse=True)

    provinces = []
    values = []

    for i in load_shedding:
        if round(i[1], 0) > 0:
            provinces.append(i[0])
            values.append(i[1])

    x = range(len(provinces))

    fig = plt.figure(figsize=(12.8, 4.8))
    ax = fig.add_subplot()

    width = 0.6
    ax.bar(x, values, width=width, color='darkgreen')
    # ax.set_title('Generation in 2060',fontweight='bold')
    ax.set_ylabel('Shadded load (GW)')
    ax.set_xlabel('Province')
    plt.xticks(x, provinces, rotation=90)

    for a, b in zip(x, values):
        plt.text(a, b, '%.f' % b, ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(bottom=0.3)

    plt.savefig(res_dir + 'tot_load_shedding.png', dpi=600)
    plt.close()


def TotalEnergyToHourVisual(vreYear, resTag):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    with open(res_dir + 'es_hour.pkl', 'rb+') as fin:
        es_hour = pickle.load(fin)
    fin.close()

    x = range(len(es_hour['bat']['Anhui']))
    width = 1
    for st in es_hour:
        folder = makeDir(res_dir + 'es_hour_fig' + dir_flag + st) + dir_flag
        for pro in es_hour[st]:
            fig = plt.figure(figsize=(12.8, 4.8))
            ax = fig.add_subplot()
            ax.bar(x, es_hour[st][pro], width, color='pink')

            ax.set_xlabel('Time (hour)')
            ax.set_ylabel('Energy in ' + st.upper() + ' storage\n (hours of mean demand)')
            ax.set_title('Energy (' + st.upper() + ') in ' + pro, fontweight="bold")
            fig.tight_layout()
            # ax.axhline(max(tot_store_conv[pro]+tot_store_re[pro]),ls=':',c='black',linewidth=0.3,label='Maximum Energy Capacity')
            if max(es_hour[st][pro]) != 0:
                plt.ylim(0, 1.2 * max(es_hour[st][pro]))
            plt.xlim(-50, len(x) + 50)
            # plt.legend(loc='upper center')
            plt.xticks([0, len(x) / 2, len(x)], [1, 4380, 8760])
            plt.savefig(folder + pro + '.png', dpi=600)
            plt.close()


def TransDirect(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    trans = {}
    trans_load = {}

    f_tl = open(os.path.join(out_output_processed_path, 'trans_matrix', 'trans_matrix_tot.csv'))
    for line in f_tl:
        line = line.replace('\n', '')
        line = line.split(',')
        trans[line[0]] = line
    f_tl.close()

    for i in trans:
        if i != 'TWh':
            for k in range(1, len(trans[i]) - 1):
                trans_load[(i, trans['TWh'][k])] = round(eval(trans[i][k]), 2)

    province_capital = {}

    f_pg = open(work_dir + 'data_csv' + dir_flag + 'geography/province_capital.csv', 'r+', encoding='utf8')

    next(f_pg)

    for line in f_pg:
        line = line.replace('\n', '')
        line = line.split(',')
        province_capital[line[0]] = line[2]

    f_pg.close()

    trans_direct = []
    city_scatter = []
    load_in_fig = {}
    for pro_pair in trans_load:
        load = trans_load[pro_pair]

        if load > 0:
            trans_direct.append((province_capital[pro_pair[0]], province_capital[pro_pair[1]]))
            city_scatter.append((province_capital[pro_pair[0]], 0.2))
            city_scatter.append((province_capital[pro_pair[1]], 0.2))
            load_in_fig[(province_capital[pro_pair[0]], province_capital[pro_pair[1]])] = (load)
    # print(trans_direct)

    direction = Geo()

    direction.add_schema(
        maptype="china",
        itemstyle_opts=opts.ItemStyleOpts(color="white", border_color="black"),
    )

    load_for_count = {
        (0, 50): [],
        (50, 100): [],
        (100, 150): [],
        (150, 200): [],
        (200, 250): [],
        (250, 500): []
    }

    for i in load_in_fig.values():
        if i < 50:
            load_for_count[(0, 50)].append(i)
        elif i < 100:
            load_for_count[(50, 100)].append(i)
        elif i < 150:
            load_for_count[(100, 150)].append(i)
        elif i < 200:
            load_for_count[(150, 200)].append(i)
        elif i < 250:
            load_for_count[(200, 250)].append(i)
        else:
            load_for_count[(250, 500)].append(i)

    # print(load_for_count)

    load_in_legend = {}

    for i in range(len(load_for_count.keys())):
        key = list(load_for_count.keys())[i]
        start = str(i)
        end = str(i) + '1'

        direction.add_coordinate(
            name=start,
            longitude=80,
            latitude=26 - 1.5 * i
        )

        direction.add_coordinate(
            name=end,
            longitude=88,
            latitude=26 - 1.5 * i
        )
        try:
            load_in_legend[(start, end)] = sum(load_for_count[key]) / len(load_for_count[key])
        except:
            load_in_legend[(start, end)] = 180

    for pair in trans_direct:
        if load_in_fig[pair] > 250:
            symbol_size_ = 12
        elif load_in_fig[pair] > 200:
            symbol_size_ = 10
        else:
            symbol_size_ = 0.05 * load_in_fig[pair]

        if pair[1] == '拉萨':
            symbol_size_ = 5

        if pair[0] == '拉萨':
            symbol_size_ = 5

        direction.add(
            '',
            [pair],
            type_=ChartType.LINES,
            symbol_size=symbol_size_,
            effect_opts=opts.EffectOpts(
                symbol=SymbolType.ARROW, symbol_size=0, color="firebrick", trail_length=0
            ),
            linestyle_opts=opts.LineStyleOpts(curve=0.2, color='black', width=0.125 + 0.01 * load_in_fig[pair]),
        )

    # print(load_in_legend)

    for pair in load_in_legend:
        if load_in_legend[pair] > 250:
            symbol_size_ = 12
        elif load_in_legend[pair] > 200:
            symbol_size_ = 10
        else:
            symbol_size_ = 0.05 * load_in_legend[pair]

        if load_in_legend[pair] < 60:
            symbol_size_ = 3

        direction.add(
            '',
            [pair],
            type_=ChartType.LINES,
            symbol_size=symbol_size_,
            effect_opts=opts.EffectOpts(
                symbol=SymbolType.ARROW, symbol_size=0, color="red", trail_length=0
            ),
            linestyle_opts=opts.LineStyleOpts(curve=0, color='black', width=0.125 + 0.01 * load_in_legend[pair]),
        )

    direction.add(
        '',
        city_scatter,
        type_=ChartType.EFFECT_SCATTER,
        color='white',
        symbol_size=3,
        effect_opts=opts.EffectOpts(
            symbol_size=2
        ),
    )
    direction.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    direction.set_global_opts(

        title_opts=opts.TitleOpts(title="Direction of Net Transmission Line in 2060", pos_left='center'))
    direction.render(os.path.join(out_output_processed_path, "graphs", "trans_direct.html"))


def transInfo30To60(windyear, tag30, tag60):
    trans30 = {}

    trans60 = {}

    diff_60_30 = {}

    dir_30 = work_dir + 'LinearOpt2030' + dir_flag + 'data_res' + dir_flag + dir_flag + windyear + dir_flag + 'res' + tag30 + dir_flag

    dir_60 = work_dir + 'LinearOpt2060' + dir_flag + 'data_res' + dir_flag + dir_flag + windyear + dir_flag + 'res' + tag60 + dir_flag

    transInfo30 = open(dir_30 + 'load_trans' + dir_flag + 'trans_matrix_tot.csv', 'r+')
    for line in transInfo30:
        line = line.replace('\n', '')
        line = line.split(',')
        trans30[line[0]] = line

    transInfo30.close()

    transInfo60 = open(dir_60 + 'load_trans' + dir_flag + 'trans_matrix_tot.csv', 'r+')
    for line in transInfo60:
        line = line.replace('\n', '')
        line = line.split(',')
        trans60[line[0]] = line
    transInfo60.close()

    folder = makeDir(
        work_dir + 'data_res' + dir_flag + 'res' + tag30 + 'To' + tag60 + dir_flag + 'load_trans') + dir_flag

    transInfo30To60 = open(folder + 'trans_matrix_tot30To60.csv', 'w+')

    for i in trans30:
        if i != 'TWh':
            for k in range(1, len(trans30[i]) - 1):
                t_30 = int(eval(trans30[i][k]))
                t_60 = int(eval(trans60[i][k]))
                diff_60_30[(i, trans30['TWh'][k])] = t_60 - t_30

    diff_60_30 = sorted(diff_60_30.items(), key=lambda item: item[1], reverse=True)

    # print((diff_60_30[:10]))
    for i in trans30:
        if i == 'TWh':
            transInfo30To60.write('%s\n' % (','.join(trans30[i])))
        else:
            transInfo30To60.write(i + ',')
            for k in range(1, len(trans30[i]) - 1):
                t_30 = int(eval(trans30[i][k]))
                t_60 = int(eval(trans60[i][k]))
                if ((i, trans30['TWh'][k]), t_60 - t_30) in diff_60_30[:10]:
                    transInfo30To60.write('!' + str(t_30) + '->' + str(t_60) + ',')
                else:
                    transInfo30To60.write(str(t_30) + '->' + str(t_60) + ',')
            transInfo30To60.write('\n')

    transInfo30To60.close()


def TransCapMap(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    newTransCap = {}

    f_ntc = open(os.path.join(out_output_path, 'new_trans_cap', 'cap_trans_new.csv'), 'r+')

    for line in f_ntc:
        line = line.replace('\n', '')
        line = line.split(',')
        line[2] = eval(line[2])
        newTransCap[(line[0], line[1])] = round(line[2], 2)

    f_ntc.close()

    existingTransCap = {}
    f_etc = pd.read_csv(os.path.join(out_input_path, 'inter_pro_trans.csv'))

    for pro in f_etc['province']:
        for i in range(len(f_etc[pro])):
            existingTransCap[(pro, f_etc['province'][i])] = 0.5 * 0.001 * f_etc[pro][i]  # MW -> GW

    f_etc = None

    totalTransCap = {}

    for pro_pair in newTransCap:
        totalTransCap[pro_pair] = newTransCap[pro_pair] + existingTransCap[pro_pair]

    provin_geo = {}

    f_pg = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')

    next(f_pg)

    for line in f_pg:
        line = line.replace('\n', '')
        line = line.split(',')
        provin_geo[line[1]] = (eval(line[3]), eval(line[4]))

    f_pg.close()

    new_trans_shp = {'cap': [], 'linestring': []}

    count = []

    for pro_pair in newTransCap:
        if pro_pair not in count:
            count.append((pro_pair[1], pro_pair[0]))
            cap = newTransCap[pro_pair] + newTransCap[(pro_pair[1], pro_pair[0])]
            if cap != 0:
                new_trans_shp['cap'].append(cap)
                new_trans_shp['linestring'].append(
                    shapely.geometry.LineString(
                        [(provin_geo[pro_pair[0]][0], provin_geo[pro_pair[0]][1]),
                         (provin_geo[pro_pair[1]][0], provin_geo[pro_pair[1]][1])]
                    )
                )
    # Add 40 GW as a placeholder for plotting
    new_trans_shp['cap'].append(40)
    new_trans_shp['linestring'].append(shapely.geometry.LineString([(0, 0), (1, 0)]))

    existing_trans_shp = {'cap': [], 'linestring': []}

    count = []

    for pro_pair in existingTransCap:
        if pro_pair not in count:
            count.append((pro_pair[1], pro_pair[0]))
            cap = existingTransCap[pro_pair] + existingTransCap[(pro_pair[1], pro_pair[0])]
            if cap != 0:
                existing_trans_shp['cap'].append(cap)
                existing_trans_shp['linestring'].append(
                    shapely.geometry.LineString(
                        [(provin_geo[pro_pair[0]][0], provin_geo[pro_pair[0]][1]),
                         (provin_geo[pro_pair[1]][0], provin_geo[pro_pair[1]][1])]
                    )
                )

    total_trans_shp = {'cap': [], 'linestring': []}

    count = []

    # f_cap_out = open(os.path.join(out_output_processed_path, f'inter_pro_trans_{curr_year}.csv'), 'w+')

    for pro_pair in totalTransCap:
        if pro_pair not in count:
            count.append((pro_pair[1], pro_pair[0]))
            cap = totalTransCap[pro_pair] + totalTransCap[(pro_pair[1], pro_pair[0])]
            if cap != 0:
                total_trans_shp['cap'].append(cap)
                total_trans_shp['linestring'].append(
                    shapely.geometry.LineString(
                        [(provin_geo[pro_pair[0]][0], provin_geo[pro_pair[0]][1]),
                         (provin_geo[pro_pair[1]][0], provin_geo[pro_pair[1]][1])]
                    )
                )
    #             f_cap_out.write('%s,%s,%s\n' % (pro_pair[0], pro_pair[1], cap))
    # f_cap_out.close()

    # Change the default font family
    font = {'family': 'Arial', 'weight': 'bold', 'size': 24}
    plt.rcParams.update({'font.family': font["family"]})

    trans_shp = {
        'Existing': existing_trans_shp,
        'New': new_trans_shp,
        'Total': total_trans_shp
    }

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'province' + dir_flag + 'china_provinces_nmsplit.shp')
    # china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')
    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    for trans_kind in trans_shp:
        trans_gdf = gpd.GeoDataFrame(trans_shp[trans_kind], geometry=trans_shp[trans_kind]['linestring'])
        bound = getBound()

        if trans_kind == 'Existing':
            legend_values = [0.1, 15, 30, 45, 60]
            cmap_vmax = 80
            cmap_name = "autumn_r"
        elif trans_kind == 'Total':
            legend_values = [0.4, 20, 40, 60, 80]
            cmap_vmax = 80
            cmap_name = "autumn_r"
        elif trans_kind == 'New':
            cmap_vmax = 40
            cmap_name = "cividis_r"

        fig = plt.figure(figsize=(12, 8))

        ax_1 = plt.subplot()
        scheme = mc.Quantiles(trans_gdf['cap'], k=5)

        gplt.sankey(
            trans_gdf,
            hue='cap',
            scale='cap',
            legend=False,
            cmap=cmap_name,
            # legend_var='hue',
            # color='dimgray',
            # scheme=scheme,
            # legend_values=legend_values,
            # legend_kwargs={
            #    'marker':'s',
            #    'title':'Capacity',
            #    'shadow':False,
            #    'markeredgewidth':0
            # },
            ax=ax_1
        )

        # Color bar adjustment
        cax = fig.add_axes([0.20, 0.15, 0.20, 0.04])

        im = plt.cm.ScalarMappable(cmap=cmap_name,
                                   norm=plt.Normalize(vmin=0, vmax=cmap_vmax))

        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', label='GW')
        cbar.ax.tick_params(labelsize=10)

        # cbar.set_ticks([-1,125,250,500,750,1000])
        # cbar.update_ticks()

        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_family('Arial')

        china_nine_lines.geometry.plot(ax=ax_1, edgecolor='#747678', facecolor='None', linewidth=0.5)
        # gplt.polyplot(china, ax=ax_1, edgecolor='#747678', facecolor='None', linewidth=0.4)
        china.geometry.plot(ax=ax_1, edgecolor='#747678', facecolor='None', linewidth=0.4)
        china_grids.geometry.plot(ax=ax_1, edgecolor='#747678', facecolor='None', alpha=0, linewidth=0.5)

        ax_1.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
        ax_1.set_ylim(bound.geometry[0].y, bound.geometry[1].y)

        # ax_1.set_title(f'{curr_year}', font=font)
        ax_1.set_title(f'{curr_year-10}-{curr_year}', font=font)
        # ax_1.set_title('' + trans_kind.capitalize() + ' transmission lines', font=font)
        # ax=gplt.polyplot(china_nine_lines,ax=ax,projection=gcrs.AlbersEqualArea())

        ax_child = fig.add_axes((0.68, 0.15, 0.25, 0.25))
        gplt.sankey(trans_gdf, hue='cap', scale='cap', legend=False, cmap=cmap_name,
                    ax=ax_child
                    )
        ax_child = china_nine_lines.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', linewidth=0.3)
        ax_child = china.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', linewidth=0.2)
        ax_child = china_grids.geometry.plot(ax=ax_child, edgecolor='#747678', facecolor='None', alpha=0, linewidth=0.3)
        ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
        ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
        ax_child.set_xticks([])
        ax_child.set_yticks([])
        ax_child.axis('on')

        plt.savefig(os.path.join(out_output_processed_path, "graphs",
                                 f'trans_cap_{trans_kind.lower()}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def TransLoadMap(vre_year, res_tag, curr_year, transKind):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'province' + dir_flag + 'china_provinces_nmsplit.shp')
    # china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')

    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    trans = {}
    trans_load = {}

    if transKind == 'net':
        f_tl = open(os.path.join(out_output_processed_path, 'trans_matrix', 'trans_matrix_tot.csv'))
    elif transKind == 'bi':
        f_tl = open(os.path.join(out_output_processed_path, 'trans_matrix', 'trans_matrix_tot_bi.csv'))

    for line in f_tl:
        line = line.replace('\n', '')
        line = line.split(',')
        trans[line[0]] = line

    f_tl.close()

    if transKind == 'net':
        for i in trans:
            if i != 'TWh':
                for k in range(1, len(trans[i]) - 1):
                    trans_load[(i, trans['TWh'][k])] = round(eval(trans[i][k]), 2)
    elif transKind == 'bi':
        k2pro = {}
        pro2k = {}
        trans_keys = list(trans.keys())

        for i in range(len(trans_keys)):
            k2pro[i] = trans_keys[i]
            pro2k[trans_keys[i]] = i

        pro_count = []
        for i in trans:
            if i != 'TWh':
                for k in range(1, len(trans[i]) - 1):
                    key = (i, trans['TWh'][k])
                    if key not in pro_count:
                        trans_load[(i, trans['TWh'][k])] = round(eval(trans[i][k]), 2) + round(
                            eval(trans[k2pro[k]][pro2k[i]]), 2)
                    pro_count.append(key)

    provin_geo = {}

    f_pg = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')

    next(f_pg)

    for line in f_pg:
        line = line.replace('\n', '')
        line = line.split(',')
        provin_geo[line[0]] = (eval(line[3]), eval(line[4]))

    f_pg.close()

    trans_shp = {'load': [], 'linestring': [], 'start': [], 'end': []}

    for pro_pair in trans_load:
        load = trans_load[pro_pair]
        if load > 0:
            if load >= 100:
                trans_shp['load'].append(102)
            else:
                trans_shp['load'].append(load)
            trans_shp['linestring'].append(
                shapely.geometry.LineString(
                    [(provin_geo[pro_pair[0]][0], provin_geo[pro_pair[0]][1]),
                     (provin_geo[pro_pair[1]][0], provin_geo[pro_pair[1]][1])]
                )
            )

            trans_shp['start'].append(
                shapely.geometry.Point(
                    (provin_geo[pro_pair[0]][0], provin_geo[pro_pair[0]][1])
                )
            )

            trans_shp['end'].append(
                shapely.geometry.Point(
                    (provin_geo[pro_pair[1]][0], provin_geo[pro_pair[1]][1])
                )
            )

    trans_shp['load'].append(102)
    trans_shp['linestring'].append(
        shapely.geometry.LineString([(0, 0), (1, 0)])
    )

    trans_shp['start'].append(
        shapely.geometry.Point((0, 0))
    )

    trans_shp['end'].append(
        shapely.geometry.Point((1, 0))
    )

    trans_gdf = gpd.GeoDataFrame(trans_shp, geometry=trans_shp['linestring'])
    bound = getBound()

    fig = plt.figure(figsize=(12, 8))

    ax_1 = plt.subplot()

    gplt.sankey(
        trans_gdf,
        hue='load',
        scale='load',
        # legend=True,
        cmap='autumn_r',
        # legend_var='hue',
        # legend_kwargs={'label':'TWh'},
        ax=ax_1
    )

    cax = fig.add_axes([0.22, 0.15, 0.02, 0.24])

    im = plt.cm.ScalarMappable(cmap='autumn_r', norm=plt.Normalize(vmin=0, vmax=max(trans_gdf['load'])))

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Arial')

    trans_start_gdf = gpd.GeoDataFrame(trans_shp, geometry=trans_shp['start'])

    # gplt.pointplot(trans_start_gdf,ax=ax_1,color='black',marker='o',s=2)

    trans_end_gdf = gpd.GeoDataFrame(trans_shp, geometry=trans_shp['end'])

    # gplt.pointplot(trans_end_gdf,ax=ax_1,color='black',marker='o',s=2)

    gplt.polyplot(china, ax=ax_1, linewidth=0.4)
    china_nine_lines.geometry.plot(ax=ax_1, color='black', linewidth=0.7)
    china_grids.geometry.plot(ax=ax_1, color='white', edgecolor='black', alpha=0, linewidth=0.5)

    ax_1.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
    ax_1.set_ylim(bound.geometry[0].y, bound.geometry[1].y)

    # ax_1.set_title('Total '+'('+transKind+')'+' Load of Inter-Regional Transmission Line in '+scenYear,fontweight='bold',fontsize=12)
    # ax=gplt.polyplot(china_nine_lines,ax=ax,projection=gcrs.AlbersEqualArea())

    ax_child = fig.add_axes((0.7, 0.15, 0.2, 0.2))

    ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black', linewidth=0.5)
    ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', linewidth=0.4)
    ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
    ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
    ax_child.set_xticks([])
    ax_child.set_yticks([])

    font = {'family': 'Arial', 'weight': 'bold', 'size': 18}

    ax_1.set_title('b) ' + transKind.capitalize() + ' transmission load (TWh)', font=font)

    plt.savefig(os.path.join(out_output_processed_path, "graphs", f'trans_load_{transKind}.png'), dpi=600)
    plt.close()


def TransUtilization(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    year_count = 1

    provin_geo = {}

    pro_name_abb = {}

    f_pg = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')

    next(f_pg)

    for line in f_pg:
        line = line.replace('\n', '')
        line = line.split(',')
        provin_geo[line[1]] = (eval(line[3]), eval(line[4]))
        pro_name_abb[line[1]] = line[0]
    f_pg.close()

    newTransCap = {}

    f_ntc = open(res_dir + 'new_trans_cap' + dir_flag + 'cap_trans_new.csv', 'r+')

    for line in f_ntc:
        line = line.replace('\n', '')
        line = line.split(',')
        line[2] = eval(line[2])
        newTransCap[(line[0], line[1])] = round(line[2], 2)

    f_ntc.close()

    existingTransCap = {}
    f_etc = pd.read_csv(
        work_dir + dir_flag + scen_dir[scenYear] + dir_flag + 'data_csv' + dir_flag + 'inter_pro_trans.csv')

    for pro in f_etc['province']:
        for i in range(len(f_etc[pro])):
            existingTransCap[(pro, f_etc['province'][i])] = 0.001 * f_etc[pro][i]  # MW -> GW

    f_etc = None

    totalTransCap = {}

    for pro_pair in newTransCap:
        totalTransCap[pro_pair] = newTransCap[pro_pair] + existingTransCap[pro_pair]

    trans = {}
    trans_load = {}

    f_tl = open(res_dir + 'trans_matrix' + dir_flag + 'trans_matrix_tot_bi.csv')

    for line in f_tl:
        line = line.replace('\n', '')
        line = line.split(',')
        trans[line[0]] = line

    f_tl.close()

    for i in trans:
        if i != 'TWh':
            for k in range(1, len(trans[i]) - 1):
                trans_load[(i, trans['TWh'][k])] = round(eval(trans[i][k]), 2)

    transUtilization = {}

    for pro_pair in totalTransCap:
        if pro_pair not in transUtilization:
            if totalTransCap[pro_pair] + totalTransCap[(pro_pair[1], pro_pair[0])] != 0:
                pro_pair_abb = (pro_name_abb[pro_pair[0]], pro_name_abb[pro_pair[1]])
                transUtilization[pro_pair] = (
                        (1000 * (trans_load[pro_pair_abb] + trans_load[(pro_pair_abb[1], pro_pair_abb[0])])) /
                        (year_count * 8760 * (totalTransCap[pro_pair] + totalTransCap[(pro_pair[1], pro_pair[0])]))
                )

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china.shp')
    china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')

    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    trans_shp = {'utilization': [], 'linestring': []}

    for pro_pair in transUtilization:
        utilization = transUtilization[pro_pair]
        if utilization > 0:
            trans_shp['utilization'].append(utilization)
            trans_shp['linestring'].append(
                shapely.geometry.LineString(
                    [(provin_geo[pro_pair[0]][0], provin_geo[pro_pair[0]][1]),
                     (provin_geo[pro_pair[1]][0], provin_geo[pro_pair[1]][1])]
                )
            )

    trans_gdf = gpd.GeoDataFrame(trans_shp, geometry=trans_shp['linestring'])

    city_shp = {'point': []}

    for pro in provin_geo:
        city_shp['point'].append(
            shapely.geometry.Point(
                provin_geo[pro]
            )
        )

    city_gdf = gpd.GeoDataFrame(city_shp, geometry=city_shp['point'])

    bound = getBound()

    fig = plt.figure(figsize=(12, 8))

    ax_1 = plt.subplot()

    # gplt.pointplot(city_gdf,ax=ax_1,color='black',marker='o',s=6)

    gplt.sankey(
        trans_gdf,
        hue='utilization',
        legend=False,
        cmap='viridis_r',
        # legend_var='hue',
        # legend_kwargs={'label':'Utilization Rate (0~1)'},
        ax=ax_1
    )

    cax = fig.add_axes([0.22, 0.15, 0.02, 0.24])

    im = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=0, vmax=max(trans_gdf['utilization'])))

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    # cbar.set_ticks([-1,125,250,500,750,1000])

    # cbar.update_ticks()

    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Arial')

    gplt.polyplot(china, ax=ax_1, linewidth=0.4)
    china_nine_lines.geometry.plot(ax=ax_1, color='black', linewidth=0.7)
    china_grids.geometry.plot(ax=ax_1, color='white', edgecolor='black', alpha=0, linewidth=0.5)

    ax_1.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
    ax_1.set_ylim(bound.geometry[0].y, bound.geometry[1].y)

    font = {'family': 'Arial', 'weight': 'bold', 'size': 18}

    ax_1.set_title('a) Utilization rate (0~1)', font=font)

    # ax=gplt.polyplot(china_nine_lines,ax=ax,projection=gcrs.AlbersEqualArea())

    ax_child = fig.add_axes((0.7, 0.15, 0.2, 0.2))

    ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black', linewidth=0.5)
    ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.75)
    ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
    ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
    ax_child.set_xticks([])
    ax_child.set_yticks([])

    plt.savefig(res_dir + 'trans_utilization.png', dpi=600)
    plt.close()


def TransCongestion(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    year_count = 1

    provin_geo = {}

    pro_name_abb = {}

    f_pg = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_pg)
    for line in f_pg:
        line = line.replace('\n', '')
        line = line.split(',')
        provin_geo[line[1]] = (eval(line[3]), eval(line[4]))
        pro_name_abb[line[1]] = line[0]
    f_pg.close()

    newTransCap = {}
    f_ntc = open(os.path.join(out_output_path, 'new_trans_cap', 'cap_trans_new.csv'), 'r+')
    for line in f_ntc:
        line = line.replace('\n', '')
        line = line.split(',')
        line[2] = eval(line[2])
        newTransCap[(line[0], line[1])] = round(line[2], 2)
    f_ntc.close()

    existingTransCap = {}
    f_etc = pd.read_csv(os.path.join(out_input_path, 'inter_pro_trans.csv'))

    for pro in f_etc['province']:
        for i in range(len(f_etc[pro])):
            existingTransCap[(pro, f_etc['province'][i])] = 0.001 * f_etc[pro][i]  # MW -> GW

    f_etc = None
    totalTransCap = {}
    for pro_pair in newTransCap:
        totalTransCap[pro_pair] = newTransCap[pro_pair] + existingTransCap[pro_pair]

    with open(os.path.join(out_output_processed_path, 'trans_tot_hourly.pkl'), 'rb+') as fin:
        transTotHourly = pickle.load(fin)
    fin.close()

    transCongestion = {}

    for pro_pair in transTotHourly:
        transCongestion[pro_pair] = 0
        if totalTransCap[pro_pair] != 0:
            for h in range(year_count * 8760):
                if math.fabs(transTotHourly[pro_pair][h] - totalTransCap[pro_pair]) <= 0.01:
                    transCongestion[pro_pair] += 1

    for pro_pair in transCongestion:
        transCongestion[pro_pair] = transCongestion[pro_pair] / (year_count * 8760)

    china = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'province' + dir_flag + 'china_provinces_nmsplit.shp')
    # china = china.dissolve(by='OWNER').reset_index(drop=False)
    china_grids = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'grids' + dir_flag + 'Grid_regions_four.shp')
    china_nine_lines = gpd.read_file(work_dir + 'data_shp' + dir_flag + 'china_nine_dotted_line.shp')

    trans_shp = {'congestion': [], 'linestring': []}

    # Save results
    f_tcongest = open(os.path.join(out_output_processed_path, "graphs", 'trans_congestion.csv'), 'w+')

    for pro_pair in totalTransCap:
        capacity = totalTransCap[pro_pair]
        if capacity > 0:
            trans_shp['congestion'].append(transCongestion[pro_pair])
            f_tcongest.write('%s,%s,%s\n' % (pro_pair[0], pro_pair[1], transCongestion[pro_pair]))
            trans_shp['linestring'].append(
                shapely.geometry.LineString(
                    [(provin_geo[pro_pair[0]][0], provin_geo[pro_pair[0]][1]),
                     (provin_geo[pro_pair[1]][0], provin_geo[pro_pair[1]][1])]
                )
            )

    f_tcongest.close()

    trans_gdf = gpd.GeoDataFrame(trans_shp, geometry=trans_shp['linestring'])

    city_shp = {'point': []}

    for pro in provin_geo:
        city_shp['point'].append(
            shapely.geometry.Point(
                provin_geo[pro]
            )
        )

    city_gdf = gpd.GeoDataFrame(city_shp, geometry=city_shp['point'])

    bound = getBound()

    fig = plt.figure(figsize=(12, 8))

    ax_1 = plt.subplot()

    # gplt.pointplot(city_gdf,ax=ax_1,color='black',marker='o',s=6)

    gplt.sankey(
        trans_gdf,
        hue='congestion',
        legend=False,
        cmap='viridis_r',
        # legend_var='hue',
        # legend_kwargs={'label':'Congestion Rate (0~1)'},
        ax=ax_1
    )

    cax = fig.add_axes([0.22, 0.15, 0.02, 0.24])

    im = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=0, vmax=max(trans_gdf['congestion'])))

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    # cbar.set_ticks([-1,125,250,500,750,1000])

    # cbar.update_ticks()

    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Arial')

    gplt.polyplot(china, ax=ax_1, linewidth=0.4)
    china_nine_lines.geometry.plot(ax=ax_1, color='black', linewidth=0.7)
    china_grids.geometry.plot(ax=ax_1, color='white', edgecolor='black', alpha=0.0, linewidth=0.5)

    ax_1.set_xlim(bound.geometry[0].x, bound.geometry[1].x)
    ax_1.set_ylim(bound.geometry[0].y, bound.geometry[1].y)
    font = {'family': 'Arial', 'weight': 'bold', 'size': 18}
    ax_1.set_title('b) Congestion rate (0~1)', font=font)
    # ax=gplt.polyplot(china_nine_lines,ax=ax,projection=gcrs.AlbersEqualArea())

    ax_child = fig.add_axes((0.7, 0.15, 0.2, 0.2))

    ax_child = china_nine_lines.geometry.plot(ax=ax_child, color='black', linewidth=0.5)
    ax_child = china.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.75)

    china_grids.geometry.plot(ax=ax_child, color='white', edgecolor='black', alpha=0.0, linewidth=0.5)
    ax_child.set_xlim(bound.geometry[2].x, bound.geometry[3].x)
    ax_child.set_ylim(bound.geometry[2].y, bound.geometry[3].y)
    ax_child.set_xticks([])
    ax_child.set_yticks([])

    plt.savefig(os.path.join(out_output_processed_path, "graphs", 'trans_congestion.png'), dpi=600)
    plt.close()


if __name__ == '__main__':
    # Test the following functions
    res_tag = "test_0224_365days_all_years_b1_low"
    vre_year = "w2015_s2015"
    curr_year = 2060

    windsolarCellMap(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    # windsolarInstalledFactor(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    TransCapMap(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    # TransDirect(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    # TransLoadMap(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year, transKind="bi")
    # TransLoadMap(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year, transKind="net")
    # TransCongestion(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)

    # TODO legacy code below
    # installedCapByProvince()
    # StorageProfile(vre_tag,scenario_tag)
    # loadSheddingByProvince(vre_tag,scenario_tag)
    # VREValueDistribution(vre_tag,scenario_tag)
    # TotalEnergyToHourVisual(vre_tag,scenario_tag)

    # wind and solar cell information
    # windsolarCapCurt(vre_tag,scenario_tag)
    # curtWS(2016,2015,5)

    # map
    scen_pool = ['Base1220', 'BAT_OC_0.5X', 'BAT_OC_2X', 'Solar_OC_0.5X',
                 'Solar_OC_2X', 'Solar_PF_0.5', 'Transmission_0.5X', 'Transmission_2X',
                 'Wind_PF_0.5', 'Resv_0.1', 'Resv_within_province', 'Nuclear_MR_0.5', 'Nuclear_150GW',
                 'Neg_emission_0', 'Neg_emission_4', 'Neg_emission_2', 'Gas_cap_0.5X',
                 'Demand_1.2X', 'With_CAES', 'With_CAES_VRB', 'With_VRB', 'East_province_pf_0.5',
                 'Wind_OC_2X', 'Wind_PF_0.5', 'Load_shedding', 'Wind_OC_0.5X']

    # GeneratorDistribution(vre_tag,scenario_tag)
    # windsolarIntedMap()

    # storage
    # averageStoreLengthVision(vre_tag,scenario_tag)
    # storageCap(vre_tag,scenario_tag)
    # totEnergyStore(vre_tag,'bat',scenario_tag)
    # totEnergyStore(vre_tag,'phs',scenario_tag)
    # StorageProfile(vre_tag,scenario_tag)

    # cf_curve(2016,2015,3.0)

    # for scenario_tag in tmp:
    #    LoadProfile(vre_tag,scenario_tag)

    # LoadProfile(vre_tag,scenario_tag,1)
    # DemandProfile()

    # lcoeNation(2016,2015,365,5,2.62,3.3,1)
