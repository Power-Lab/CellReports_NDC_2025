import os
import shutil
import math
import pickle

import numpy as np
import pandas as pd

from callUtility import dirFlag, getWorkDir

pd.set_option('mode.chained_assignment', None)
work_dir = getWorkDir()
csv_folder = os.path.join(work_dir, "data_csv")


class MultiYearAutomation:
    def __init__(self, yr_start: int, yr_end: int, yr_step: int, res_tag: str, vre_year: str,
                 emission_target: str, demand_sensitivity: str, run_mode: str):
        """
        Attributes
        ----------
        yr_start: int
            The starting year of the requested year sequence. This should be greater than 2020

        yr_end: int
            The end year of the requested year sequence

        yr_step: int
            The distance between each consecutive years in the year sequence.

        rel_path: str
            The relative path (starting from where this file is currently located at) to the folder that all the outputs will be generated.
            If the specified folder does not exist, a folder will be generated.
        """

        self.yr_start = yr_start
        self.yr_end = yr_end
        self.yr_step = yr_step
        self.res_tag = res_tag
        self.vre_year = vre_year
        self.emission_target = emission_target
        self.demand_sensitivity = demand_sensitivity
        self.run_mode = run_mode
        self.out_path = os.path.join(work_dir, "data_res", res_tag + "_" + vre_year)

        # create an empty folder if the folder does not exist
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

        # for each predicted year, create a projection for it.
        # range does not include yrEnd, so extend by one step
        # The year list is a full list, not filtered by yr_start.
        yr_full_ls = list(range(2020, yr_end + yr_step, yr_step))
        yr_req_ls = list(range(yr_start, yr_end + 1, yr_step))
        for yr in yr_req_ls:
            curr_yr_dir = os.path.join(self.out_path, str(yr))
            if not os.path.exists(curr_yr_dir):
                os.mkdir(curr_yr_dir)

        self.yr_full_ls = yr_full_ls
        self.yr_req = yr_req_ls

    def demand_projection(self) -> None:
        """
        Projecting hourly electricity demand at a provincial level for the year sequence specified by the user.

        High level description
        ----------------------
        This function uses two inputs not specified by the user: hourly demand
        for the year 2060 and total demand at the year 2020. Assuming the
        annual profile curve remains the same, the algorithm scale back the
        hourly demand at the year 2060 linearly from 2020 to 2060. 
        
        
        Side Effects
        ------------
        This function generate all demand projection under the folder specified
        by the constructor function. At the top level, each year in the
        requested year sequence will have its own folder, and each year folder
        will contain all the provincial hourly
        projection, each csv file
        containing projection for one province.
        """

        # under each year folder, create a folder called "prov_demand_hourly"
        for yr in self.yr_req:
            curr_path = os.path.join(self.out_path, str(yr), "provin_demand_hourly")
            os.mkdir(curr_path)

        # total provincial demand at 2020, set index so I can use provin as row index
        province_dem_ttl_2020 = pd.read_csv(os.path.join(work_dir, "data_csv", "demand_assumptions",
                                                         "province_dem_2030.csv"))
        province_dem_ttl_2020 = province_dem_ttl_2020.set_index("provin")

        # convert from MWh to GWh
        province_dem_ttl_2020.loc[:, "dem"] = province_dem_ttl_2020.loc[:, "dem"] / 1000

        # calculate total nation demand in 2060
        provin_dem_hour_2060_path = os.path.join(work_dir, "data_csv", "demand_assumptions",
                                                 "province_demand_by_hour_2060")
        nation_dem_full_60 = pd.read_csv(os.path.join(provin_dem_hour_2060_path, "nation_dem_full.csv"), header=None)
        nation_dem_full_60.columns = ["hour", "dem"]
        nation_ttl_dem_60 = sum(nation_dem_full_60["dem"] / 1e6)  # PWh

        # for each provincial hourly projection at 2060, scale back based on scaling factors of each decade
        files_ls = os.listdir(provin_dem_hour_2060_path)
        for curr_csv in files_ls:
            curr_prov = curr_csv.split(sep=".")[0]
            curr_prov_hourly_dem = pd.read_csv(os.path.join(provin_dem_hour_2060_path, curr_csv), header=None)
            curr_prov_hourly_dem.columns = ["hour", "dem"]

            # Anticipating we cannot find the demand in the 2020 total demand df
            try:
                curr_ttl_dem_20 = province_dem_ttl_2020.loc[curr_prov, "dem"]
            except:
                continue

            # # project scaling factor linearly
            # scale_fac_60 = curr_ttl_dem_60 / curr_ttl_dem_20
            # # create the entire scaling factor list first, then subset to
            # # requested years
            # # scale_fac_dic_req = list(self.project_scales(scale_fac_60).items())
            # # provin_dem.csv is actually 2030 data -- recalculate the scaling factors here
            # post_2030_yr = [yr for yr in self.yr_req if yr >= 2030]
            # pre_2030_yr = [yr for yr in self.yr_req if yr < 2030]
            # step = (scale_fac_60 - 1) / (len(post_2030_yr) - 1)
            # scale_fac_dic_req = [(self.yr_req[i],
            #                       1 + step * (i - len(pre_2030_yr)))
            #                      for i in range(len(self.yr_req))]

            if self.emission_target == "2C":
                # Total national demand under the 2C scenario
                # nation_demand_dict = {2025: 8.95, 2030: 10.4, 2035: 11.45, 2040: 12.5,
                #                       2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}  # PWh, used for the pathways
                # if self.demand_sensitivity == "2030b_2035p10":
                #     nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 16.5, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # elif self.demand_sensitivity == "2030b_2035m10":
                #     nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 13.5, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # elif self.demand_sensitivity == "2030m10_2035b":
                #     nation_demand_dict = {2025: 8.95, 2030: 11.25, 2035: 15.0, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # elif self.demand_sensitivity == "2030m10_2035p10":
                #     nation_demand_dict = {2025: 8.95, 2030: 11.25, 2035: 16.5, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # elif self.demand_sensitivity == "2030m10_2035m10":
                #     nation_demand_dict = {2025: 8.95, 2030: 11.25, 2035: 13.5, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # else:
                #     nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 15.0, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}  # adjust for 2030 and 2035
                if self.demand_sensitivity == "2030b_2035p10":
                    nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 14.85, 2040: 12.5,
                                          2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                elif self.demand_sensitivity == "2030m10_2035b":
                    nation_demand_dict = {2025: 8.95, 2030: 13.5, 2035: 15.0, 2040: 12.5,
                                          2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                elif self.demand_sensitivity == "2030m10_2035m10":
                    nation_demand_dict = {2025: 8.95, 2030: 13.5, 2035: 14.5, 2040: 12.5,
                                          2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                else:
                    nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 13.5, 2040: 12.5,
                                          2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}  # adjust for 2030 and 2035
            else:  # "15C"
                # Total national demand under the 1.5C scenario
                # nation_demand_dict = {2025: 9.5, 2030: 11.5, 2035: 12.45, 2040: 13.4,
                #                       2045: 14.1, 2050: 14.8, 2055: 15.4, 2060: 16.0}  # PWh
                # if self.demand_sensitivity == "2030b_2035p10":
                #     nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 16.5, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # elif self.demand_sensitivity == "2030b_2035m10":
                #     nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 13.5, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # elif self.demand_sensitivity == "2030m10_2035b":
                #     nation_demand_dict = {2025: 8.95, 2030: 11.25, 2035: 15.0, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # elif self.demand_sensitivity == "2030m10_2035p10":
                #     nation_demand_dict = {2025: 8.95, 2030: 11.25, 2035: 16.5, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # elif self.demand_sensitivity == "2030m10_2035m10":
                #     nation_demand_dict = {2025: 8.95, 2030: 11.25, 2035: 13.5, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                # else:
                #     nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 15.0, 2040: 12.5,
                #                           2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}  # adjust for 2030 and 2035
                if self.demand_sensitivity == "2030b_2035p10":
                    nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 14.85, 2040: 12.5,
                                          2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}
                else:
                    nation_demand_dict = {2025: 8.95, 2030: 12.5, 2035: 13.5, 2040: 12.5,
                                          2045: 13.2, 2050: 13.9, 2055: 14.5, 2060: 15.1}  # adjust for 2030 and 2035

            scale_fac_dic_req = [(self.yr_req[i],
                                  nation_demand_dict[self.yr_req[i]] / nation_ttl_dem_60
                                  ) for i in range(len(self.yr_req))]
            # for each province, generate a csv demand file for each year in
            # the year sequence
            for i in range(len(scale_fac_dic_req)):
                curr_yr, curr_scale_fac = scale_fac_dic_req[i]

                if self.demand_sensitivity == "p5":
                    curr_scale_fac = curr_scale_fac * (1 + 0.05)
                elif self.demand_sensitivity == "m5":
                    curr_scale_fac = curr_scale_fac * (1 - 0.05)
                elif self.demand_sensitivity == "p20":
                    curr_scale_fac = curr_scale_fac * (1 + 0.2)
                elif self.demand_sensitivity == "m20":
                    curr_scale_fac = curr_scale_fac * (1 - 0.2)
                else:
                    pass

                # Multiply hourly profile by scale_fac at each year
                curr_dem_scaled = curr_prov_hourly_dem.copy()
                curr_dem_scaled.loc[:, "dem"] = curr_dem_scaled.loc[:, "dem"] * curr_scale_fac
                curr_yr_dem_path = os.path.join(self.out_path, str(curr_yr), "provin_demand_hourly", curr_csv)
                curr_dem_scaled.to_csv(curr_yr_dem_path, index=False)

    def cost_projection(self):
        """
        For each requested year of the year sequence, project all the costs,
        including costs of non-VRE generation resources, 
        storage technologies, and inter-provincial transmission lines

        The techno-economic costs are interpolated linearly based on cost data
        from 2020 and 2060. The costs are listed in data_csv/cost_2020 and
        data_csv/cost_2060.
        """

        # read in both df, filter to get rid of NAs, inner join
        cost_2020 = pd.read_csv(os.path.join(
            work_dir, "data_csv", "cost_assumptions", "cost_2020.csv"))
        cost_2020['res_id'] = cost_2020['resource_type'] + "_" + \
                              cost_2020['cost_name']
        cost_2020 = cost_2020[cost_2020['cost'].notnull()]

        cost_2060 = pd.read_csv(os.path.join(
            work_dir, "data_csv", "cost_assumptions", "cost_2060.csv"))
        cost_2060['res_id'] = cost_2060['resource_type'] + "_" + \
                              cost_2060['cost_name']
        cost_2060 = cost_2060[cost_2060['cost'].notnull()]

        cost_ttl = pd.merge(cost_2020.loc[:, ["res_id", "cost", "unit"]],
                            cost_2060.loc[:, ["res_id", "cost"]], how="inner",
                            on="res_id", suffixes=["_2020", "_2060"])

        # build a collection, each requested year contains a dict, which
        # will be converted into a df and saved to year folders within
        # out_path folder.
        yr_dic = {}

        # iterate through each cost item
        for _, r in cost_ttl.iterrows():
            cost_res_id = r["res_id"]
            cost_unit = r["unit"]
            cost_scales_dict = self.project_scales(r["cost_2060"] / r["cost_2020"])

            scale_cost_dic_req = list({yr: scale * r["cost_2060"] for (yr, scale) in cost_scales_dict.items()}.items())

            # for each cost, interpolate and weave into yr_dic
            for yr, interpolated_cost in scale_cost_dic_req:
                if not str(yr) in yr_dic:
                    yr_dic[str(yr)] = {"res_id": [], "cost": [], "unit": []}
                curr_yr_dic = yr_dic[str(yr)]
                curr_yr_dic["res_id"].append(cost_res_id)
                curr_yr_dic["cost"].append(interpolated_cost)
                curr_yr_dic["unit"].append(cost_unit)

        # convert dict to csv and save
        for yr in yr_dic:
            curr_df = pd.DataFrame.from_dict(yr_dic[yr])
            curr_df.to_csv(
                os.path.join(
                    self.out_path, yr, "costs.csv"
                ),
                index=False
            )

    def cost_knitting(self, exo_var_dict):
        """
        This function takes in the exogenous variable dictionary
        (scen_params.json), and modify all the time-varying exo param based 
        on the linear time projection. 
        Note, the labels in scen_params.json and costs.csv's are different. 
        The conversion of label is recorded in cost2scenParam_dictionary.csv 
        in data_csv.

        Argument
        --------
        exo_var_dict: dictionary
            The dictionary from scen_param.json that contains all the exogenous
            variables.

        Returns
        -------
        Dictionary
            A dictionary object that has the same structure as `exo_var_dict` 
            but have incorporated all the new projections from `cost.csv`
        """

        # name conversion dictioanry 
        print("old dict")
        print(exo_var_dict)
        convert_dict = pd.read_csv(os.path.join(work_dir, "data_csv",
                                                "cost2scenParam_dictionary.csv"))
        convert_dict.set_index("cost name", inplace=True)
        update_param_ls = convert_dict.index

        # set of params that will be updated

        # load in the necessary data 
        for yr in self.yr_req:
            curr_cost_path = os.path.join(self.out_path, str(yr), "costs.csv")
            curr_new_cost = pd.read_csv(curr_cost_path)
            print(curr_new_cost)
            curr_new_cost.set_index("res_id", inplace=True)

            # for all params that needs update, update them
            for cost_name in update_param_ls:
                scen_name = convert_dict.loc[cost_name, "scen_params name"]
                new_cost = curr_new_cost.loc[cost_name, "cost"]
                # double level dict are indicated by ">"
                if ">" in scen_name:
                    # some params are in dict that has three levels. split them.
                    # this is hard coded in, there should be a better way 
                    # to pass into the dictionary keys.
                    name_ls = [x.strip() for x in scen_name.split(">")]
                    if len(name_ls) == 3:
                        exo_var_dict[name_ls[0]][name_ls[1]][name_ls[2]] = new_cost
                    elif len(name_ls) == 2:
                        exo_var_dict[name_ls[0]][name_ls[1]] = new_cost
                    else:
                        exo_var_dict[scen_name] = new_cost
        return exo_var_dict

    def project_prov_hydro_helper(self, total, sw, nw, e_c, other, year):
        """
        This function projects provincial level hydro generation capacity. 
        The projection is structured so that the total hydro generation capacity
        complies with the projection provided by GEIDCO. See 中国2030年能源电力
        发展规划研究及2060年展望, accessed through https://yhp-website.oss-cn-beijing.aliyuncs.com/upload/%E3%80%8A%E4%B8%AD%E5%9B%BD2030%E5%B9%B4%E8%83%BD%E6%BA%90%E7%94%B5%E5%8A%9B%E5%8F%91%E5%B1%95%E8%A7%84%E5%88%92%E7%A0%94%E7%A9%B6%E5%8F%8A2060%E5%B9%B4%E5%B1%95%E6%9C%9B%E3%80%8B_1616498546246.pdf

        The function first 
        compute regional growth rate based on the regional capacity projection, 
        country-level hydro capacity, and current provincial-level hydro 
        capacity. 
        Then, the current provincial hydro capacity is scaled by the growth rate
        based on which region the province belongs to. 

        The code is modified based off of Ziheng's original code.

        Arguments
        ---------
        total: float
            total amount of hydro nationwide, with a unit of 10^2 GW. (亿千瓦). 
        sw: float 
            fraction of hydro capacity in the southwest region
        nw: float 
            fraction of hydro capacity in the north west region
        e_c: float 
            fraction of hydro capacity in the east and central region
        other: float 
            fraction of hydro capacity in all the other regions.
        sub_path: string
            the name of the folder that is directly located under self.out_path 
        year: int
            the requested year

        Side Effects
        ------------
        Output the provincial hydro capacity as a csv file under data_res under 
        each year folder
        """

        work_dir = getWorkDir()
        dir_flag = dirFlag()

        pro_loc_in_china = {}

        f_province = open(work_dir + 'data_csv' + dir_flag +
                          'geography/China_provinces_hz.csv', 'r+')
        next(f_province)

        for line in f_province:
            line = line.replace('\n', '')
            line = line.split(',')

            if line[6] == 'East' or line[6] == 'Central':
                region = 'e_c'
            elif line[6] == 'SW' or line[6] == 'NW':
                region = line[6]
            else:
                region = 'other'

            pro_loc_in_china[line[1]] = region

        f_province.close()

        region_hydro = {}
        provin_hydro_cap = {}

        f_hydro = open(work_dir + 'data_csv' + dir_flag + 'capacity_assumptions/hydro_2020.csv', 'r+')

        for line in f_hydro:
            line = line.replace('\n', '')
            line = line.split(',')

            if pro_loc_in_china[line[0]] not in region_hydro:
                region_hydro[pro_loc_in_china[line[0]]] = 0

            provin_hydro_cap[line[0]] = eval(line[1])
            region_hydro[pro_loc_in_china[line[0]]] += eval(line[1])

        f_hydro.close()

        # convert to MW
        region_hydro_plan = {
            'SW': 100000 * total * sw,
            'NW': 100000 * total * nw,
            'e_c': 100000 * total * e_c,
            'other': 100000 * total * other
        }

        hydro_beta_province = {}
        provin_hydro_cap_proj = {}

        # compute the projected hydro capacity so that regional projection
        # complies with the GEIDCO projection at a regional level
        for pro in provin_hydro_cap:
            region = pro_loc_in_china[pro]
            hydro_beta_province[pro] = region_hydro_plan[region] / \
                                       region_hydro[region]
            provin_hydro_cap_proj[pro] = hydro_beta_province[pro] * \
                                         provin_hydro_cap[pro]

        hydro_prov_df = pd.DataFrame.from_dict(
            provin_hydro_cap_proj,
            orient="index")

        hydro_prov_df.reset_index(inplace=True)
        hydro_prov_df.columns = ["province", f"{year}_cap_mw"]

        # also bind region column for testing purposes
        hydro_prov_df_region = [pro_loc_in_china[i]
                                for i in hydro_prov_df["province"]]
        hydro_prov_df["region"] = hydro_prov_df_region
        hydro_prov_df = hydro_prov_df.sort_values(by="province", ascending=True).reset_index(drop=True)

        hydro_prov_df.to_csv(os.path.join(self.out_path,
                                          str(year), "hydro.csv"), index=False)

    def project_prov_hydro(self):
        """
        A wrapper for project_prov_hydro_helper, mostly parsing the data 
        before feeding the data to project_prov_hydro_helper to do to real 
        work. 
        """

        if self.run_mode == "ndc":
            hydro_plan = pd.read_csv(
                os.path.join(work_dir, "data_csv", "capacity_assumptions", "hydro_plan_ndc.csv")
            )
        else:
            hydro_plan = pd.read_csv(
                os.path.join(work_dir, "data_csv", "capacity_assumptions", "hydro_plan.csv")
            )
        hydro_plan = hydro_plan.set_index("year")
        for yr in self.yr_req:
            curr_plan = list(hydro_plan.loc[yr, :])
            self.project_prov_hydro_helper(*curr_plan, yr)

    def project_prov_nuclear(self):

        nuclear_2020 = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "nuclear_2020.csv"),
                                   header=None)
        nuclear_2020.columns = ["province", "cap_mw"]
        nuclear_2050 = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "nuclear_2060.csv"),
                                   header=None)
        nuclear_2050.columns = ["province", "cap_mw"]
        nuclear_ttl = pd.merge(nuclear_2020, nuclear_2050,
                               on="province", suffixes=["_2020", "_2050"])

        yr_dic = {}

        # nuclear stops to grow in 2050, and will stay fixed between 2050 and
        # 2060
        yr_full_ls = [i for i in self.yr_full_ls if i <= 2050]
        for _, r in nuclear_ttl.iterrows():
            # # avoid divide by zero... I am pretty sure there are
            # # other good ways to fix this.
            # if r["cap_mw_2020"] == 0:
            #     r["cap_mw_2020"] = 0.000001
            # nuclear_scale_dict = self.project_scales(r["cap_mw_2050"] /
            #                                          r["cap_mw_2020"])
            # nuclear_cap_dict_fil = {yr: scale * r["cap_mw_2020"] for (yr, scale)
            #                         in nuclear_scale_dict.items() if yr <= 2050}
            # nuclear_cap_ls = list(nuclear_cap_dict_fil.values())

            nuclear_cap_ls = np.linspace(
                r['cap_mw_2020'], r["cap_mw_2050"], num=len(yr_full_ls))
            nuclear_cap_ls = list(nuclear_cap_ls)

            # repeat the cap of 2050 for all years after 2050
            # since the projection only goes to 2050, and stay the same after
            if self.yr_end > 2050:
                rep_times = int((self.yr_end - 2050) / self.yr_step)
                nuclear_cap_ls = nuclear_cap_ls + \
                                 rep_times * [nuclear_cap_ls[-1]]
            # nuclear_dic = dict(zip(nuclear_scale_dict.keys(), nuclear_cap_ls))
            # nuclear_req_ls = list({yr: cap for (yr, cap) in
            #                        nuclear_dic.items() if yr >= self.yr_start and yr <= self.yr_end}.items())

            # for i in range(len(nuclear_req_ls)):
            for i in range(len(self.yr_full_ls)):
                yr = self.yr_full_ls[i]
                curr_cap = nuclear_cap_ls[i]

                # Update provincial nuclear capacities in NDC mode
                if self.run_mode == "ndc":
                    if i == 2:  # 2030
                        curr_cap = curr_cap / 111.1733333 * 125
                    elif i == 3:  # 2035
                        curr_cap = curr_cap / 137.88 * 175.47
                    else:
                        pass

                # yr, curr_cap = nuclear_req_ls[i]
                if not str(yr) in yr_dic:
                    yr_dic[str(yr)] = {"province": [], f"{yr}_cap_mw": []}
                curr_yr_dict = yr_dic[str(yr)]
                curr_yr_dict["province"].append(r["province"])
                curr_yr_dict[f"{yr}_cap_mw"].append(curr_cap)

        # for i in range(len(nuclear_req_ls)):
        for yr in self.yr_full_ls[1:]:
            # yr, _ = nuclear_req_ls[i]
            curr_df = pd.DataFrame.from_dict(yr_dic[str(yr)]).\
                sort_values(by="province", ascending=True).reset_index(drop=True)
            curr_df.to_csv(
                os.path.join(
                    self.out_path, str(yr), "nuclear.csv"
                ),
                index=False
            )

    def project_prov_firm(self, resource_type: str):
        """

        This function project firm power 
        Arguments 
        ---------

        resource_type: str
            "coal"

        """

        prov_ccs_cap_df = pd.DataFrame()
        prov_unabated_cap_df = pd.DataFrame()

        # set parameters
        retrofit_start_year = 2030
        coal_lifespan = 40

        # data from Global Energy Monitor, Coal Plants Tracker
        # import and filter to only the type of resource
        retrofit_schedule = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "coal_plan",
                                                     f"retrofit_schedule_start_{retrofit_start_year}_v2022.csv"))
        retrofit_schedule = retrofit_schedule.loc[
                            retrofit_schedule['resource'] == resource_type, :].reset_index(drop=True)

        # similar, but with retirement schedule
        retirement_schedule = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "coal_plan",
                                                       f"plants_retirement_schedule_{coal_lifespan}_v2022.csv"))
        retirement_schedule = retirement_schedule.loc[
                              retirement_schedule['resource'] == resource_type, :].reset_index(drop=True)

        # Rename Inner Mongolia to East/West Inner Mongolia
        east_inner_mongolia_list = ["Chifeng", "Tongliao", "Hulunbuir", "Hinggan", "Xilingol"]
        west_inner_mongolia_list = ["Hohhot", "Baotou", "Bayannur", "Wuhai", "Ordos", "Ulanqab", "Alxa"]
        for idx in retrofit_schedule.index:
            if retrofit_schedule["Subnational unit (province, state)"].iloc[idx] == "Inner Mongolia":
                if retrofit_schedule["Major area (prefecture, district)"].iloc[idx] in east_inner_mongolia_list:
                    retrofit_schedule["Subnational unit (province, state)"].iloc[idx] = "EastInnerMongolia"
                else:
                    retrofit_schedule["Subnational unit (province, state)"].iloc[idx] = "WestInnerMongolia"
        for idx in retirement_schedule.index:
            if retirement_schedule["Subnational unit (province, state)"].iloc[idx] == "Inner Mongolia":
                if retirement_schedule["Major area (prefecture, district)"].iloc[idx] in east_inner_mongolia_list:
                    retirement_schedule["Subnational unit (province, state)"].iloc[idx] = "EastInnerMongolia"
                else:
                    retirement_schedule["Subnational unit (province, state)"].iloc[idx] = "WestInnerMongolia"

        # Rename Tibet to Xizang
        for idx in retrofit_schedule.index:
            if retrofit_schedule["Subnational unit (province, state)"].iloc[idx] == "Tibet":
                retrofit_schedule["Subnational unit (province, state)"].iloc[idx] = "Xizang"
        for idx in retirement_schedule.index:
            if retirement_schedule["Subnational unit (province, state)"].iloc[idx] == "Tibet":
                retirement_schedule["Subnational unit (province, state)"].iloc[idx] = "Xizang"

        # Create a dictionary to track provincial cap over years
        retrofit_schedule["final_end_year"] = retrofit_schedule["final_end_year"].apply(
            lambda x: x if x <= 2060 else 2060)
        retirement_schedule["final_end_year"] = retirement_schedule["final_end_year"].apply(
            lambda x: x if x <= 2060 else 2060)
        retirement_schedule = retirement_schedule[
            (retirement_schedule["final_end_year"] >= 2020) &
            (retirement_schedule["Status"].isin(["construction", "operating", "permitted", "proposed"]))] \
            .reset_index(drop=True)
        prov_res_cap_df = \
            retirement_schedule \
                .groupby(by='Subnational unit (province, state)') \
                .agg({"Capacity (MW)": "sum"})
        prov_res_cap_dict = prov_res_cap_df.to_dict()["Capacity (MW)"]

        yr_prov_dict = {}

        retrofit_plants_set = set(retrofit_schedule.loc[:, "Tracker ID"])

        # iterating each row of retrofit_schedule, and create a dictionary
        # indexed by year and status pair, containing the provincial retrofit
        # and retirement capacity. Retrofit and unabated tracks CCS and regular
        # firm power gen capacity profile.
        for _, r in retrofit_schedule.iterrows():
            curr_cap = r["Capacity (MW)"]
            curr_retrofit_yr = r["retrofit_year"]
            curr_prov = r["Subnational unit (province, state)"]
            curr_retirement_yr = r["final_end_year"]

            if not curr_retrofit_yr in yr_prov_dict:
                yr_prov_dict[curr_retrofit_yr] = {}

            curr_dict = yr_prov_dict[curr_retrofit_yr]

            # update to convert unabated to retrofit
            if not (curr_prov, "retrofit") in curr_dict:
                curr_dict[(curr_prov, "retrofit")] = curr_cap
            else:
                curr_dict[(curr_prov, "retrofit")] += curr_cap

            # taking away unabated cap since the plant is retrofitted into
            # CCS
            if (curr_prov, "unabated") not in curr_dict:
                curr_dict[(curr_prov, "unabated")] = -curr_cap
            else:
                curr_dict[(curr_prov, "unabated")] -= curr_cap

            # subtract when retrofitted plant retires
            if not curr_retirement_yr in yr_prov_dict:
                yr_prov_dict[curr_retirement_yr] = {}
            curr_dict = yr_prov_dict[curr_retirement_yr]

            if not (curr_prov, "retrofit") in curr_dict:
                curr_dict[(curr_prov, "retrofit")] = -curr_cap
            else:
                curr_dict[(curr_prov, "retrofit")] -= curr_cap

        for _, r in retirement_schedule.iterrows():
            curr_cap = r["Capacity (MW)"]
            curr_retirement_yr = r["final_end_year"]
            curr_start_yr = r["final_start_year"]
            curr_prov = r["Subnational unit (province, state)"]
            curr_id = r["Tracker ID"]

            # retirement, appending negative value to yr-province dict
            if curr_id not in retrofit_plants_set:

                if not curr_retirement_yr in yr_prov_dict:
                    yr_prov_dict[curr_retirement_yr] = {}
                curr_dict = yr_prov_dict[curr_retirement_yr]

                if (curr_prov, "unabated") not in curr_dict:
                    curr_dict[(curr_prov, "unabated")] = -curr_cap
                else:
                    curr_dict[(curr_prov, "unabated")] -= curr_cap

                # TODO comment out the section below as new plants are already counted in prov_res_cap_dict
                # # adding more capacity to yr-province dict for new plants
                # if not curr_start_yr in yr_prov_dict:
                #     yr_prov_dict[curr_start_yr] = {}
                # curr_dict = yr_prov_dict[curr_start_yr]
                #
                # if (curr_prov, "unabated") not in curr_dict:
                #     curr_dict[(curr_prov, "unabated")] = curr_cap
                # else:
                #     curr_dict[(curr_prov, "unabated")] += curr_cap

        # filter to only contain years larger or equal to 2020.
        yr_prov_dict_filtered = {yr: _ for (
            yr, _) in yr_prov_dict.items() if yr >= 2020}

        # CCS generation capacity as indicated by "retrofit"
        prov_ccs_cap_dict = dict.fromkeys(prov_res_cap_df.index.to_list(), 0)

        # for retrofit, assume starts in 2020.
        for yr in range(2020, self.yr_end + 1):
            if yr not in yr_prov_dict_filtered:
                continue

            # modify ccs capacity tracker
            prov_dict = yr_prov_dict_filtered[yr]
            for prov_n_status, cap in prov_dict.items():
                prov, status = prov_n_status
                if status == "retrofit":
                    # add to or subtract from prov ccs cap tracker
                    prov_ccs_cap_dict[prov] += cap

            # If the year is not requested, don't write out the year
            if yr not in self.yr_req:
                continue

            # write out to the file if the year is requested, otherwise just
            # modify the tracker
            if yr in self.yr_req:
                out_path = os.path.join(self.out_path, str(yr),
                                        resource_type + "_" + "ccs.csv")
                curr_df = pd.DataFrame.from_dict(
                    prov_ccs_cap_dict, orient="index", columns=[str(yr) + "_cap_mw"]).reset_index(drop=False)
                curr_df.columns = ["province", f"{yr}_cap_mw"]
                # Add capacity of 0 if province names are not part of the province column
                df_province = pd.read_csv(os.path.join(csv_folder, "geography", 'China_provinces_hz.csv'))
                for pro in list(df_province["provin_py"]):
                    if pro not in list(curr_df["province"]):
                        curr_df.loc[len(curr_df), ["province", f"{yr}_cap_mw"]] = pro, 0
                # Save csv results
                curr_df.to_csv(out_path, index=False)
                # track in a single dataframe for testing purposes
                prov_ccs_cap_df = pd.concat([prov_ccs_cap_df, curr_df], axis=1)

        # going through the years and modify unabated cap
        for yr in range(2020, self.yr_end + 1):
            if yr in yr_prov_dict_filtered:
                prov_dic = yr_prov_dict_filtered[yr]
                for prov_n_status, cap in prov_dic.items():
                    prov, status = prov_n_status
                    if status == "unabated":
                        # add to unabated dict
                        prov_res_cap_dict[prov] += cap
            if yr in self.yr_req:
                # write the file as csv files under each year folder
                out_path = os.path.join(self.out_path, str(yr),
                                        resource_type + "_" + "unabated.csv")
                curr_df = pd.DataFrame.from_dict(
                    prov_res_cap_dict, orient="index", columns=[str(yr) + "_cap_mw"]).reset_index(drop=False)
                curr_df.columns = ["province", f"{yr}_cap_mw"]
                # Add capacity of 0 if province names are not part of the province column
                df_province = pd.read_csv(os.path.join(csv_folder, "geography", 'China_provinces_hz.csv'))
                for pro in list(df_province["provin_py"]):
                    if pro not in list(curr_df["province"]):
                        curr_df.loc[len(curr_df), ["province", f"{yr}_cap_mw"]] = pro, 0
                # Save csv results
                curr_df.to_csv(out_path, index=False)

                # track in a single df for testing purposes
                prov_unabated_cap_df = pd.concat(
                    [prov_unabated_cap_df, curr_df], axis=1)

        # Merge coal/gas ccs csv files with coal/gas unabated csv files to obtain total capacity
        for yr in self.yr_req:
            if not os.path.exists(os.path.join(self.out_path, str(yr), f"{resource_type}_ccs.csv")):
                capacity_ccs = pd.DataFrame(columns=["province", f"{yr}_cap_mw"])
                out_path = os.path.join(self.out_path, str(yr), resource_type + "_" + "ccs.csv")
                # Add capacity of 0 if province names are not part of the province column
                df_province = pd.read_csv(os.path.join(csv_folder, "geography", 'China_provinces_hz.csv'))
                for pro in list(df_province["provin_py"]):
                    if pro not in list(capacity_ccs["province"]):
                        capacity_ccs.loc[len(capacity_ccs), ["province", f"{yr}_cap_mw"]] = pro, 0
                # Save csv results
                capacity_ccs.to_csv(out_path, index=False)
        for yr in self.yr_req:
            if os.path.exists(os.path.join(self.out_path, str(yr), f"{resource_type}_ccs.csv")):
                capacity_ccs = pd.read_csv(os.path.join(self.out_path, str(yr), f"{resource_type}_ccs.csv"))
                capacity_ccs.columns = ["province", "cap_ccs_mw"]
            else:
                capacity_ccs = pd.DataFrame(columns=["province", "cap_ccs_mw"])
            capacity_unabated = pd.read_csv(os.path.join(self.out_path, str(yr), f"{resource_type}_unabated.csv"))
            capacity_unabated.columns = ["province", "cap_unabated_mw"]
            capacity_total = pd.merge(capacity_unabated, capacity_ccs, on="province", how="left").fillna(0)
            capacity_total["cap_total_mw"] = capacity_total["cap_ccs_mw"] + capacity_total["cap_unabated_mw"]
            capacity_total = capacity_total[["province", "cap_total_mw", "cap_unabated_mw", "cap_ccs_mw"]]
            capacity_total.to_csv(os.path.join(self.out_path, str(yr), f"{resource_type}.csv"),
                                  header=True, index=False)

        return [prov_ccs_cap_df, prov_unabated_cap_df]

    def project_coal_n_coal_ccs(self, emission_target="2C"):
        # Scale 2020 provincial coal capacity
        for yr in self.yr_req:
            if self.run_mode == "ndc":
                coal_capacity = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions",
                                                         f"coal_{emission_target}_plan_ndc", f"coal_{yr}.csv"))
            else:
                coal_capacity = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions",
                                                         f"coal_{emission_target}_plan", f"coal_{yr}.csv"))
            coal_capacity = coal_capacity[["province", "cap_total_mw", "cap_unabated_mw", "cap_ccs_mw"]]
            coal_capacity.to_csv(os.path.join(self.out_path, str(yr), "coal.csv"),
                                 header=True, index=False)

    def project_gas_n_gas_ccs(self):
        # Read 2020 provincial gas capacity data
        df_gas_2020 = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "gas_2023.csv"),
                                  header=None)
        df_gas_2020.columns = ["province", "2020_cap_total_mw"]
        df_gas_2020["province_share"] = df_gas_2020["2020_cap_total_mw"] / df_gas_2020["2020_cap_total_mw"].sum()

        # Obtain nationwide gas and gas ccs capacity projections
        df_gas_total = pd.DataFrame()
        df_gas_total["Year"] = np.arange(2020, 2065, 5)
        df_gas_total["gas_ccs_cap"] = [0] * 4 + [np.nan] * 4 + [320000]
        df_gas_total["gas_ccs_cap"] = df_gas_total["gas_ccs_cap"].interpolate(method="linear")
        if self.run_mode == "ndc":
            df_gas_total["gas_total_cap_report"] = [df_gas_2020["2020_cap_total_mw"].sum(),
                                                    130000, 130000, np.nan, np.nan, np.nan, 330000, np.nan, 320000]
        else:
            df_gas_total["gas_total_cap_report"] = [df_gas_2020["2020_cap_total_mw"].sum(),
                                                    150000, 185000, np.nan, np.nan, np.nan, 330000, np.nan, 320000]
        df_gas_total["gas_total_cap_report"] = df_gas_total["gas_total_cap_report"].interpolate(method="linear")
        df_gas_total["gas_unabated"] = df_gas_total["gas_total_cap_report"] - df_gas_total["gas_ccs_cap"]

        # Scale 2020 provincial gas capacity
        for yr in self.yr_req:
            df_curr_yr_gas = df_gas_2020.copy(deep=True)
            df_curr_yr_gas_ccs = df_gas_2020.copy(deep=True)

            df_curr_yr_gas[f"{yr}_cap_mw"] = df_curr_yr_gas["province_share"] * \
                                             df_gas_total.loc[df_gas_total["Year"] == yr, "gas_unabated"].item()
            df_curr_yr_gas = df_curr_yr_gas[["province", f"{yr}_cap_mw"]]
            df_curr_yr_gas_ccs[f"{yr}_cap_mw"] = df_curr_yr_gas_ccs["province_share"] * \
                                                 df_gas_total.loc[df_gas_total["Year"] == yr, "gas_ccs_cap"].item()
            df_curr_yr_gas_ccs = df_curr_yr_gas_ccs[["province", f"{yr}_cap_mw"]]

            # # Save csv results
            # df_curr_yr_gas_ccs.to_csv(
            #     os.path.join(self.out_path, str(yr), "gas_ccs.csv"),
            #     index=False)
            # df_curr_yr_gas.to_csv(
            #     os.path.join(self.out_path, str(yr), "gas_unabated.csv"),
            #     index=False)

            # Merge unabated and ccs into one csv file
            df_curr_yr_gas_ccs.columns = ["province", "cap_ccs_mw"]
            df_curr_yr_gas.columns = ["province", "cap_unabated_mw"]
            capacity_total = pd.merge(df_curr_yr_gas, df_curr_yr_gas_ccs, on="province", how="left").fillna(0)
            capacity_total["cap_total_mw"] = capacity_total["cap_ccs_mw"] + capacity_total["cap_unabated_mw"]
            capacity_total = capacity_total[["province", "cap_total_mw", "cap_unabated_mw", "cap_ccs_mw"]].\
                sort_values(by="province", ascending=True).reset_index(drop=True)
            capacity_total.to_csv(os.path.join(self.out_path, str(yr), "gas.csv"),
                                  header=True, index=False)

    def project_bio_n_beccs(self, beccs_start_yr=2040):
        """
        Project provincial BECCS capacity. 

        Description
        -----------
        We assumed that BECCS capacity expansion decision starts at the year
        2040 (non-zero capacity). The user can change this assumption in the function.
        Capacity is projected linearly from BECCS_START_YR to 2060.

        Side effects
        ------------
        Creating a csv file called "beccs.csv" under each year folder under the
        out path specified by the user. 

        Returns
        -------
        pd.DataFrame
            Contains BECCS projections across requested years, starting from 
            the BECCS_START_YR. This is mostly for testing purposes

        """

        if self.emission_target == "15C":
            beccs_2060 = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "beccs_2060_15C.csv"),
                                     header=None)
        else:
            beccs_2060 = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "beccs_2060.csv"),
                                     header=None)
        beccs_2060.columns = ["province", "capacity_gw"]

        # requested year, but starts from BECCS_START_YR
        if self.yr_start > beccs_start_yr:
            year_ls = list(range(
                self.yr_start - self.yr_step,
                2060 + self.yr_step,
                self.yr_step))
        else:
            year_ls = list(range(
                beccs_start_yr - self.yr_step,
                2060 + self.yr_step,
                self.yr_step))

        # assembling series into a list 
        assembly_ls = []

        # Stack each row first
        for _, r in beccs_2060.iterrows():
            curr_prov = r["province"]
            curr_cap_2060 = r["capacity_gw"]

            curr_dict = {}
            curr_dict["prov"] = curr_prov

            # create projection for this province
            if curr_cap_2060 == 0:
                for yr in year_ls:
                    curr_dict[yr] = 0
            else:
                beccs_proj = np.linspace(0, curr_cap_2060, len(year_ls))

                # assemble curr dict from projection
                for i in range(beccs_proj.size):
                    curr_dict[year_ls[i]] = beccs_proj[i]

            assembly_ls.append(pd.Series(curr_dict))

        df_beccs_proj = pd.DataFrame(assembly_ls)

        for yr in self.yr_req:
            if yr in year_ls:
                df_curr_yr_beccs = df_beccs_proj.loc[:, ["prov", yr]]
                df_curr_yr_beccs.rename(columns={yr: f"{yr}_cap_gw",
                                                 "prov": "province"}, inplace=True)
            else:
                df_curr_yr_beccs = pd.DataFrame(columns=["province", f"{yr}_cap_gw"])

            # Add capacity of 0 if province names are not part of the province column
            df_province = pd.read_csv(os.path.join(csv_folder, "geography", 'China_provinces_hz.csv'))
            for pro in list(df_province["provin_py"]):
                if pro not in list(df_curr_yr_beccs["province"]):
                    df_curr_yr_beccs.loc[len(df_curr_yr_beccs), ["province", f"{yr}_cap_gw"]] = pro, 0

            # Save csv results
            df_curr_yr_beccs[f"{yr}_cap_mw"] = df_curr_yr_beccs[f"{yr}_cap_gw"] * 1000
            df_curr_yr_beccs[["province", f"{yr}_cap_mw"]].sort_values(by="province", ascending=True).\
                reset_index(drop=True).to_csv(os.path.join(self.out_path, str(yr), "beccs.csv"),index=False)

            # Copy-paste bio csv files to each year's folder
            if self.run_mode == "ndc":
                bio_pseudo = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "bio_ndc.csv"),
                                         header=None)
            else:
                bio_pseudo = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "bio.csv"),
                                         header=None)
            bio_pseudo.columns = ["province", f"{yr}_cap_mw"]
            bio_pseudo.sort_values(by="province", ascending=True).reset_index(drop=True)\
                .to_csv(os.path.join(self.out_path, str(yr), "bio.csv"), index=None)

        return df_beccs_proj

    def project_chp_n_chp_ccs(self, chp_ccs_start_yr=2040):
        CAP_COL_IDX = 1
        df_chp_ccs_2060 = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "chp_ccs_2060.csv"),
                                      header=None)
        df_chp_ccs_2060.columns = ["prov", "cap"]

        # at 2030, chp ccs starts from 0
        df_chp_ccs_2030 = df_chp_ccs_2060.copy()
        df_chp_ccs_2030.iloc[:, CAP_COL_IDX] = 0

        # Part 1: linear increases in CHP CCS capacity
        df_proj_chp_ccs = self.lin_project_prov_cap(
            chp_ccs_start_yr - self.yr_step,
            2060,
            df_chp_ccs_2030,
            df_chp_ccs_2060)

        # Part 2: interpolate total CHP and CHP CCS capacity
        chp_total_2020 = 498000  # MW
        df_project_chp_total_2020 = df_chp_ccs_2060.copy()
        df_project_chp_total_2020["cap"] = \
            df_project_chp_total_2020["cap"] / np.sum(df_project_chp_total_2020["cap"]) * chp_total_2020
        df_project_total = self.lin_project_prov_cap(
            2020,
            2060,
            df_project_chp_total_2020,
            df_chp_ccs_2060)

        # Part 3: deduct CHP CCS capacity from total capacity to obtain CHP capacity
        df_proj_chp = df_proj_chp_ccs.copy(deep=True)
        for yr in self.yr_req:
            if yr not in df_proj_chp_ccs.columns:
                df_proj_chp_ccs[yr] = 0
            df_proj_chp[yr] = df_project_total[yr] - df_proj_chp_ccs[yr]

        # requested year, but starts from CHP_CCS_START_YR
        if self.yr_start > chp_ccs_start_yr:
            year_ls = list(range(
                self.yr_start,
                2060 + self.yr_step,
                self.yr_step))
        else:
            year_ls = list(range(
                chp_ccs_start_yr,
                2060 + self.yr_step,
                self.yr_step))

        for yr in self.yr_req:
            df_curr_yr_chp = df_proj_chp.loc[:, ["prov", yr]]
            df_curr_yr_chp.rename(columns={yr: f"{yr}_cap_mw", "prov": "province"}, inplace=True)
            if yr in year_ls:
                df_curr_yr_chp_ccs = df_proj_chp_ccs.loc[:, ["prov", yr]]
                df_curr_yr_chp_ccs.rename(columns={yr: f"{yr}_cap_mw",
                                                   "prov": "province"}, inplace=True)
            else:
                df_curr_yr_chp_ccs = pd.DataFrame(columns=["province", f"{yr}_cap_mw"])

            # Add capacity of 0 if province names are not part of the province column
            df_province = pd.read_csv(os.path.join(csv_folder, "geography", 'China_provinces_hz.csv'))
            for pro in list(df_province["provin_py"]):
                if pro not in list(df_curr_yr_chp_ccs["province"]):
                    df_curr_yr_chp_ccs.loc[len(df_curr_yr_chp_ccs), ["province", f"{yr}_cap_mw"]] = pro, 0
                if pro not in list(df_curr_yr_chp["province"]):
                    df_curr_yr_chp.loc[len(df_curr_yr_chp), ["province", f"{yr}_cap_mw"]] = pro, 0

            # Save csv results
            df_curr_yr_chp_ccs.sort_values(by="province", ascending=True).reset_index(drop=True).to_csv(
                os.path.join(self.out_path, str(yr), "chp_ccs.csv"),
                index=False)
            # df_curr_yr_chp.to_csv(
            #     os.path.join(self.out_path, str(yr), "chp.csv"),
            #     index=False)

        return df_proj_chp_ccs

    def initialize_inputs_and_outputs(self):
        for yr in self.yr_req:
            # Create a folder for optimization inputs, outputs and processed outputs
            out_input_path = os.path.join(self.out_path, str(yr), "inputs")
            os.mkdir(out_input_path)
            out_output_path = os.path.join(self.out_path, str(yr), "outputs")
            os.mkdir(out_output_path)
            out_output_processed_path = os.path.join(self.out_path, str(yr), "outputs_processed")
            os.mkdir(out_output_processed_path)
            out_output_processed_graph_path = os.path.join(self.out_path, str(yr), "outputs_processed", "graphs")
            os.mkdir(out_output_processed_graph_path)

    def automate_inputs(self):
        """
        The 'main' function, running separate projection methods.
        """
        print("Projecting demand...")
        self.demand_projection()

        print("Projecting costs...")
        self.cost_projection()

        print("Projecting hydro capacity...")
        self.project_prov_hydro()
        print("Projecting nuclear capacity...")
        self.project_prov_nuclear()
        print("Projecting beccs capacity")
        self.project_bio_n_beccs()

        print("Projecting gas unabated and ccs capacity...")
        self.project_gas_n_gas_ccs()

        print("Projecting coal unabated and ccs capacity...")
        self.project_coal_n_coal_ccs(emission_target="2C")
        print("Projecting coal chp and coal chp ccs capacity")
        self.project_chp_n_chp_ccs()

        print("Initializing inputs and outputs folders")
        self.initialize_inputs_and_outputs()

    ############################# Helper Function ################################

    def project_scales(self, end_scale_fac: float) -> dict:
        """
        Linearly interpolate scaling factoring into the future, given the end 
        scaling factor.

        Parameters
        ----------
        end_scale_fac: float
            The scaling factor for the last year. 

        Return
        -------
        dict
            a dictionary whose key are years and values are the scaling factors
            for these years.

        Example
        -------
        Say the yr_start = 2020, yr_end = 2060, yr_step = 5,

        project_scales(2) will return the following dictionary:

            {2020: 1.0, 2030: 1.25, 2040: 1.5, 2050: 1.75, 2060: 2.0}

        """

        # homogenous year step. Find the greatest common divider between
        # the difference between start year and 2020 and the year step that
        # the user passed in.
        hom_yr_step = math.gcd(self.yr_start - 2020, self.yr_step)
        full_ls = list(np.linspace(
            start=2020,
            stop=self.yr_end,
            num=int((self.yr_end - 2020) / hom_yr_step) + 1,
            dtype=int
        )
        )
        scale_ls = list(
            np.linspace(
                start=1,
                stop=end_scale_fac,
                num=len(full_ls),
                dtype=float
            )
        )
        full_dict = dict(zip(full_ls, scale_ls))

        filtered_dict = {yr: scale for (yr, scale) in full_dict.items() if yr in self.yr_req}

        return filtered_dict

    def lin_project_prov_cap(self, start_yr, end_yr, df_start_cap, df_end_cap):
        """
        A helper function that project provincial capacity given the start line 
        capacity DataFrame and the endline capacity DataFrame.

        Parameters 
        ----------
        start_yr: int 
            start line year.
        end_yr: int
            end line year.
        df_start_cap: pd.DataFrame
            DataFrame containing provincial capacity at the start line year. 
            The DataFrame should only have two columns: the first column 
            is the province name, and the second column is the capacity. 
            Make sure that the unit for start and end DataFrame matches.
        df_end_cap: pd.DataFrame
            DataFrame containing provincial capacity at the end line year.
            The DataFrame should only have two columns: the first column 
            is the province name, and the second column is the capacity. 
            Make sure that the unit for start and end DataFrame matches.

        Returns
        ------
        pd.DataFrame
            A DataFrame describing provincial projection at each province for 
            all the years between `start_yr` and `end_yr`, with a step size of 
            `self.yr_step`. 
        """

        year_ls = list(range(start_yr, end_yr + self.yr_step, self.yr_step))
        assembly_ls = []

        # check if start and end df have same provincial names
        PROV_COL_IDX = 0
        CAP_COL_IDX = 1
        start_prov_set = set(df_start_cap.iloc[:, PROV_COL_IDX])
        end_prov_set = set(df_end_cap.iloc[:, PROV_COL_IDX])
        if not start_prov_set == end_prov_set:
            raise ValueError("df_start_cap and df_end_cap have different \
            provinces listed.")

        # reset names for both df 
        df_start_cap.columns = ["prov", "cap"]
        df_end_cap.columns = ["prov", "cap"]

        df_start_cap.set_index("prov", inplace=True)

        # stack up each row first
        for _, r in df_end_cap.iterrows():
            curr_prov = r[PROV_COL_IDX]
            curr_cap_start = df_start_cap.loc[curr_prov, "cap"]
            curr_cap_end = r[CAP_COL_IDX]

            curr_dict = {}
            curr_dict["prov"] = curr_prov

            # create projection fro this province
            cap_proj = np.linspace(curr_cap_start, curr_cap_end, len(year_ls))

            # assemble curr dict from projection
            for i in range(cap_proj.size):
                curr_dict[year_ls[i]] = cap_proj[i]

            assembly_ls.append(pd.Series(curr_dict))

        df_cap_proj = pd.DataFrame(assembly_ls)
        return df_cap_proj

    def get_prev_year(self, curr_year):
        for i in range(len(self.yr_req)):
            if self.yr_req[i] == curr_year:
                prev_year = self.yr_req[i - 1]
                break
        return prev_year

    ####################### Pre-processing Code #####################

    def vre_cap_pre_processing(self, resource, curr_year):
        """
        Variable renewable energy capacity post processing.

        This function takes the vre cap model result and transform it into 
        a form that can be taken in as model input for the next model run. 

        Arguments
        ---------
        resource: str
            Either "solar" or "wind", representing which vre requires post 
            processing. 
        curr_year: int
            The year for which the model output is used. Note this is NOT the 
            year where the model result comes from. 

        Side Effect
        -----------
        Create the correctly formatted vre cap file that can be directly used 
        as a model input.

        """

        # find the previous year, load the result from previous year folder
        # xxx_info.csv -- wind_info.csv or solar_info.csv, which store model 
        # results about vre
        prev_year = self.get_prev_year(curr_year)
        past_cap = pd.read_csv(os.path.join(self.out_path, str(prev_year), resource + "_info.csv"), header=None)

        GW2MW_FAC = 1000

        d = {"lon": past_cap.iloc[:, 2],  # col2 - lon
             "lat": past_cap.iloc[:, 3],  # col3 - lat
             # % existing cap * vre potential (GW) * GW2MW_FAC
             "existing_cap_mw": past_cap.iloc[:, 1] * past_cap.iloc[:, 4] * GW2MW_FAC}

        processed_df = pd.DataFrame(data=d)

        processed_df.to_csv(os.path.join(self.out_path, str(
            curr_year), "integrated_" + resource + ".csv"), header=False, index=False)

    def storage_pre_processing(self, curr_year):
        """
        Convert energy storage modeling result to input for the next time period

        This function modifies the `model_exovar.pkl` by combining model result
        from the previous time period. 

        Arguments
        ---------
        curr_year: int 
            
        """
        pkl_path = os.path.join(work_dir, "data_pkl", "model_exovar.pkl")
        res_path = os.path.join(self.out_path, str(curr_year),
                                "model_exovar.pkl")

        # if first year, copy file from data_pkl
        if curr_year == self.yr_start:
            shutil.copy2(pkl_path, res_path)

        # otherwise, combine exovar from previous year and model result from 
        # previous year
        else:
            prev_year = self.get_prev_year(curr_year)
            prev_dir = os.path.join(self.out_path, str(prev_year),
                                    "model_exovar.pkl")
            old_exo_var = pd.read_pickle(prev_dir)

            prev_phs_rst = pd.read_csv(os.path.join(self.out_path,
                                                    str(prev_year), "es_phs_cap.csv"))
            # convert dataframe to dictionary 
            phs_lbd_dict = dict(zip(prev_phs_rst.iloc[0, :],
                                    prev_phs_rst.iloc[1, :]))

            old_exo_var["phs_lb"] = phs_lbd_dict

            # Save the pickled dictionary to the current directory.
            with open(res_path, "wb") as old_exo:
                pickle.dump(old_exo_var, old_exo)


if __name__ == "__main__":
    # Five-year interval example
    test_multi_input = MultiYearAutomation(yr_start=2025,
                                           yr_end=2060,
                                           yr_step=5,
                                           res_tag="test_0629",
                                           vre_year="w2015_s2015",
                                           emission_target="2C",
                                           demand_sensitivity="none",
                                           run_mode="ndc"
                                           )

    test_multi_input.automate_inputs()
