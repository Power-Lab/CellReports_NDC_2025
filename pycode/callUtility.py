import platform
import os
import math
from pyparsing import Word
import pypinyin
import difflib
import Levenshtein
from shapely.geometry import Point
import geopandas as gpd


#print(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def dirFlag():
    '''
    Determine the path separator based on the system from which the 
    function is called
    '''
    flag = platform.system()
    if 'Darwin' in flag or 'Linux' in flag:
        return '/'
    else:
        return '\\'

def makeDir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    return path

def getWorkDir():
    '''
    Get the current working directory -- where the file is located in the 
    system 
    '''
    dir_flag = dirFlag()
    work_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + dir_flag
    return work_dir

def getResDir(vre_year,res_tag):
    '''
    Construct the resource directory that incorporates the year of the 
    renewable energy and the tag of the renewable energy
    '''
    dir_flag = dirFlag()
    res_dir = getWorkDir()+'data_res'+dir_flag+str(res_tag)+'_'+str(vre_year)
    return res_dir

def str2int(data):
    for i in range(len(data)):
        if data[i] != '0':
            break
    data = eval(data[i:])
    return data

def maxINT(num):
    num = int(num)
    strNum = str(num)
    return (int(strNum[0]) + 1) * pow(10,len(strNum)-1) 
    #return (num + 100)

def list2dict(data):
    d = {}
    for i in range(len(data)):
        if(len(data[0]) == 1):
            d[i] = float(data[i])
            d[i] = round(d[i],2)
        else:
            for j in range(len(data[0])):
                t = tuple([i,j])
                d[t] = float(data[i][j])
                d[t] = round(d[t],2)
    return d

def str2eval(data):
    for i in range(len(data)):
        data[i] = eval(data[i])
    return data

def firsttNonZero(data):
    key = 0
    for i in range(len(data)):
        if data[i] != 0:
            key = i
            break
    return key

def line2list(line):
    line = line.replace('\n', '')
    line = line.split(',')
    line = str2eval(line)
    return line

def degree_to_radian(degree):
    return degree * math.pi / 180

def radian_to_degree(radian):
    return radian * 180.0 / math.pi

def haver_sin(theta):
    v = math.sin(theta/2)
    return v * v

def geo_distance(lat1,lon1,lat2,lon2):
    earth_radius = 6371.0
    lat1 = degree_to_radian(lat1)
    lon1 = degree_to_radian(lon1)
    lat2 = degree_to_radian(lat2)
    lon2 = degree_to_radian(lon2)

    var_lat = math.fabs(lat1-lat2)
    var_lon = math.fabs(lon1-lon2)
    h = haver_sin(var_lat) + math.cos(lat1) * math.cos(lat2) * haver_sin(var_lon)
    distance = 2 * earth_radius * math.asin(math.sqrt(h))
    
    return distance

def pinyin(word):
    s = ''
    pinyinlist = pypinyin.pinyin(word, style=pypinyin.NORMAL)
    for i in pinyinlist:
        s += ''.join(i)
    return s

def similarity_le(word1,word2):
    return Levenshtein.ratio(word1,word2)

def similarity_di(word1,word2):
    return difflib.SequenceMatcher(None,word1,word2).quick_ratio()

def similarity(word1,word2):
    return similarity_le(word1,word2)

def getBound():
    bound = gpd.GeoDataFrame({
    'x': [68, 140, 106.5, 123],
    'y': [15, 55, 2.8, 24.5]
    })
    # 添加矢量列
    bound.geometry = bound.apply(lambda row: Point([row['x'], row['y']]), axis=1)
    # 初始化CRS
    bound.crs = 'EPSG:4326'
    return bound

def getProvinName():
    work_dir = getWorkDir()
    dir_flag = dirFlag()
    provins = []
    f_provins = open(work_dir+'data_csv'+dir_flag+'geography/China_provinces_hz.csv','r+')
    next(f_provins)
    for line in f_provins:
        line = line.replace('\n','')
        line = line.split(',')
        provins.append(line[1])
    f_provins.close()

    return provins

def getCRF(wacc,years):
    '''
    Get capital recovery factor
    '''
    wacc = 0.01 * wacc
    crf = (wacc * math.pow(1+wacc,years)) / (math.pow(1+wacc,years)-1)

    return round(crf,3)

def getRamp(load,hour_seed,hour_pre):
    ramp_up = 0
    ramp_down = 0

    for h in hour_seed[1:]:
        if load[h] > load[hour_pre[h]]:
            ramp_up += (load[h]-load[hour_pre[h]])
        elif load[h] < load[hour_pre[h]]:
            ramp_down += (load[hour_pre[h]]-load[h])
    
    return [ramp_up,ramp_down]

def extractRegionName(name):
    name = name.replace(' Sheng','')
    name = name.replace(' Shi','')
    name = name.replace(' Zizhiqu','')
    name = name.replace('uygur','')
    name = name.replace('huizu','')
    name = name.replace('zhuangzu','')

    return name

def extractProvinceName(name):
    if name == '内蒙古东':
        name_py = 'EastInnerMongolia'
    elif name == '内蒙古西':
        name_py = 'WestInnerMongolia'
    elif name == '陕西':
        name_py = 'Shaanxi'
    else:
        name_py = pinyin(name)
        name_py = name_py.capitalize()

    return name_py

def GetHourSeed(vreYear,resTag):
    """
    HourSeed -- the hours of a year that the optimization covers. 
    """
    res_dir = getResDir(vreYear,resTag)+dirFlag()
    hour_seed = [] 
    
    f_hour_seed = open(res_dir+'hour_seed.csv','r')
    for  line in f_hour_seed:
        line = line.replace('\n','')
        line = eval(line)
        hour_seed.append(line)
    f_hour_seed.close()

    return hour_seed

def VreYearSplit(vreYear):
    '''
    Vre comes in with the format of ryyyy_ryyyy (for example, w2015_s2015).
    where r represents the type of resources (w for wind and s for solar). 
    yyyy is the year from which the input solar or wind generation capacity data
    derived. This function return the years in a list format. 
    
    '''
    vreYear = vreYear.split('_')

    return [vreYear[0][1:],vreYear[1][1:]]


def SplitMultipleVreYear(vreYear):
    if vreYear[0] == '2':
        return [vreYear]
    else:
        vreYears = []

        for i in range(0,len(vreYear),2):
            vreYears.append('20'+vreYear[i:i+2])

        return vreYears


def GetWinterHour():
    winter_hour = []
    f_wh = open(getWorkDir()+'data_csv'+dirFlag()+'winter_hour.csv','r+')

    for line in f_wh:
        line = line.replace('\n','')
        winter_hour.append(eval(line))

    f_wh.close()

    return winter_hour


if __name__ == '__main__':

    print((15700*getCRF(7.4,35)+750))
