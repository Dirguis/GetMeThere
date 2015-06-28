#!/usr/bin/python -tt

#Author: Damien Forthomme
#Takes inputs from the front end and generates the predictions

import urllib2
import simplejson as json
import pandas as pd
import numpy as np
import sys
import time
import datetime
from geopy.distance import vincenty
from shapely.geometry import Point, Polygon
import csv
from sklearn.externals import joblib
import copy as cp
import re
import pickle
import os
import pytz


#Define the polygons to get the various zones
#in NYC
def findZone(PtPoint):
    
    UpperManhattanGPS  = ((40.755277, -73.952954), (40.774002, -73.997360), (40.879312, -73.933256), (40.872627, -73.909309), (40.834518, -73.934200), (40.809057, -73.933514), (40.801521, -73.927849), (40.794244, -73.910597), (40.778322, -73.928793), (40.779167, -73.938148), (40.770717, -73.938663))
    UppManPoly    = Polygon(UpperManhattanGPS)
    MidManhattanGPS    = ((40.740256, -73.966375), (40.758657, -74.009762), (40.774002, -73.997360), (40.755277, -73.952954))
    MidManPoly    = Polygon(MidManhattanGPS)
    MidLowManhattanGPS = ((40.725996, -73.968034), (40.745312, -74.014983), (40.758657, -74.009762), (40.740256, -73.966375))
    MidLowManPoly = Polygon(MidLowManhattanGPS)
    LowManhattanGPS    = ((40.710188, -73.975587), (40.695125, -74.027729), (40.745312, -74.014983), (40.725996, -73.968034))
    LowManPoly    = Polygon(LowManhattanGPS)
    BronxGPS           = ((40.917345, -73.916621), (40.877023, -73.774013), (40.804558, -73.776588), (40.794244, -73.910597), (40.801521, -73.927849), (40.809057, -73.933514), (40.834518, -73.934200), (40.872627, -73.909309), (40.879312, -73.933256))
    BronxPoly     = Polygon(BronxGPS)
    BrooklynGPS        = ((40.535302, -74.069614), (40.583035, -73.863277), (40.625262, -73.887653), (40.642979, -73.853321), (40.693758, -73.867054), (40.679700, -73.894863), (40.727850, -73.929195), (40.730652, -73.938682), (40.735725, -73.942115), (40.739562, -73.955934), (40.734099, -73.968036), (40.725996, -73.968034), (40.710188, -73.975587), (40.695125, -74.027729))
    BrooklynPoly  = Polygon(BrooklynGPS)
    QueensGPS          = ((40.804558, -73.776588), (40.730364, -73.694829), (40.723339, -73.729505), (40.593637, -73.742208), (40.536258, -73.943395), (40.553216, -73.953351), (40.616575, -73.776197), (40.642979, -73.853321), (40.693758, -73.867054), (40.679700, -73.894863), (40.727850, -73.929195), (40.730652, -73.938682), (40.735725, -73.942115), (40.739562, -73.955934), (40.734099, -73.968036), (40.740256, -73.966375), (40.755277, -73.952954), (40.770717, -73.938663), (40.779167, -73.938148), (40.778322, -73.928793), (40.794244, -73.910597))
    QueensPoly    = Polygon(QueensGPS)
    
    if PtPoint.within(MidManPoly):
        initZone = '0'  #MidMan
    elif PtPoint.within(MidLowManPoly):
        initZone = '1'  #MidLowMan
    elif PtPoint.within(LowManPoly):
        initZone = '2'  #LowMan
    elif PtPoint.within(UppManPoly):
        initZone = '3'  #UppMan
    elif PtPoint.within(BronxPoly):
        initZone = '4'  #Bronx
    elif PtPoint.within(BrooklynPoly):
        initZone = '5'  #Brook'
    elif PtPoint.within(QueensPoly):
        initZone = '6'  #Queens
    else:
        initZone = 'NaN'
    return initZone
  

#When given an address, format it to Google readable address
def splitAddress(address):
    split_address = address.split(' ')
    concatSplit_address = ''
    for word in split_address:
        concatSplit_address += word + r'+'
    concatSplit_address = concatSplit_address[:-1]
    return concatSplit_address

#Round time to half hours (see code futher down to round to hours or half hours)
def roundTime(dt=None, roundTo=60):
    """Round a datetime object to any time laps in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt == None : dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

#Tokenize the hours bases on what half hour bin they fall into
def tokenizeHours(strHour):
    return str(int(strHour[:2])*2 + int(strHour[3:-1])/3)

#Dictionnary of average speeds for all zone combinations
#The files are refered to by the pick up zone, the drop off zone and the
#day of the week
def dicName(x, AvgSpeedsDic):
    fileName = 'avgSpeedZone' + str(int(x.PickZone)) + str(int(x.DropZone)) + str(int(x.dayInWeek))
    return AvgSpeedsDic[fileName][x.pickT]


#Get info weather now and check for freezing, rain or snow
#Get useful URLs too
def GetWeatherNow(featurePd):
    apiWU = ''
    urlWU = 'http://api.wunderground.com/api/' + apiWU + '/geolookup/conditions/q/NY/New_York.json'
    f = urllib2.urlopen(urlWU)
    json_string = f.read()
    f.close()
    parsed_json = json.loads(json_string)
    if int(parsed_json['current_observation']['temp_f']) < 32:
        featurePd['freezing'] = 1
    if float(parsed_json['current_observation']['precip_1hr_in']) > 0.0:
        featurePd['rain']  = 1
    if re.match(parsed_json['current_observation']['weather'].lower(),'snow'):
        featurePd['snow']     = 1
    if re.match(parsed_json['current_observation']['weather'].lower(),'rain'):
        featurePd['rain']  = 1
        
    WUInfo = {}
    WUInfo['WULink2Forecast'] = parsed_json['current_observation']['forecast_url']
    WUInfo['WULink2Image']    = parsed_json['current_observation']['icon_url']
    WUInfo['tempf']           = parsed_json['current_observation']['temp_f']
    WUInfo['weather']         = parsed_json['current_observation']['weather']
        
    #featurePd['rain'] = 0
    #featurePd['snow'] = 0
    #featurePd['freezing'] = 0
    return featurePd, WUInfo


#Get info weather in the future and check for freezing, rain or snow
#Get useful URLs too
def GetFutureWeather(featurePd, hoursDiff):
    apiWU = ''
    urlWU = 'http://api.wunderground.com/api/' + apiWU + '/hourly/q/NY/New_York.json'
    f = urllib2.urlopen(urlWU)
    json_string = f.read()
    f.close()
    parsed_json = json.loads(json_string)
    #increment to get the next hour
    #get the nooa data seta and use this one to get the forecast in terms of snow, rain and temp
    #parsed_json['hourly_forecast'][hoursDiff]['FCTTIME']['hour']
    if int(parsed_json['hourly_forecast'][hoursDiff]['temp']['english']) < 32:
        featurePd['freezing'] = 1
    if float(parsed_json['hourly_forecast'][hoursDiff]['snow']['english'])>0:
        featurePd['snow']     = 1
    if float(parsed_json['hourly_forecast'][hoursDiff]['qpf']['english'])>0:
        featurePd['rain']  = 1
    
    WUInfo = {}
    WUInfo['WULink2Forecast'] = 'http://www.wunderground.com/weather-forecast/zmw:10001.5.99999?MR=1'
    WUInfo['WULink2Image']    = parsed_json['hourly_forecast'][hoursDiff]['icon_url']
    WUInfo['tempf']           = parsed_json['hourly_forecast'][hoursDiff]['temp']['english']
    WUInfo['weather']         = parsed_json['hourly_forecast'][hoursDiff]['condition']
    #print hoursDiff
    #featurePd['rain'] = 0
    #featurePd['snow'] = 0
    #featurePd['freezing'] = 0
    return featurePd, WUInfo
  
  
#Process information given by user
def inputInfo(origin, destination, featurePd, *arg):
    
    returnDic = {}
    nbPeople = '1'
    departTime = ''
    #Check for extra optional arguments
    #If there are differentiate between 
    #nbPeople and departTime based on
    #the size
    if arg:
        argLengthArgument = [len(x) for x in arg]
        inc = 0
        for Largument in argLengthArgument:
            if (Largument < 3) & (Largument > 0):
                nbPeople = arg[inc]
            elif (Largument > 3):
                departTime = arg[inc]
            inc += 1
            
    #Time given on AWS machine is UTC. Need to offset time to be consistent with Eastern time.
    utcTime     = datetime.datetime.now(pytz.timezone('UTC'))
    easternTime = datetime.datetime.now(pytz.timezone('US/Eastern'))
    deltaHours = int(easternTime.hour - utcTime.hour)
    #print deltaHours
    if deltaHours > 0:
        deltaHours -= 24
    departTimePOSIXstr = ''
    
    #WUInfo is a dic that gathers all the info related to weather
    WUInfo = {}
    #handle case when the time of day is lower than current time. Assume the person wants the next day
    if departTime:
        returnDic['departTime'] = departTime
        currentTime          = datetime.datetime.now() + datetime.timedelta(hours=deltaHours)
        #print currentTime
        departDateTime       = datetime.datetime.strptime(departTime, '%H:%M')
        if departDateTime.hour < currentTime.hour:
            departTimeObj    = datetime.datetime(currentTime.year,currentTime.month,currentTime.day+1, departDateTime.hour, departDateTime.minute)        
        else:
            departTimeObj    = datetime.datetime(currentTime.year,currentTime.month,currentTime.day, departDateTime.hour, departDateTime.minute)
        #Get the hour difference between now and the depart time to assess weather conditions
        timeDiff             = (departTimeObj - currentTime)
        hoursDiff, remainder = divmod(timeDiff.seconds, 3600)

        #Get weather info and process the time
        featurePd, WUInfo    = GetFutureWeather(featurePd, hoursDiff)
        Htoken               = tokenizeHours(str(roundTime(departTimeObj, roundTo = 30*60))[11:16]) 
        departTimePOSIXstr   = str(round(time.mktime((departTimeObj-datetime.timedelta(hours=deltaHours)).timetuple())))
    else:
        departTimeObj        = datetime.datetime.now()
        Htoken               = tokenizeHours(str(roundTime(departTimeObj, roundTo = 30*60))[11:16])
        returnDic['departTime']        = 'now'
        
    departTimePOSIXstr = departTimePOSIXstr[:-2]
    
    #start building the pandas dataframe of features
    featurePd['hour'+Htoken] = 1
    featurePd['pickT'] = int(Htoken)
        
    
    dayOfWeek = departTimeObj.weekday()
    #print dayOfWeek
    if dayOfWeek < 5:
        dayInWeek = '0'   #week day
        featurePd['WD0'] = 1
    else:
        dayInWeek = '1'   #weekend
        featurePd['WD1'] = 1
        
        
    #Make sure input values are realistic
    try:
        if int(nbPeople) > 6:
            print 'invalid entry, the number of people should be less 1-6'
            stop()
    except:
        Error = 'Too many people!'
        return Error, 1
    try:
        if int(nbPeople) < 1:
            print 'invalid entry, the number of people should be less 1-6'
            stop()
    except:
        Error = 'Not enough people!'
        return Error, 1
        
     
    featurePd['Pcount'] = int(nbPeople)
    featurePd['PCount'+nbPeople] = 1
    

    
    #Get Google info starting with the subway
    apiKey = ''
    originSplit = splitAddress(origin)
    destinationSplit = splitAddress(destination)
    httpReq = 'https://maps.googleapis.com/maps/api/directions/json?origin='
    httpReq = httpReq + originSplit + '&' + 'destination=' + destinationSplit
    if departTimePOSIXstr:
        httpReq = httpReq + '&departure_time=' + \
        departTimePOSIXstr + '&mode=transit&transit_mode=subway&key=' + apiKey
    else:
        httpReq = httpReq + '&mode=transit&transit_mode=subway&key=' + apiKey
        
    GoogleMapsAddress = 'https://www.google.com/maps?saddr=' + originSplit + '&daddr=' + destinationSplit + '&mode=transit&transit_mode=subway'
    
    try:
        f = urllib2.urlopen(httpReq)
        json_string = f.read()
        f.close()
    except:
        Error = 'error with fetching info online: are you connected to internet?'
        return Error, 1
    
    try:
        Res = json.loads(json_string)
        MetroWalkTripTime = Res['routes'][0]['legs'][0]['duration']['text']
        TripDistance_directions = Res['routes'][0]['legs'][0]['distance']['text']
    except:
        Error = 'address must be within NYC!'
        return Error, 1
    
    WalkTime = 0
    for inc in range(len(Res['routes'][0]['legs'][0]['steps'])):
        if Res['routes'][0]['legs'][0]['steps'][inc]['travel_mode'] == 'WALKING':
            WalkTime += Res['routes'][0]['legs'][0]['steps'][inc]['duration']['value']
    WalkTime = int(round(WalkTime / 60))
    
    
    #Get the full addresses and the coordinates
    addressList  = []
    locationList = []
    addressList.append(Res['routes'][0]['legs'][0]['start_address'])
    locationList.append(Res['routes'][0]['legs'][0]['start_location'])
    addressList.append(Res['routes'][0]['legs'][0]['end_address'])
    locationList.append(Res['routes'][0]['legs'][0]['end_location'])
    
    
    #Get the route distance and durations if the time is asked for now
    httpReq = 'https://maps.googleapis.com/maps/api/distancematrix/json?origins=' + originSplit + '&'
    if departTimePOSIXstr:
        httpReq = httpReq + 'destinations=' + destinationSplit + '&departure_time=' + \
        departTimePOSIXstr + '&mode=driving&key=' + apiKey
    else:
        httpReq = httpReq + 'destinations=' + destinationSplit + '&mode=driving&key=' + apiKey
    
    try:
        f = urllib2.urlopen(httpReq)
        json_string = f.read()
        f.close()
    except:
        Error = 'error with fetching info online: are you connected to internet?'
        return Error, 1
        
    Res = json.loads(json_string)
    tripDist = Res['rows'][0]['elements'][0]['distance']['value']/1.60934/1000
    
    featurePd['tripT']    = Res['rows'][0]['elements'][0]['duration']['value']
    featurePd['tripDist'] = Res['rows'][0]['elements'][0]['distance']['value']/1.60934/1000

    
    #Fill returnDic, dictionnary with carries all the info I need
    returnDic['featurePd']         = featurePd
    returnDic['MetroWalkTripTime'] = MetroWalkTripTime
    returnDic['tripDist']          = tripDist
    returnDic['locationList']      = locationList
    returnDic['addressList']       = addressList
    returnDic['nbPeople']          = nbPeople
    returnDic['WUInfo']            = WUInfo
    returnDic['WalkTime']          = WalkTime
    returnDic['GoogleMapsAddress'] = GoogleMapsAddress

    returnList = [featurePd, MetroWalkTripTime, tripDist, locationList, addressList, nbPeople, departTime]
    
    #Note that if there is an error with the try/except, the function returns a string with the 
    #explanation of the mistake and the value 1 instead of returnDic and 0
    return returnDic, 0
  
  
def makeInputVector(featurePd, inputDic):
    
    #Polygons to define LaGuardia and JFK
    LGPoly  = Polygon(((40.764, -73.89), (40.783, -73.89), (40.783, -73.854), (40.764, -73.854)))
    JFKPoly = Polygon(((40.673416, -73.795706), (40.649716, -73.836905), (40.614542, -73.768240), (40.638254, -73.728758)))
    
    #Get back the coordinates and trip distance values
    tripDist = inputDic['tripDist']
    pickLong = inputDic['locationList'][0]['lng']
    pickLat  = inputDic['locationList'][0]['lat']
    dropLong = inputDic['locationList'][1]['lng']
    dropLat  = inputDic['locationList'][1]['lat']
    
    
    
    #check that the coordinates are meaningful
    if (pickLong == 0) | (pickLat == 0) | (dropLong == 0) | (dropLat == 0):
        print 'invalid entry, coordinates = 0'
        stop()
        
    #make sure that the coordinates are within the correct boundaries
    try:
        if (pickLong < -74.02) | (pickLong > -73.76) | (pickLat < 40.57) | (pickLat > 40.9):
            stop()
        if (dropLong < -74.02) | (dropLong > -73.76) | (dropLat < 40.57) | (dropLat > 40.9):
            stop()
    except:
        Error = 'addresses must be within NYC!'
        return Error, 0
        
        
       
    #Check if LaGuardia or JFK are part of the trip
    PtPick = Point(pickLat, pickLong)
    PtDrop = Point(dropLat, dropLong)
    if PtPick.within(LGPoly) | PtDrop.within(LGPoly):
        rate = 6
    if PtPick.within(JFKPoly) | PtDrop.within(JFKPoly):
        rate = 2
    else:
        rate = 1
        
    featurePd['rate'] = rate
    
    #Get the zone to get the proper averagespeeds and classification
    PickZone = findZone(PtPick)
    DropZone = findZone(PtDrop)
    
    #print PickZone
    inputDic['JFKZone'] = 0
    
    #Check if flat rate applies going to or from JFK
    if (rate == 2):
        if (PickZone == '0') | (PickZone == '1') | (PickZone == '2') | (PickZone == '3'):
            inputDic['JFKZone'] = '7'
        if (DropZone == '0') | (DropZone == '1') | (DropZone == '2') | (DropZone == '3'):
            inputDic['JFKZone'] = '7'
    
    if (PickZone == 'NaN') | (DropZone == 'NaN'):
        print 'error, no data for the addresses specified. Must be in NYC'
        stop()
        
        
    
    featurePd['PickZone'] = int(PickZone)
    featurePd['DropZone'] = int(DropZone)
    featurePd['Pzone'+str(PickZone)] = 1
    featurePd['Dzone'+str(DropZone)] = 1
    
    
    #Note that if there is an error with the try/except, the function returns a string with the 
    #explanation of the mistake and the value 1 instead of inputDic and featuredPd
    return inputDic, featurePd

  
  
def PredictItinary(origin, destination, nbPeople, departTime):
    
    #returnDic is the dictionnary that contains everything useful
    
    #Get common path
    commonDir                    = os.path.abspath('.')
    featurePd                    = pd.read_csv(commonDir + '/app/LoadFiles/featureNames.csv')
    returnDic, flag              = inputInfo(origin, destination, featurePd, nbPeople, departTime)

    if flag == 1:
        return returnDic, 1
        
    featurePd                    = returnDic['featurePd']
    locationList                 = returnDic['locationList']
    returnDic, featurePd         = makeInputVector(featurePd, returnDic)

    try: 
        featurePd.shape[1]
    except:
        return returnDic, 1
    
    
    returnDic['cabTimeStr']      = 'Trip time: ' + str(int(round(featurePd['tripT']/60.0))) + ' mins'
    
    #If time in the future, predict it and then use it to predict the fare
    if not (departTime == 'now'):
        #Load the average speeds between various zones
        with open(commonDir + '/app/LoadFiles/AvgSpeedsDic.txt', 'rb') as AutoPickleFile:
            AvgSpeedsDic = pickle.load(AutoPickleFile)

        featurePdTmp = cp.copy(featurePd)
        featurePd = featurePd.drop('tripT', axis=1)

        featurePd['avgSpeeds'] = 0
        tmp = featurePd.apply(dicName, AvgSpeedsDic=AvgSpeedsDic, axis=1)
        featurePd['avgSpeeds'] =  featurePd['tripDist']/tmp
        trainingSetStdTime  = pd.read_csv(commonDir + '/app/LoadFiles/trainingSetStdTime.csv')
        trainingSetMeanTime = pd.read_csv(commonDir + '/app/LoadFiles/trainingSetMeanTime.csv')

        TimeEstimator = joblib.load(commonDir + '/app/Regression/CabGBRTime_a.pkl')
        #First argument is std and second is mean
        TimeNormalization = np.loadtxt(commonDir + '/app/LoadFiles/timeNormalization.csv', delimiter=',')
        npFeatures = featurePd.as_matrix()
        nptrainingSetMeanTime = trainingSetMeanTime.as_matrix()
        nptrainingSetStdTime  = trainingSetStdTime.as_matrix()
        #transform the data using the same normalization and centering used in the fit
        NormalizedCenteredFeatures = (featurePd-trainingSetMeanTime)/trainingSetStdTime
        NormalizedCenteredFeatures = NormalizedCenteredFeatures.as_matrix()
        #make the time prediction
        predicted_Time = TimeEstimator.predict(NormalizedCenteredFeatures) * TimeNormalization[0] + TimeNormalization[1]

        featurePd = cp.copy(featurePdTmp)
        featurePd['tripT'] = predicted_Time
        returnDic['cabTimeStr'] = 'Estimated trip time: ' + str(int(round(predicted_Time*0.75/60.0))) + '-' + str(int(round(predicted_Time*1.25/60.0))) + ' mins'

    else:
        featurePd, WUInfo   = GetWeatherNow(featurePd)
        returnDic['WUInfo'] = WUInfo
        
    
    trainingSetStd = pd.read_csv(commonDir + '/app/LoadFiles/trainingSetStd.csv')
    trainingSetMean = pd.read_csv(commonDir + '/app/LoadFiles/trainingSetMean.csv')

    FareEstimator = joblib.load(commonDir + '/app/Regression/CabRegressionFare_a.pkl')
    #Firts argument is std and second is mean
    FareNormalization = np.loadtxt(commonDir + '/app/LoadFiles/fareNormalization.csv', delimiter=',')

    npFeatures = featurePd.as_matrix()
    nptrainingSetMean = trainingSetMean.as_matrix()
    nptrainingSetStd  = trainingSetStd.as_matrix()
    #transform the data using the same normalization and centering used in the fit
    NormalizedCenteredFeatures = (featurePd-trainingSetMean)/trainingSetStd
    NormalizedCenteredFeatures = NormalizedCenteredFeatures.as_matrix()

    #special case if going to JFK
    OneTicketMetro = 2.75
    intNbPeople = int(nbPeople)
    if returnDic['JFKZone'] == '7':
        cabCost = 52.8*(1.15)
        OneTicketMetro += 5
        cabCostStr = '%3.1f' % cabCost
        returnDic['cabCostStr'] = 'Cost per person: $' + str(cabCost/intNbPeople)
    else:
        cabCost = FareEstimator.predict(NormalizedCenteredFeatures) * FareNormalization[0] + FareNormalization[1]
        cabCostStr = '%3.1f' % cabCost
        returnDic['cabCostStr'] = 'Cost per person: $' + str(int(round(cabCost*0.91/intNbPeople))) + '-$' + str(int(round(cabCost*1.09/intNbPeople)))
    
    if (not (departTime == 'now')) & (returnDic['JFKZone'] != '7'):
        returnDic['cabCostStr'] = 'Cost per person: $' + str(int(round(cabCost*0.73/intNbPeople))) + '-$' + str(int(round(cabCost*1.27/intNbPeople)))
    
    cabCost = '%3.1f' % cabCost
    
    
    address = []
    address.append(returnDic['addressList'][0])
    address.append(returnDic['addressList'][1])
    cabTime       = '%3d mins' % round(featurePd['tripT']/60.0)
    metroWalkTime = returnDic['MetroWalkTripTime']
    metroCost     = OneTicketMetro
    
    returnDic['cabTime']   = cabTime
    returnDic['cabCost']   = cabCost
    returnDic['metroCost'] = metroCost

    if returnDic['JFKZone'] == '7':
        returnDic['JFKZoneStr'] = 'Metro and possibly airtrain'
    else:
        returnDic['JFKZoneStr'] = 'Metro'
    
    return returnDic, 0
  
  
  
if __name__ == '__main__':
    
    origin = '45W, 25th street, NYC'
    destination = 'Museum Of Modern Art NYC'
    nbPeople = '3'
    departTime = ''
    PredictItinary(origin, destination, nbPeople, departTime)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
