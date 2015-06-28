from flask import render_template, request
from app import app
#from flask.ext.googlemaps import GoogleMaps
import PredictFare
import json

@app.route('/') 
def addresses_input(): 
  return render_template("input.html") 


@app.route('/about')
def index():
    return render_template("about.html")

@app.route('/error')
def error():
    return render_template("error.html")


@app.route('/slides')
def slides():
    return render_template("slides.html")

 
@app.route('/output') 
def addresses_output():
    address1 = request.args.get('address1')
    address2 = request.args.get('address2')
    #print address1
    #print address2
    departTime     = request.args.get('clockPicker')
    #departTime     = request.args.get('testclock')
    print departTime
    nbPeople = request.args.get('nbPeople')
    if (departTime == None) | (departTime == ''):
        departTime = 'now'
    if (len(nbPeople) == 0):
        nbPeople = '1'
    
    #flag = 0
    #if (int(nbPeople) > 6):
    #    errorstr = 'Too many people!'
    #    flag = 1
    #elif (int(nbPeople) < 1):
    #    errorstr = 'Not enough people!'
    #    flag = 1
    
    #if flag == 0:
    returnDic, flag = PredictFare.PredictItinary(address1, address2, nbPeople, departTime)
    if flag == 1:
        return render_template("error.html", errorstr = returnDic)
    else:
        return render_template("output.html", returnDic = returnDic)
    #else:
        #return render_template("error.html", errorstr=errorstr)










