#!/usr/bin/env python
import sys
import glob
import json
import requests
import warnings
import astropy.units as u

if __name__=='__main__':
    exptime = 0                           
    token = sys.argv[1]
    f = glob.glob('LCO_json/*json')
    for infile in f:
        with open(infile, 'r') as indata:
            a = json.load(indata)
        resp_validate = requests.post('https://observe.lco.global/api/userrequests/validate/',\
		         headers={'Authorization': 'Token {}'.format(token)}, json=a).json()
        if len(resp_validate['errors']) == 0:
            exptime += resp_validate['request_durations']['duration']
            resp_submit = requests.post('https://observe.lco.global/api/userrequests/',\
		     headers={'Authorization': 'Token {}'.format(token)}, json=a)
	    try:
    		resp_submit.raise_for_status()
	    except requests.exceptions.HTTPError as exc:
                print resp_submit.json()
            	message = 'Request failed: {}'.format(infile)
            	warnings.warn(message)
	else:
            print resp_validate
	    print 'Not submitted: {}'.format(infile)

    exptime*= u.second
    left = exptime - 48*u.hour
    print exptime, left

