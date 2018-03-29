#!/usr/bin/env python
import sys
import requests
import numpy as np
import astropy.table as at

if __name__=='__main__':
    targets = at.Table.read('targets_LCO2018A_002.txt', format='ascii')
    headers={'Authorization': 'Token {}'.format(sys.argv[1])}
    for x in targets['targetname']:
        obs = requests.get('https://observe.lco.global/api/userrequests/?proposal=LCO2018A-002&title={}'.format(x.split('.')[0]),headers=headers).json()
        for y in obs['results']:
            print(y['group_id'])
