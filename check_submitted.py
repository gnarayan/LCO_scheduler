#!/usr/bin/env python
import sys
import requests
import numpy as np

if __name__=='__main__':
    targets = np.recfromtxt('targets_LCO2017AB_002.txt', names=True)
    headers={'Authorization': 'Token {}'.format(sys.argv[1])}
    for x in targets.targetname:
        obs = requests.get('https://observe.lco.global/api/userrequests/?proposal=LCO2017AB-002&title={}'.format(x.split('.')[0]),headers=headers).json()
        for y in obs['results']:
            print y['group_id']
