#!/usr/bin/env python
from __future__ import print_function
import sys
import os
from six.moves import input
import datetime
import numpy as np
import astropy.table as at
import astropy.time
import requests
import glob
import json
import time
import warnings
from collections import Counter
import astropy.units as u
from matplotlib.mlab import rec2txt

if __name__=='__main__':
    headers={'Authorization': 'Token {}'.format(sys.argv[1])}
    targets = at.Table.read('targets_LCO2017AB_002.txt', format='ascii')

    # get the failed requests that have not already been resubmitted
    failed = requests.get('https://observe.lco.global/api/userrequests/?user=gnarayan&proposal=LCO2017AB-002&state=WINDOW_EXPIRED',headers=headers).json()
    fid = set([f['group_id'] for f in failed['results']])

    pid = set()
    qid = set()

    now = datetime.datetime.now()
    now = astropy.time.Time(now, format='datetime')

    for x in targets['targetname']:
        req = 'https://observe.lco.global/api/userrequests/?user=gnarayan&proposal=LCO2017AB-002&title={}&state=PENDING'.format(x.split('.')[0])
        pending = requests.get(req, headers=headers).json()
        thispid = set([p['group_id'] for p in pending['results']])
        pid = pid | thispid
        qid = qid | thispid
        req = 'https://observe.lco.global/api/userrequests/?user=gnarayan&proposal=LCO2017AB-002&title={}&state=COMPLETED'.format(x.split('.')[0])
        completed = requests.get(req, headers=headers).json()
        #thiscount = Counter([c['group_id'] for c in completed['results']])
        thiscid = set([c['group_id'] for c in completed['results']])
        err = thiscid & thispid
        if len(err) > 0:
            message = 'Object {} has completed requests that are pending {}'.format(x, err)
            warnings.warn(message, RuntimeWarning)
        pid = pid | thiscid

    failed_need_resub = fid - pid
    print("PENDING")
    print(qid)
    print("FAILED NEED RESUB")
    print(failed_need_resub)
    pending_targets_blocks = [x.rsplit('_',1) for x in pid]
    pending_targets_blocks = np.rec.fromrecords(pending_targets_blocks, names='target,blocks')
    if len(failed_need_resub) == 0:
        message = 'Nothing to process. Bye.'
        print(message)
        sys.exit(0)

    exptime = 0
    for f in failed_need_resub:
        resub_group_id = f
        target, block_num = f.rsplit('_',1)

        # SANITY CHECK - make sure we have a request file corresponding to this
        # THIS IS JUST IN CASE SOMEONE ELSE SUBMITS A REQUEST THAT FAILED WITH
        # THE SAME GROUP ID BUT IT IS STILL REGISTERED TO YOU
        # IF THIS HAPPENS, FIND OUT WHO
        orig_request_file_pattern = '{}_{:02n}_of_[0-9][0-9].json'.format(target, int(block_num)+1)
        orig_request_file_pattern = os.path.join('LCO_json',orig_request_file_pattern)
        orig_request_files = list(sorted(glob.glob(orig_request_file_pattern)))
        nfiles = len(orig_request_files)
        if nfiles != 1:
            message = 'Huh. Could not find a unique original request file for target {}. {}'.format(target, orig_request_files)
            warnings.warn(message, RuntimeWarning)
        else:
            orig_request_file = orig_request_files[0]

            # move the original failed request to a file indiciating that it failed to execute
            today = time.strftime("%Y%m%d")
            failed_filename = orig_request_file.replace('.json','_EXPIRED_{}.json'.format(today))
            os.rename(orig_request_file, failed_filename)

        # what blocks are still pending for this target
        mask = (pending_targets_blocks.target == target)
        remaining_requests = len(pending_targets_blocks.blocks[mask])
        if remaining_requests == 0:
            message = 'There are no more pending requests for the target {}. Cannot reschedule'.format(target)
            warnings.warn(message)

        resub_blocks= [int(x) for x in pending_targets_blocks.blocks[mask]]
        resub_blocks = np.array(resub_blocks)
        resub_files = []
        # get the corresponding requests for each pending block
        for resub_block in resub_blocks:
            resub_request_file_pattern = '{}_{:02n}_of_[0-9][0-9].json'.format(target, resub_block+1)
            resub_request_file_pattern = os.path.join('LCO_json',resub_request_file_pattern)
            resub_files += list(sorted(glob.glob(resub_request_file_pattern)))
        resub_files = list(sorted(resub_files))

        nfiles = len(resub_files)
        if nfiles >= 1:
            pass
        else:
            message = 'No matching resub files for target {}'.format(target)
            warnings.warn(message, RuntimeWarning)
            continue

        # allow the user to pick which file they want
        ctr = 0
        state = False
        while((ctr < nfiles) & (~state)):
            # load the new request
            resub_file = resub_files[ctr]

            a = None
            with open(resub_file, 'r') as resub:
                a = json.load(resub)
                windows = a['requests'][0]['windows']
                useful = [astropy.time.Time(x['end'],format='iso') > now for x in windows]
                useful = np.array(useful)
                if not np.any(useful):
                    print("file {} is not useful since all windows have passed".format(resub_file))
                    ctr += 1
                    continue

            # get the observation note for the new request that matches the original
            _, fileid, _, totalid = resub_file.replace('.json','').rsplit('_',3)
            resub_observation_note = '{:02n}_of_{}'.format(int(block_num)+1, totalid)

            # make sure the output filename for this new request matches the original
            orig_request_file = resub_file.replace('{:02n}_of_'.format(resub_block+1),'{:02n}_of_'.format(int(block_num) + 1))
            out_json_file = orig_request_file
            infile = out_json_file

            # check if we are OK to proceed
            message = 'Resubmitting {} as {}/{}, Proceed [y/n]? '.format(resub_file, resub_group_id, resub_observation_note)
            resp = input(message)
            if resp[0].lower() in ('t','y'):
                state = True
                pass
            elif resp[0].lower() in ('f','n'):
                message = 'SKIPPING {}'.format(resub_file)
                warnings.warn(message)
                ctr += 1
                continue

        # the user skipped all the alternatives
        if state is False:
            message = 'You did not accept any of the new proposed observations for target {}. Skipping {}'.format(target, resub_group_id)
            warnings.warn(message)
            continue

        # the user selected some file, so load the request
        a = None
        with open(resub_file, 'r') as resub:
            a = json.load(resub)

        # update the next request with the failed request group id and note
        a['group_id'] = resub_group_id
        a['requests'][0]['observation_note'] = resub_observation_note

        # check that this new request is good
        resp_validate = requests.post('https://observe.lco.global/api/userrequests/validate/',\
		         headers=headers, json=a).json()
        if len(resp_validate['errors']) == 0:
            exptime += resp_validate['request_durations']['duration']

            # resubmit the request
            resp_submit = requests.post('https://observe.lco.global/api/userrequests/',\
		     headers=headers, json=a)
            try:
                resp_submit.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                print(resp_submit.json())
                message = 'Request failed: {}'.format(infile)
                warnings.warn(message)
                continue
        else:
            print(resp_validate)
            print('Not submitted: {}'.format(infile))
            continue

        # save the request to a file for a now
        with open(out_json_file, 'w') as outf:
            json.dump(a, outf, indent=2, sort_keys=True)

    exptime*= u.second
    left = exptime - 48*u.hour
    print(exptime, left)
