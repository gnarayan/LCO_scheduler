#!python
import sys
import os
import glob
import json
import multiprocessing
import cPickle as pickle
import requests
import httplib
import warnings
warnings.simplefilter('once')
from collections import Counter, OrderedDict
import datetime
import numpy as np
from matplotlib.mlab import rec2txt
from astropy.coordinates import SkyCoord as c
from astropy.coordinates import EarthLocation, get_sun
from astropy.time import Time
import astropy.units as u
import astroplan as ap


class OrderedCounter(Counter, OrderedDict):
        pass


def setup_target(this_target, startsemester, endsemester, plan_sites, verbose=False):

    # we used to have to do some ugly string conversion here, but now we just fix it in the text file input
    nice_target_name = this_target.targetname

    # convert the coordinates to decimal degrees (for LCO and astroplan)
    position = (this_target.RA, this_target.Dec)
    dec_pos = c(*position, unit=(u.deg, u.deg))
    rad = dec_pos.ra.value
    decd= dec_pos.dec.value


    # create a target for astroplan
    plan_target = ap.FixedTarget(name=nice_target_name, coord=dec_pos)
    print plan_target

    # we want this many observations split over the window
    n_full = this_target.n_obs

    # as the target gets fainter, we actually need to split the exposures up a
    # bit more, so the individual ones don't get too long
    # this guards against guiding failure, phase blurring, cosmic rays, streaks...
    if this_target.gmag > 19.1:
        n_split = 4
    else:
        n_split = 3

    # compute a reasonable exposure time
    # this is a deg=2 polynomial fit for gmag vs the log (base 10) total exptime
    # it's only valid from g=15 to 20 maag (which is really the range we care about)
    # this gives us a reasonable S/N ~100 at airmass 1.3. We split this into n_split exposures
    exptime_poly = np.array([-0.04543623, 1.8909311, -16.43566578])
    exptime_total = 10.**np.polyval(exptime_poly, this_target.gmag)
    requested_exposure = round(exptime_total/n_split)

    coord_equinox= 2000.

    target = {'name':nice_target_name,\
                'type':'SIDEREAL',\
                'ra':rad,\
                'dec':decd,\
                'epoch':coord_equinox,\
                'equinox':'J2000',\
                'coordinate_system':'ICRS'}

    molecule    = {'ag_mode':'ON',\
                    'ag_name':'',\
                    'bin_x':1,\
                    'bin_y':1,\
                    'defocus':0.0,\
                    'exposure_count':n_split,\
                    'exposure_time':requested_exposure,\
                    'fill_window':False,\
                    'filter':"gp",\
                    'instrument_name':'1M0-SCICAM-SINISTRO',\
                    'type':'EXPOSE'}

    # get a list of the available dates from the start to the end of the semester
    avail_dates =Time(np.arange(startsemester.mjd, endsemester.mjd+1, 1.), format='mjd')

    # for each target, on each date, at each site, store the rise and set times
    target_date_sites = {}

    # kick out dates on which the moon illumination is too high
    if this_target.gmag > 18.4:
        moon_illum_limit = 0.40
    else:
        moon_illum_limit = 0.85

    # in a perfect world, the target would be up for the entire useful window
    # in which case we want to split n_split_full observations over n_full_dates
    n_full_dates = len(avail_dates)

    # in reality though, targets rise and set below horizons, so we may have less than the full window
    # get the rise and set times of the target for each date, and check if it's still night
    horizon = -18*u.deg
    horizon2 = 60*u.deg
    mask = []
    for date in avail_dates:
        nice_date = date.iso.split(' ')[0]
        target_date_sites[nice_date] = {}
        for site in plan_sites:

            # when does the sun rise and set
            sun_set_time = site.sun_set_time(date, which='nearest', horizon=horizon) + 5*u.minute
            sun_rise_time = site.sun_rise_time(sun_set_time, which='next', horizon=horizon) - 5*u.minute

            # when does the target rise and set
            target_rise_time = site.target_rise_time(sun_set_time, plan_target, which='nearest', horizon=horizon2) + 5*u.minute

            # deal with edge cases where we get the rise time on the next day, and really the target is up before the sunset
            if sun_set_time - target_rise_time >= 0.5*u.day:
                target_rise_time = site.target_rise_time(sun_set_time+0.45*u.day, plan_target, which='nearest', horizon=horizon2) + 5*u.minute
            # deal with edge case where the target rise and sunset are really close to each other in time
            if target_rise_time - sun_set_time > 0.95*u.day:
                target_rise_time = site.target_rise_time(sun_set_time-0.1*u.day, plan_target, which='nearest', horizon=horizon2) + 5*u.minute
            if target_rise_time > sun_rise_time:
                target_rise_time = site.target_rise_time(sun_rise_time, plan_target, which='previous', horizon=horizon2) + 5*u.minute

            if target_rise_time.jd < 0:
                # the target never rises on this date at this site
                # move on to the next site
                if verbose:
                    message = "Target {} does not cross horizon, so we get {:.3f} for rise time.\
                            Skipping date {} at site {}".format(nice_target_name,\
                            target_rise_time.jd, date.iso.split(" ")[0], site.name)
                    warnings.warn(message, RuntimeWarning)
                continue
            target_set_time  = site.target_set_time(target_rise_time, plan_target, which='next', horizon=horizon2) - 5*u.minute

            # make sure the observing window is at night
            if (target_rise_time > sun_rise_time):
                if verbose:
                    print("Rise (T|S): {} {}\nSet  (T|S): {} {}").format(\
                        target_rise_time.iso, sun_set_time.iso, target_set_time.iso, sun_rise_time.iso)
                    message = "Target {} not observable. Skipping date {} at site {}".format(nice_target_name, nice_date, site.name)
                    warnings.warn(message, RuntimeWarning)
                continue

            # define window by (sunset time, target rise time) and min(sunrise time, target_set time)
            window_start_time = np.max([sun_set_time, target_rise_time])
            window_end_time   = np.min([sun_rise_time, target_set_time])
            time_grid = window_start_time + (window_end_time - window_start_time)*np.linspace(0, 1, 5)
            moon_alt = site.moon_altaz(time_grid).alt
            moon_up = (moon_alt > horizon).nonzero()[0]
            if len(moon_up) > 0:
                rise_ind = moon_up[0]
                moon_illum_time = time_grid[rise_ind]
                moon_illum = ap.moon_illumination(moon_illum_time)
                if moon_illum > moon_illum_limit:
                    if verbose:
                        message = "Moon is up ({}) and illumination {:.3f} is too high for target {} w/ g {:.3f} at site {}"\
                            .format(moon_alt[rise_ind], moon_illum, nice_target_name, this_target.gmag, site.name)
                        warnings.warn(message, RuntimeWarning)
                    continue

            # sanity checks for time
            if window_start_time > window_end_time:
                if verbose:
                    print nice_target_name
                    print site.name
                    print nice_date
                    print sun_set_time.iso, sun_rise_time.iso, "Sun vis window"
                    print target_rise_time.iso, target_set_time.iso, "Target vis window"
                    message = "{} {} {}\nHuh... start after end {} {}".format(nice_target_name, site.name, nice_date, window_start_time.iso, window_end_time.iso)
                    warnings.warn(message, RuntimeWarning)
                continue

            if window_end_time - window_start_time > 1.*u.day:
                if verbose:
                    print nice_target_name
                    print site.name
                    print nice_date
                    print sun_set_time.iso, sun_rise_time.iso, "Sun vis window"
                    print target_rise_time.iso, target_set_time.iso, "Target vis window"
                    message = "{} {} {}\nHuh... longer than a day {} {}".format(nice_target_name, site.name, nice_date, window_start_time.iso, window_end_time.iso)
                    warnings.warn(message, RuntimeWarning)
                continue

            # if this window is too short for the observation
            if window_end_time - window_start_time < exptime_total*u.second:
                if verbose:
                    message = 'Window too short for {} at {} ({},{})'.format(nice_target_name, site, window_start_time.iso, window_end_time.iso)
                    warnings.warn(message, RuntimeWarning)
                continue

            # it's therefore a useful night for this target
            target_date_sites[nice_date][site.name] = (window_start_time, window_end_time)
        #end for site

        # if we hae a window for this site on this date, then use the date, else don't
        if len(target_date_sites[nice_date].keys()) == 0:
            mask.append(False)
        else:
            mask.append(True)
    #end for date
    mask = np.array(mask)
    avail_dates = avail_dates[mask]

    # how many observations can we reasonably squeeze into this reduced window
    n_good_dates = len(avail_dates)

    #print "Fraction of moon-illum controlled window that target {} is observable: {:.3f}".format(nice_target_name, 1.*n_good_dates/n_full_dates)
    date_fraction =  (1.*n_good_dates / n_full_dates)

    # some logick to make sure we get a reasonable number of observations
    if date_fraction > 0.85:
        date_fraction = 1
    elif 0.4 < date_fraction <= 0.85:
        pass
    else:
        date_fraction = 0.4
    n_blocks = np.round(n_full*date_fraction)

    if n_good_dates == 0:
        # we can't schedule this target - it isn't up at all
        out = {'target':dict(target), 'molecule':dict(molecule), 'name':nice_target_name,\
                'plan_target':plan_target, 'requests':[]}
        return out

    #print "Splitting window into {:n} blocks".format(n_blocks)

    # we then need to split the available dates into roughly equal chunks
    date_blocks = np.array_split(avail_dates, n_blocks)

    ############################ CONSTRUCT LCO FORMAT REQUESTS FROM SCHEDULE  ############################

    #setup the request
    windows = None


    constraints = {'max_airmass': 1.6,\
                   'min_lunar_distance':30}
    if (1.*n_good_dates / n_full_dates) < 0.6:
        constraints['max_airmass'] = 2.0

    location    = {'telescope_class':'1m0'}



    request = {
        "constraints" : constraints,
        "location" : location,
        "molecules" : [molecule],
        "observation_note" : "",
        "observation_type" : "NORMAL",
        "target" : target,
        "type" : "request",
        "windows" : windows,
        }

    window = {'start':'',\
                'end':''}

    #### MAKE THE BLOCK'S WINDOWS ####
    out_json_files = []

    # loop over all the blocks
    for block_num, this_block_dates in enumerate(date_blocks):
        windows = []

        user_request = {
            "operator" : "SINGLE",
            "proposal" : "LCO2017AB-002",
            "requests" : [request],
            "ipp_value": 1.05,
            "submitter": "gnarayan",
            "observation_type":"NORMAL",
            "group_id":'%s_%02i'%(nice_target_name, block_num)
            }

        out_json_file = 'LCO_json/%s_%02i_of_%02i.json'%(nice_target_name, block_num+1, n_blocks)
        #print block_num, [x.iso.split(' ')[0] for x in this_block_dates]


        # loop over all the dates in this block
        for date in this_block_dates:
            nice_date = date.iso.split(' ')[0]
            site_windows = target_date_sites[nice_date]

            # loop over any sites for this date
            for this_block_date_site in site_windows:
                window_start, window_end = site_windows[this_block_date_site]
                this_window = dict(window)

                # add the window in
                this_window['start'] = window_start.iso
                this_window['end']   = window_end.iso
                windows.append(this_window)

        # update the requested window for this entire block
        request['windows'] = windows
        request['observation_note'] = '%02i_of_%02i'%(block_num+1, n_blocks)
        user_request['requests'] = [request]

        #convert the request to JSON
        json_user_request = json.dumps(user_request)

        # save the request to a file for a now
        with open(out_json_file, 'w') as outf:
            json.dump(user_request, outf, indent=2, sort_keys=True)

    out = {'target':dict(target), 'molecule':dict(molecule), 'name':nice_target_name, 'plan_target':plan_target,\
            'requests':out_json_files}

    return out


def main():

    ############################ SETUP STATIC QUANTITIES ############################
    # read the target file and setup a mapping between target name, and corresponding time file
    targets = np.recfromtxt('targets_LCO2017AB_002.txt', names=True)

    window = {'start':'',\
                'end':''}

    target_info = {}

    nproc = multiprocessing.cpu_count()

    # setup astroplan and astropysics to check the requested observation times
    # there is also the Haleakala site, but it doesn't have a 1meter
    site_names  = ['sso', 'saao', 'ctio', 'mcdonald']
    plan_sites  = [ap.Observer.at_site(site) for site in site_names]

    # The question is not where are we, but when are we?
    today = Time(datetime.datetime.now(), format='datetime')
    tomorrow = datetime.datetime.fromordinal(datetime.datetime.now().toordinal() + 2)
    startsemester  = Time(tomorrow,format='datetime')
    endsemester = Time('2017-11-30T00:00:00', format='isot')

    print "Start of window ",startsemester.to_datetime(), startsemester.jd
    print "End of window   ",endsemester.to_datetime(), endsemester.jd

    ############################ SETUP TIME-DEPENDENT REQUESTS ############################

    restore=True
    obsplan_files = glob.glob('obsplan_config_*.json')
    if len(obsplan_files) > 0 and restore:
        obsplan_files = sorted(obsplan_files)
        newest_obsplan = obsplan_files[-1]
        print("Plan files exist. Restoring %s/pkl"%newest_obsplan)
        with open(newest_obsplan, 'r') as f:
            target_info = json.load(f)

    else:
        print("Plan files don't exist, or cannot restore. Creating plan files. Setting up observing blocks.")

        # Store the requested observing blocks for ALL targets by date
        target_structures = {}

        processPool = multiprocessing.Pool(nproc)
        lock = multiprocessing.Lock()

        multi_res = []
        for this_target in targets:
            args = (this_target, startsemester, endsemester, plan_sites)
            #setup_target(*args)
            res  = processPool.apply_async(setup_target, args)
            multi_res.append(res)


        processPool.close()
        processPool.join()

        for res in multi_res:
            if res is not None:
                out = res.get()


if __name__=='__main__':
    main()
