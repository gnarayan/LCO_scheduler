#!python
import sys
import os
import glob
import json
import multiprocessing
import cPickle as pickle
import urllib
import httplib
import warnings
warnings.simplefilter('once')
from collections import Counter, OrderedDict
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import cm
from matplotlib.mlab import rec2txt
from astropy.coordinates import SkyCoord as c
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astroplan.scheduling import PriorityScheduler, Schedule, SequentialScheduler
from astroplan.plots import plot_airmass, plot_schedule_airmass
import astropy.units as u
import astroplan as ap


class OrderedCounter(Counter, OrderedDict):
        pass


def schedule_and_plot_night(date, blocks, plan_sites, plan_sched, target_structures, target_colors):
    print "Scheduling %s"%date
    fig = plt.figure(figsize=(13, 10))
    site_axes = [fig.add_subplot(2,2,i+1) for i in range(len(plan_sites))]

    off = 30*u.minute # add some padding around sunset and sunrise for the Schedule
    # create a dictionary to store the schedules for each site on this night
    this_date_site_schedules = {}

    all_targets = set([block.target.name for block in blocks])

    for site, axes, sched in zip(plan_sites, site_axes, plan_sched):

        # setup the schedule for this night
        sun_set_time  = site.sun_set_time(Time(date, format='isot'), which='next')

        # theres an odd bug if you are within a short time gap of sunset itself
        # fix it by getting the nearest time instead
        if sun_set_time.value == -999.0:
            sun_set_time  = site.sun_set_time(Time(date, format='isot'), which='nearest')

        sun_rise_time = site.sun_rise_time(sun_set_time, which='next')
        # run the scheduler
        this_date_schedule = Schedule(sun_set_time-off, sun_rise_time+off)
        sched(blocks, this_date_schedule)

        # save the scheduled observing blocks for this site
        this_date_site_schedules[site.name] = list(this_date_schedule.observing_blocks)

        # plot the blocks
        brightness_shading = True
        plotted_targets = []
        for block in this_date_schedule.observing_blocks:
            plan_target = target_structures[block.target.name]

            color = target_colors[block.target.name]
            style={'color':color}

            # plot the airmass plot for this target
            if not block.target.name in plotted_targets:
                #style['label'] = plan_target.name
                axes = plot_airmass(plan_target, site, sun_set_time,\
                        style_kwargs=style, ax=axes,\
                        brightness_shading=brightness_shading, altitude_yaxis=True)
                plotted_targets.append(block.target.name)
                brightness_shading = False

            # plot the block
            plot_start = block.start_time.plot_date
            plot_end   = block.end_time.plot_date
            axes.axvspan(plot_start, plot_end, fc=color, lw=0, alpha=0.6)
        #endblock

        # plot the unscheduled targets so we can keep a track
        plotted_targets = set(plotted_targets)
        unscheduled_targets = all_targets - plotted_targets
        if len(plotted_targets) > 0:
            brightness_shading = False
        else:
            brightness_shading = True
        for target in unscheduled_targets:
            plan_target = target_structures[target]
            color = target_colors[target]
            style={'color':color,'linestyle':'--'}
            # plot the airmass plot for this target
            axes = plot_airmass(plan_target, site, sun_set_time,\
                    style_kwargs=style, ax=axes,\
                    brightness_shading=brightness_shading, altitude_yaxis=True)
            brightness_shading = False
        if len(all_targets) > 0:
            axes.legend(loc='upper left', frameon=False, prop={'size':8}, handlelength=4.)
            axes.set_title(site.name)
            axes.axhline(2.0,linestyle=':')
    #endsite
    fig.suptitle('Observations on %s'%(date), size='x-large')
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig('%s_schedule.pdf'%(date))
    plt.close(fig)
    return date, this_date_site_schedules



def setup_target(this_target, startsemester, endsemester, target, molecule, coord_equinox, read_out, plan_sites, verbose=False):

    # we used to have to do some ugly string conversion here, but now we just fix it in the text file input
    nice_target_name = this_target.targetname

    # convert the coordinates to decimal degrees (for LCO and astroplan)
    position = (this_target.RA, this_target.Dec)
    dec_pos = c(*position, unit=(u.deg, u.deg))
    rad = dec_pos.ra.value
    decd= dec_pos.dec.value

    # setup the target block of the scheduler request
    target['name'] = nice_target_name
    target['ra'] = rad
    target['dec'] = decd
    target['epoch'] = coord_equinox

    # create a target for astroplan
    plan_target = ap.FixedTarget(name=nice_target_name, coord=dec_pos)
    print plan_target

    # we want this many observations split over the window
    n_full = 19

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

    #setup the molecule block
    molecule['exposure_count'] = n_split
    molecule['exposure_time'] = requested_exposure

    # get a list of the available dates from the start to the end of the semester
    avail_dates =Time(np.arange(startsemester.mjd, endsemester.mjd+1, 1.), format='mjd')

    # kick out dates on which the moon illumination is too high
    moon_illum = np.array([ap.moon_illumination(date) for date in avail_dates])
    if this_target.gmag > 18.4:
        moon_illum_limit = 0.40
    else:
        moon_illum_limit = 0.85
    mask = (moon_illum > moon_illum_limit)
    if np.any(mask):
        if verbose:
            message = 'Some requested dates for %s have high moon illumination fraction. Kicking them.'%nice_target_name
            warnings.warn(message, RuntimeWarning)
        avail_dates = avail_dates[~mask]

    # in a perfect world, the target would be up for the entire useful window
    # in which case we want to split n_split_full observations over n_full_dates
    n_full_dates = len(avail_dates)

    # in reality though, targets rise and set below horizons, so we may have less than the full window
    # get the rise and set times of the target for each date, and check if it's still night
    horizon = -18*u.deg
    mask = []
    for date in avail_dates:
        good_flag = False
        for site in plan_sites:
            target_rise_time = site.target_rise_time(date, plan_target, which='nearest', horizon=horizon)
            if target_rise_time.jd < 0:
                # the target never rises on this date at this site
                # move on to the next site
                if verbose:
                    message = "Target {} does not cross horizon, so we get {:.3f} for rise time.\
                            Skipping date {} at site {}".format(nice_target_name, target_rise_time.jd, date.iso.split(" ")[0], site.name)
                    warnings.warn(message, RuntimeWarning)
                continue
            target_set_time  = site.target_set_time(target_rise_time, plan_target, which='next', horizon=horizon)
            test_times = Time(np.arange(target_rise_time.mjd+0.05, target_set_time.mjd+0.1-0.05, 0.1), format='mjd')

            # note that is_night doesn't seem to like array times
            night_check = [site.is_night(t, horizon=horizon) for t in test_times]
            night_check = np.array(night_check)

            if np.any(night_check):
                # this target is up for at least some fraction of the night at some site
                # it's therefore a useful night for this target
                good_flag = True
                break
            #end if
        #end for site
        # no site had this target up at the test_times at night - date is not useable
        mask.append(good_flag)
    #end for date
    mask = np.array(mask)
    avail_dates = avail_dates[mask]

    # how many observations can we reasonably squeeze into this reduced window
    n_good_dates = len(avail_dates)

    print "Fraction of moon-illum controlled window that target {} is observable: {:.3f}".format(nice_target_name, 1.*n_good_dates/n_full_dates)
    n_blocks = np.round(n_full*(1.*n_good_dates / n_full_dates))
    if n_good_dates == 0 or n_blocks==0:
        # we can't schedule this target - it isn't up at all
        out = {'target':dict(target), 'molecule':dict(molecule), 'name':nice_target_name,\
            'blocks':None, 'plan_target':plan_target}
        return out

    print "Splitting window into {:n} blocks".format(n_blocks)

    # we then need to split the available dates into roughly equal chunks
    date_blocks = np.array_split(avail_dates, n_blocks)
    for block_num, block in enumerate(date_blocks):
        print block_num, [x.iso.split(' ')[0] for x in block]

    # this stores the blocks by date for each target, along with the block number
    # this way if we schedule the same block on different days
    # we can just append them to the same window in the request
    target_obs_date_blocks = OrderedDict()
    for block_num, this_block_dates in enumerate(date_blocks):
        # setup the observing blocks we need
        b = ap.ObservingBlock.from_exposures(plan_target, 1, requested_exposure*u.second,\
                n_split, read_out,configuration={'filter':"SDSS-g'"})
        for date in this_block_dates:
            short_date = str(date.iso.split(' ')[0])
            blocks = target_obs_date_blocks.get(short_date, None)
            if blocks is None:
                blocks = []
            blocks.append((block_num, nice_target_name, b))
            target_obs_date_blocks[short_date] = blocks
    #end for blocks
    out = {'target':dict(target), 'molecule':dict(molecule), 'name':nice_target_name,\
            'blocks':target_obs_date_blocks, 'plan_target':plan_target}
    return out



def main():

    ############################ SETUP STATIC QUANTITIES ############################
    # read the target file and setup a mapping between target name, and corresponding time file
    targets = np.recfromtxt('targets_LCO2017AB_002.txt', names=True)

    coord_equinox= 2000.

    constraints = {'max_airmass': 1.6,\
                   'min_lunar_distance':30}

    location    = {'telescope_class':'1m0'}

    proposal    = {'proposal_id':'LCO2017AB-002',\
                    'user_id':'gsnarayan@gmail.com',\
                    'password':sys.argv[1]}

    molecule    = {'ag_mode':'ON',\
                    'ag_name':'',\
                    'bin_x':1,\
                    'bin_y':1,\
                    'defocus':0.0,\
                    'exposure_count':None,\
                    'exposure_time':None,\
                    'fill_window':False,\
                    'filter':"gp",\
                    'instrument_name':'1M0-SCICAM-SINISTRO',\
                    'type':'EXPOSE'}

    target = {'name':None,\
                'ra':None,\
                'dec':None,\
                'epoch':None,\
                'equinox':'J2000',\
                'coordinate_system':'ICRS'}

    window = {'start':'',\
                'end':''}

    target_info = {}

    nproc = multiprocessing.cpu_count()

    # setup astroplan and astropysics to check the requested observation times
    # there is also the Haleakala site, but it doesn't have a 1meter
    site_names  = ['sso', 'saao', 'ctio', 'mcdonald']
    plan_sites  = [ap.Observer.at_site(site) for site in site_names]

    slew_rate = 5.*u.deg/u.second # maximum slew speed is maximum of 6 deg/sec. Assume it doesn't operate at that.
    read_out  = 42.*u.second      # SINISTRO readout time is 42 seconds
    plan_transitioner = ap.Transitioner(slew_rate, None) # the second None is for filter change ovehead.
    # constrain the observations to be within the max airmass limit
    # and at night (duh)
    plan_constraints = [ap.AirmassConstraint(constraints['max_airmass'], boolean_constraint=False),\
                            ap.AtNightConstraint.twilight_astronomical()]
    plan_sched = [PriorityScheduler(constraints=plan_constraints, observer=site,\
                    time_resolution=1.*u.minute, transitioner=plan_transitioner)\
                    for site in plan_sites]


    # The question is not where are we, but when are we?
    today = Time(datetime.datetime.now(), format='datetime')
    tomorrow = datetime.datetime.fromordinal(datetime.datetime.now().toordinal() + 1)
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
        obsplan_pkl = newest_obsplan.replace('.json','.pkl')
        print("Plan files exist. Restoring %s/pkl"%newest_obsplan)
        with open(newest_obsplan, 'r') as f:
            target_info = json.load(f)

        with open(obsplan_pkl, 'r') as f:
            obsplan = pickle.load(f)
            target_structures = obsplan['target_structures']
            obs_date_blocks = obsplan['obs_date_blocks']

    else:
        print("Plan files don't exist, or cannot restore. Creating plan files. Setting up observing blocks.")
        # Store the requested observing blocks for ALL targets by date
        obs_date_blocks = {}
        target_structures = {}

        processPool = multiprocessing.Pool(nproc)
        lock = multiprocessing.Lock()

        multi_res = []
        for this_target in targets:
            args = (this_target, startsemester, endsemester, target, molecule, coord_equinox, read_out, plan_sites)
            #setup_target(*args)
            res  = processPool.apply_async(setup_target, args)
            multi_res.append(res)
        #end for target

        processPool.close()
        processPool.join()
        for res in multi_res:
            if res is not None:
                out = res.get()
                name = out['name']
                plan_target = out['plan_target']
                thismolecule = out['molecule']
                thistarget   = out['target']

                target_structures[name] = plan_target
                target_info[name] = {'target':thistarget, 'molecule':thismolecule}
                target_obs_date_blocks = out['blocks']
                if target_obs_date_blocks is None:
                    # the object was never scheduled
                    continue
                for to in target_obs_date_blocks:
                    print target_obs_date_blocks[to]
                    blocks = obs_date_blocks.get(to,None)
                    if blocks is None:
                        blocks = []
                    blocks += target_obs_date_blocks[to]
                    obs_date_blocks[to] = blocks
        obsplan = {'target_structures':target_structures, 'obs_date_blocks':obs_date_blocks}
        print len(obs_date_blocks.keys()), " keys"

        # save the configuration
        with open('obsplan_config_%s.json'%today.iso.replace(' ','T'), 'w') as outpkl:
            json.dump(target_info, outpkl, indent=2, sort_keys=True)

        with open('obsplan_config_%s.pkl'%today.iso.replace(' ','T'), 'w') as outpkl:
           pickle.dump(obsplan, outpkl)

    # assign a color to each target
    color=cm.viridis(np.linspace(0,1,len(targets)))
    target_colors = dict(zip(target_structures.keys(), color))

    ############################ SCHEDULE REQUESTS ############################

    nights = sorted(obs_date_blocks.keys())
    print obs_date_blocks

    date_schedule_files = glob.glob('date_schedule_*.pkl')
    if len(date_schedule_files) > 0 and restore:
        date_schedule_files = sorted(date_schedule_files)
        newest_date_schedule = date_schedule_files[-1]
        print("Plan files exist. Restoring %s"%newest_date_schedule)

        with open(newest_date_schedule, 'r') as f:
            date_site_schedules = pickle.load(f)
    else:
        # save all the schedules for each date and site
        date_site_schedules = {}

#        processPool = multiprocessing.Pool(nproc)
#        lock = multiprocessing.Lock()

#        multi_res = []
        for date in nights:
            entries = obs_date_blocks[date]
            blocks = [x[2] for x in entries]
            args = (date, blocks, plan_sites, plan_sched, target_structures, target_colors)
            schedule_and_plot_night(*args)
#            res  = processPool.apply_async(schedule_and_plot_night, args)
#            multi_res.append(res)

#        processPool.close()
#        processPool.join()
#        for res in multi_res:
#            if res is not None:
#                try:
#                    date, this_date_site_schedule = res.get()
#                    date_site_schedules[date] = this_date_site_schedule
#                except Exception, e:
#                    warnings.warn('Something threw this bloody exception %s'%e)
#                    pass
#
#        with open('date_schedule_%s.pkl'%today.isot, 'w') as outpkl:
#            pickle.dump(date_site_schedules, outpkl)
    ############################ CONSTRUCT LCO FORMAT REQUESTS FROM SCHEDULE  ############################


#    target_obs_ctr = OrderedCounter()
#    unobs_date_blocks = {}
#    for date in nights:
#        blocks = obs_date_blocks[date]
#        this_date_site_schedule = date_site_schedules[date]
#        unscheduled_blocks = []
#        for block in blocks:
#            targetname = block.target.name
#            windows = []
#            for site in this_date_site_schedule:
#                this_site_blocks = this_date_site_schedule[site]
#                nsite_blocks = len(this_site_blocks)
#                for i in range(nsite_blocks):
#                    # block exists, add it's window, pop it, stop update the stored blocks, stop looking
#                    if targetname == this_site_blocks[i].target.name:
#                        thiswindow = dict(window)
#                        thiswindow['start'] = this_site_blocks[i].start_time.iso
#                        thiswindow['end'] = this_site_blocks[i].end_time.iso
#                        windows.append(thiswindow)
#                        this_site_blocks.pop(i)
#                        this_date_site_schedule[site] = this_site_blocks
#                        break
#                    # not the right block, keep looking
#                    else:
#                        pass
#                #end blocks at this site
#            #end sites
#            if len(windows) == 0:
#                unscheduled_blocks.append(block)
#            else:
#                target = target_info[targetname]['target']
#                molecule = target_info[targetname]['molecule']
#                count = target_obs_ctr.get(targetname, 1)
#
#                #setup the request
#                request = {
#                    "constraints" : constraints,
#                    "location" : location,
#                    "molecules" : [molecule],
#                    "observation_note" : "",
#                    "observation_type" : "NORMAL",
#                    "target" : target,
#                    "type" : "request",
#                    "windows" : windows,
#                    }
#
#                user_request = {
#                    "operator" : "single",
#                    "requests" : [request],
#                    "ipp_value": 1.05,
#                    "type" : "compound_request",
#                    "group_id":'%s_%02i_%s'%(targetname, count, date)
#                    }
#
#                #convert the request to JSON
#                json_user_request = json.dumps(user_request)
#                # save the request to a file for a now
#                with open('LCO_json/%s_observation_%s_%02i.json'%(targetname, date, count), 'w') as outf:
#                    json.dump(user_request, outf, indent=2, sort_keys=True)
#                count += 1
#                target_obs_ctr[targetname] = count
#                params = urllib.urlencode({'username': proposal['user_id'],
#                                   'password': proposal['password'],
#                                                  'proposal': proposal['proposal_id'],
#                                                                 'request_data' : json_user_request})
#
#                print params
#
#                #submit the request
#                headers = {'Content-type': 'application/x-www-form-urlencoded'}
#                conn = httplib.HTTPSConnection("lco.global")
#                conn.request("POST", "/observe/service/request/submit", params, headers)
#                conn_response = conn.getresponse()
#
#                # The status can tell you if sending the request failed or not.
#                # 200 or 203 would mean success, 400 or anything else fail
#                status_code = conn_response.status
#
#                # If the status was a failure, the response text is a reason why it failed
#                # If the status was a success, the response text is tracking number of the submitted request
#                response = conn_response.read()
#                print response
#
#                response = json.loads(response)
#
#                if status_code == 200:
#                    try:
#                        print "http://lco.global/observe/request/" + response['id']
#                    except KeyError:
#                        print response['error']
#
#            #endelse
#        #end requested blocks on this date
#        unobs_date_blocks[date] = unscheduled_blocks
#    #end dates
#    unobsplan = {'target_obs_ctr':target_obs_ctr, 'unobs_date_blocks':unobs_date_blocks}
#    with open('unscheduled_obs_%s.pkl'%today.isot, 'w') as outpkl:
#        pickle.dump(unobsplan, outpkl)
#



if __name__=='__main__':
    main()
