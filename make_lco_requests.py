#!python
import sys
import os
import glob
import json
import multiprocessing
import cPickle as pickle
import warnings
warnings.simplefilter('always')
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



def read_completed_obs():
    completed_obs = {}
    with open('completed_observations', 'r') as f:
        for line in f.readlines():
            target, obs = line.strip().split(':')
            completed_obs[target] = np.array(obs.split(), dtype='int', ndmin=1)
    return completed_obs



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



def setup_target(this_target, f, completed_obs, target, molecule,\
                target_moon_illum_limit, target_nsplit, today, coord_equinox, read_out, max_window):
    #read in the requested observations file
    nice_target_name = f.replace('.time','')
    obj = np.recfromtxt(f, names='Time,Year,Month,Day,Hr')
    # some files have 19 observations - remove the extras from the end
    if len(obj) > this_target.n_ep:
        obj = obj[0:this_target.n_ep]
    
    
    # remove completed observations from consideration
    this_completed_obs = completed_obs[nice_target_name]
    obs_inds = np.arange(this_target.n_ep)
    if len(this_completed_obs) > 0:
        this_completed_obs -= 1 #obs number and index differ by one
        remaining_obs = set(range(this_target.n_ep)) - set(this_completed_obs)
        remaining_obs = np.array(sorted(list(remaining_obs)))
        message = '%s: Removing completed observations - %s'\
                %(nice_target_name, ' '.join([str(x+1) for x in this_completed_obs]))
        warnings.warn(message)
        obj = obj[remaining_obs]
        obs_inds = obs_inds[remaining_obs]
    
    
    # adjust dates on which the moon illumination is too high
    moon_illum = np.array([ap.moon_illumination(Time(x, format='mjd')) for x in obj.Time])
    moon_illum_limit = target_moon_illum_limit.get(nice_target_name, 0.65)
    mask = (moon_illum > moon_illum_limit)
    if np.any(mask):
        message = 'Some requested dates for %s have high moon illumination fraction. Adjusting'%nice_target_name
        warnings.warn(message, RuntimeWarning)
    while np.any(mask):
        obj.Time[mask] += 1. #only adjust those days with high moon illumination into the future
        moon_illum = np.array([ap.moon_illumination(Time(x, format='mjd')) for x in obj.Time])
        mask = (moon_illum > moon_illum_limit)
    
    # Make sure the observations are sorted after all this shenanigans
    obj.sort(order='Time')
    
    # Instead, we'll get the dates for the observations
    # and the number of observations requested on this date
    obs_dates = [Time(x, format='mjd') for x in obj.Time]
    
    # Fix the date for any incomplete observations in the past
    for i in range(len(obs_dates)):
        if obs_dates[i] < today:
            message = '%s requested date %s is earlier than today %s. Moving to future'\
                    %(nice_target_name, obs_dates[i].isot, today.isot)
            warnings.warn(message)
            obs_dates[i] = today + max_window*u.day
    obs_ctr = OrderedCounter(x.isot.split('T')[0] for x in obs_dates)
    
    #convert the coordinates to decimal degrees (for LCO and astroplan)
    position = (this_target.RA, this_target.DEC)
    dec_pos = c(*position, unit=(u.hourangle, u.deg))
    rad = dec_pos.ra.value
    decd= dec_pos.dec.value
    plan_target = ap.FixedTarget(name=nice_target_name, coord=dec_pos)
    
    # the exposure times are designed with 90s slew + 15s settle + 51 second readout
    # the transitioner doesn't know anything about the other objects that LCOGT will schedule
    # it doesn't have a simple fixed overhead for each block (as opposed to each exoposure)
    # so, instead compute an exposure time that will fill the block with the slew + settle included
    n_split = target_nsplit.get(nice_target_name, 3)
    optimum_exposure = round((this_target.time_ep - n_split*47.)/n_split)
    requested_exposure = np.floor((this_target.time_ep - n_split*47. - 105.)/n_split)
    
    #setup the target block
    target['name'] = nice_target_name
    target['ra'] = rad
    target['dec'] = decd
    target['epoch'] = coord_equinox
    
    #setup the molecule block
    molecule['exposure_count'] = n_split
    molecule['exposure_time'] = requested_exposure
    
    target_obs_date_blocks = {}
    for to, n_blocks in obs_ctr.iteritems():
        # setup the observing blocks we need
        blocks = target_obs_date_blocks.get(to,None)
        if blocks is None:
            blocks = []
        for i in range(n_blocks):
            b = ap.ObservingBlock.from_exposures(plan_target, 1, optimum_exposure*u.second,\
                    n_split, read_out,configuration={'filter':"SDSS-g'"})
            blocks.append(b)
        target_obs_date_blocks[to] = blocks
    #end for nights
    out = {'target':dict(target), 'molecule':dict(molecule), 'name':nice_target_name,\
            'blocks':target_obs_date_blocks, 'plan_target':plan_target}
    return out



def main():

    ############################ SETUP STATIC QUANTITIES ############################
    # read the target file and setup a mapping between target name, and corresponding time file
    targets = np.recfromtxt('targets', names=True)

    time_files   = glob.glob('*time')
    target_names = [int(f.lstrip('SDSSJ').lstrip('WD').lstrip('0').replace('.time','').split('-')[0]) for f in time_files]
    target_time_files = dict(zip(target_names, time_files))
    target_nsplit = {'SDSSJ022817':4, 'SDSSJ081508':4}
    target_moon_illum_limit = {'SDSSJ022817':0.25, 'SDSSJ081508':0.25}

    completed_obs = read_completed_obs()

    max_window = 2. #allow observations in maximum +/- 2 days
    coord_equinox= 2000.

    constraints = {'max_airmass': 2.0}

    location    = {'telescope_class':'1m0'}

    proposal    = {'proposal_id':'2016B-007',\
                    'user_id':'gsnarayan@gmail.com',\
                    'password':''}

    molecule    = {'ag_mode':'ON',\
                    'ag_name':'',\
                    'bin_x':1,\
                    'bin_y':1,\
                    'defocus':0.0,\
                    'exposure_count':None,\
                    'exposure_time':None,\
                    'fill_window':False,\
                    'filter':"SDSS-g'",\
                    'instrument_name':'1M0-SCICAM-SINISTRO',\
                    'type':'EXPOSE'}

    target = {'name':None,\
                'ra':None,\
                'dec':None,\
                'epoch':None}

    window = {'start':'',\
                'end':''}

    target_info = {}

    nproc = multiprocessing.cpu_count()

    # setup astroplan and astropysics to check the requested observation times
    # there is also the Haleakala site, but it doesn't have a 1meter
    site_names  = ['sso', 'saao', 'ctio', 'mcdonald']
    plan_sites  = [ap.Observer.at_site(site) for site in site_names]
    slew_rate = 2.*u.deg/u.second # slew speed is maximum of 6 degrees per sec - assume it doesn't operate at that
    read_out  = 47.*u.second      # SINISTRO readout time is 47 seconds
    plan_transitioner = ap.Transitioner(slew_rate, None)
    # constrain the observations to be within the max airmass limit
    # in grey time at worst, and at night (duh)
    #plan_constraints = [ap.AirmassConstraint(constraints['max_airmass'], boolean_constraint=False),\
    #                        ap.MoonIlluminationConstraint(min=None,max=0.65),\
    #                        ap.MoonSeparationConstraint(min=18*u.deg,max=None),\
    #                        ap.AtNightConstraint.twilight_astronomical()]
    plan_constraints = [ap.AirmassConstraint(constraints['max_airmass'], boolean_constraint=False),\
                            ap.AtNightConstraint.twilight_astronomical()]
    plan_sched = [PriorityScheduler(constraints=plan_constraints, observer=site,\
                    time_resolution=1.*u.minute, transitioner=plan_transitioner)\
                    for site in plan_sites]


    # The question is not where are we, but when are we?
    today = Time(datetime.datetime.now(),format='datetime')


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
        print("Plan files don't exist, or cannot restore. Creating")
        # Store the requested observing blocks for ALL targets by date
        obs_date_blocks = {}
        target_structures = {}

        processPool = multiprocessing.Pool(nproc)
        lock = multiprocessing.Lock()

        multi_res = []
        for this_target in targets:
            f = target_time_files[this_target.Name]
            args = (this_target, f, completed_obs, target, molecule,\
                    target_moon_illum_limit, target_nsplit, today, coord_equinox, read_out, max_window)
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
                for to in target_obs_date_blocks:
                    blocks = obs_date_blocks.get(to,None)
                    if blocks is None:
                        blocks = []
                    blocks += target_obs_date_blocks[to]
                    obs_date_blocks[to] = blocks
        obsplan = {'target_structures':target_structures, 'obs_date_blocks':obs_date_blocks}
        
        # save the configuration
        with open('obsplan_config_%s.json'%today.isot, 'w') as outpkl:
            json.dump(target_info, outpkl, indent=2, sort_keys=True)
        
        with open('obsplan_config_%s.pkl'%today.isot, 'w') as outpkl:
            pickle.dump(obsplan, outpkl)

    # assign a color to each target 
    color=cm.viridis(np.linspace(0,1,len(targets)))
    target_colors = dict(zip(target_structures.keys(), color))

    ############################ SCHEDULE REQUESTS ############################

    nights = sorted(obs_date_blocks.keys())

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
        
        processPool = multiprocessing.Pool(nproc)
        lock = multiprocessing.Lock()
        
        multi_res = []
        for date in nights:
            blocks = obs_date_blocks[date]
            args = (date, blocks, plan_sites, plan_sched, target_structures, target_colors)
            res  = processPool.apply_async(schedule_and_plot_night, args)
            multi_res.append(res)
        
        processPool.close()
        processPool.join()
        for res in multi_res:
            if res is not None:
                try:
                    date, this_date_site_schedule = res.get()
                    date_site_schedules[date] = this_date_site_schedule
                except Exception, e:
                    warnings.warn('Something threw this bloody exception %s'%e)
                    pass

        with open('date_schedule_%s.pkl'%today.isot, 'w') as outpkl:
            pickle.dump(date_site_schedules, outpkl)
    ############################ CONSTRUCT LCO FORMAT REQUESTS FROM SCHEDULE  ############################


    target_obs_ctr = OrderedCounter()
    unobs_date_blocks = {}
    for date in nights:
        blocks = obs_date_blocks[date]
        this_date_site_schedule = date_site_schedules[date]
        unscheduled_blocks = []
        for block in blocks:
            targetname = block.target.name
            windows = []
            for site in this_date_site_schedule:
                this_site_blocks = this_date_site_schedule[site]
                nsite_blocks = len(this_site_blocks)
                for i in range(nsite_blocks):
                    # block exists, add it's window, pop it, stop update the stored blocks, stop looking
                    if targetname == this_site_blocks[i].target.name:
                        thiswindow = dict(window)
                        thiswindow['start'] = this_site_blocks[i].start_time.iso
                        thiswindow['end'] = this_site_blocks[i].end_time.iso
                        windows.append(thiswindow)
                        this_site_blocks.pop(i)
                        this_date_site_schedule[site] = this_site_blocks
                        break
                    # not the right block, keep looking
                    else:
                        pass
                #end blocks at this site
            #end sites
            if len(windows) == 0:
                unscheduled_blocks.append(block)
            else:
                target = target_info[targetname]['target']
                molecule = target_info[targetname]['molecule']
                count = target_obs_ctr.get(targetname, 1)

                #setup the request
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

                user_request = {
                    "operator" : "single",
                    "requests" : [request],
                    "type" : "compound_request"
                    }

                #convert the request to JSON
                json_user_request = json.dumps(user_request)
                # save the request to a file for a now
                with open('LCO_json/%s_observation_%s_%02i.json'%(targetname, date, count), 'w') as outf:
                    json.dump(user_request, outf, indent=2, sort_keys=True)
                count += 1
                target_obs_ctr[targetname] = count
            #endelse
        #end requested blocks on this date
        unobs_date_blocks[date] = unscheduled_blocks
    #end dates
    unobsplan = {'target_obs_ctr':target_obs_ctr, 'unobs_date_blocks':unobs_date_blocks}
    with open('unscheduled_obs_%s.pkl'%today.isot, 'w') as outpkl:
        pickle.dump(unobsplan, outpkl)

    ############################ CONSTRUCT LCO FORMAT REQUESTS FROM SCHEDULE  ############################
        
#                #params = urllib.urlencode({'username': proposal['user_id'],
#                #                   'password': proposal['password'],
#                #                                  'proposal': proposal['proposal_id'],
#                #                                                 'request_data' : json_user_request})
#
#                ##submit the request
#                #headers = {'Content-type': 'application/x-www-form-urlencoded'}
#                #conn = httplib.HTTPSConnection("lcogt.net")
#                #conn.request("POST", "/observe/service/request/submit", params, headers)
#                #conn_response = conn.getresponse()
#
#                ## The status can tell you if sending the request failed or not.
#                ## 200 or 203 would mean success, 400 or anything else fail
#                #status_code = conn_response.status
#
#                ## If the status was a failure, the response text is a reason why it failed
#                ## If the status was a success, the response text is tracking number of the submitted request
#                #response = conn_response.read()
#
#                #response = json.loads(response)
#                #print response
#
#                #if status_code == 200:
#                #try:
#                #    print "http://lcogt.net/observe/request/" + response['id']
#                #except KeyError:
#                #    print response['error']
#
#
#            # end to observation dateloop
#
#


if __name__=='__main__':
    main()
