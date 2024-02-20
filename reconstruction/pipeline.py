import datetime
import sys
import time

from reconstruction.color_map_optimization import run as color_map_optimization
from reconstruction.initialize_config import initialize_config
from reconstruction.integrate_scene import run as integrate_scene
from reconstruction.make_fragments import run as make_fragments
from reconstruction.refine_registration import run as refine_registration
from reconstruction.register_fragments import run as register_fragments


def pipeline(config, slac=False, debug=False):
    config = {**config, "debug_mode": debug}

    initialize_config(config)

    times = []
    start_time = time.time()
    make_fragments(config)
    times.append(time.time() - start_time)

    start_time = time.time()
    register_fragments(config)
    times.append(time.time() - start_time)

    start_time = time.time()
    refine_registration(config)
    times.append(time.time() - start_time)

    start_time = time.time()
    integrate_scene(config)
    times.append(time.time() - start_time)

    start_time = time.time()
    color_map_optimization(config)
    times.append(time.time() - start_time)

    print("====================================")
    print("Elapsed time (in h:m:s.ms)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- Optimize color map  %s" % datetime.timedelta(seconds=times[4]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))

    return config["depth_scale"]
