#!/usr/bin/env python3
"""Did the policy learn to stop, or is it predicting the dataset mean?

Most of a pick episode is spent not driving: the base approaches, stops, and the
arm does the rest. In the demonstrations the base is moving in 33.8% of the
commands. A policy that has learned the task reproduces that; one that has only
learned the marginal distribution of the actions drives all the time, at roughly
the dataset's mean speed, and sails past the object.

That is what both fine-tunes did at ~1000 steps -- 71.5% moving, against 33.8%
in the demonstrations -- and it is why the 247-episode run scored *worse* on
distance-to-object than the 74-episode one despite reproducing the
demonstrations well open loop: it drove faster, so it ended up further away.

    python did_the_policy_stop.py /home/bags/eval_v2_1000 /home/bags/pick_e_00

Compare an evaluation bag against a collection bag. The number to look at is
"moving"; the mean velocities say how close the output is to the dataset mean.
"""

import numpy as np, pathlib, sys
from rosbags.highlevel import AnyReader
for bag in sys.argv[1:]:
    p = pathlib.Path(bag)
    with AnyReader([p]) as r:
        v = []
        conns = [c for c in r.connections if c.topic == "/omni_base_controller/cmd_vel"]
        for conn, stamp, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            v.append((m.linear.x, m.linear.y, m.angular.z))
    v = np.array(v)
    if not len(v):
        print(f"=== {p.name}: no cmd_vel"); continue
    speed = np.hypot(v[:, 0], v[:, 1])
    print(f"=== {p.name}  ({len(v)} commands)")
    print(f"  vx     mean={v[:,0].mean():7.4f}  std={v[:,0].std():6.4f}  min={v[:,0].min():7.3f} max={v[:,0].max():7.3f}")
    print(f"  vy     mean={v[:,1].mean():7.4f}  std={v[:,1].std():6.4f}")
    print(f"  wz     mean={v[:,2].mean():7.4f}  std={v[:,2].std():6.4f}")
    print(f"  |v|    mean={speed.mean():7.4f}   moving (>0.01): {100*np.mean(speed>0.01):5.1f}%")
    print(f"  net displacement if integrated at 10 Hz: {np.abs(v[:,0].sum())/10:.2f} m along x")
