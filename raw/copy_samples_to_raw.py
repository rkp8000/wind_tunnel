"""
Copy all of the trajectories in wind_tunnel_raw_db that have been marked as samples into wind_tunnel_raw_sample_db.
"""
from __future__ import print_function, division
from connect import session, session_sample, engine_sample
import models


# make sure we got da correct tables
models.Base.metadata.create_all(engine_sample)

# copy all experiments
experiments = session.query(models.Experiment)
for experiment in experiments:
    experiment_sample = models.Experiment(id=experiment.id,
                                          file_name=experiment.file_name,
                                          insect=experiment.insect,
                                          odorant=experiment.odorant,
                                          wind_speed=experiment.wind_speed,
                                          visual_environment=experiment.visual_environment,
                                          sampling_frequency=experiment.sampling_frequency)
    session_sample.add(experiment_sample)
    session_sample.commit()

    # add sample trajectories
    for sample_traj in experiment.sample_trajectories:
        # get positions, velocities, and odors
        positions = sample_traj.trajectory.positions(session)
        velocities = sample_traj.trajectory.velocities(session)
        odors = sample_traj.trajectory.odors(session)

        # add timepoints
        for ctr in range(len(positions)):

            # add next timepoint
            tp = models.Timepoint(timestep=ctr,
                                  position_x=positions[ctr][0],
                                  position_y=positions[ctr][1],
                                  position_z=positions[ctr][2],
                                  velocity_x=velocities[ctr][0],
                                  velocity_y=velocities[ctr][1],
                                  velocity_z=velocities[ctr][2],
                                  odor=odors[ctr])
            session_sample.add(tp)

            # get start (and end) tp id if first timepoint
            if ctr == 0:
                session_sample.flush()
                start_tp_id = tp.id
                end_tp_id = tp.id + len(positions) - 1

        # make trajectory
        traj = models.Trajectory(id=sample_traj.trajectory.id,
                                 start_timepoint_id=start_tp_id,
                                 end_timepoint_id=end_tp_id,
                                 raw=True,
                                 clean=False,
                                 odor_state=sample_traj.odor_state,
                                 experiment=experiment_sample)

        session_sample.add(traj)
        session_sample.commit()