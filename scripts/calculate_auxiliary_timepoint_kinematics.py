"""
Loop through all trajectories and calculate auxiliary timepoint quantities.
Calculate:
    speed
    acceleration
    heading
    angular velocity
    distance from wall
"""
from __future__ import print_function, division
import numpy as np
from db_api.connect import session, commit
from db_api import models
import kinematics

WALL_BOUNDS = ((-0.3, 1.), (-0.15, 0.15), (-0.15, 0.15))


def main():
    # loop over all experiments
    for expt in session.query(models.Experiment):

        print("In experiment '{}'".format(expt.id))

        dt = 1 / expt.sampling_frequency

        for traj in session.query(models.Trajectory).filter_by(experiment=expt):

            positions = traj.positions(session)
            velocities = traj.velocities(session)

            # calculate kinematic quantities
            velocities_a = kinematics.norm(velocities)

            accelerations = kinematics.acceleration(velocities, dt)
            accelerations_a = kinematics.norm(accelerations)

            headings = kinematics.heading(velocities)

            angular_velocities = kinematics.angular_velocity(velocities, dt)
            angular_velocities_a = kinematics.norm(angular_velocities)

            angular_accelerations = kinematics.acceleration(angular_velocities, dt)
            angular_accelerations_a = kinematics.norm(angular_accelerations)

            distance_from_wall = kinematics.distance_from_wall(positions, WALL_BOUNDS)

            # store kinematic quantities in timepoints
            for ctr, tp in enumerate(traj.timepoints(session)):

                tp.velocity_a = velocities_a[ctr]
                tp.acceleration_x, tp.acceleration_y, tp.acceleration_z = accelerations[ctr]
                tp.acceleration_a = accelerations_a[ctr]

                tp.heading_xy, tp.heading_xz, tp.heading_xyz = headings[ctr]

                tp.angular_velocity_x, tp.angular_velocity_y, tp.angular_velocity_z = angular_velocities[ctr]
                tp.angular_velocity_a = angular_velocities_a[ctr]

                tp.angular_acceleration_x, tp.angular_acceleration_y, tp.angular_acceleration_z = angular_accelerations[ctr]
                tp.angular_acceleration_a = angular_accelerations_a[ctr]

                tp.distance_from_wall = distance_from_wall[ctr]

                session.add(tp)

            commit(session)


if __name__ == '__main__':
    main()