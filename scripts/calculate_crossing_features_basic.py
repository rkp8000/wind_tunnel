"""
Store basic crossing features in database.
These include: position_x, position_y, position_z, heading_xy, heading_xz, heading_xyz for entry, peak, and exit
timepoints.
"""
from __future__ import print_function, division
import numpy as np
from db_api import models
from db_api.connect import session, commit

DETERMINATION = 'chosen0'


def main():

    for expt in session.query(models.Experiment):
        threshold = session.query(models.Threshold).filter_by(experiment=expt,
                                                              determination=DETERMINATION).first()
        for cg in threshold.crossing_groups:
            print(cg.id)
            for crossing in cg.crossings:

                position_x_entry = crossing.timepoint_field(session, 'position_x', 0, 0, 'entry', 'entry')[0]
                position_y_entry = crossing.timepoint_field(session, 'position_y', 0, 0, 'entry', 'entry')[0]
                position_z_entry = crossing.timepoint_field(session, 'position_z', 0, 0, 'entry', 'entry')[0]

                position_x_peak = crossing.timepoint_field(session, 'position_x', 0, 0, 'peak', 'peak')[0]
                position_y_peak = crossing.timepoint_field(session, 'position_y', 0, 0, 'peak', 'peak')[0]
                position_z_peak = crossing.timepoint_field(session, 'position_z', 0, 0, 'peak', 'peak')[0]

                position_x_exit = crossing.timepoint_field(session, 'position_x', 0, 0, 'exit', 'exit')[0]
                position_y_exit = crossing.timepoint_field(session, 'position_y', 0, 0, 'exit', 'exit')[0]
                position_z_exit = crossing.timepoint_field(session, 'position_z', 0, 0, 'exit', 'exit')[0]

                heading_xy_entry = crossing.timepoint_field(session, 'heading_xy', 0, 0, 'entry', 'entry')[0]
                heading_xz_entry = crossing.timepoint_field(session, 'heading_xz', 0, 0, 'entry', 'entry')[0]
                heading_xyz_entry = crossing.timepoint_field(session, 'heading_xyz', 0, 0, 'entry', 'entry')[0]

                heading_xy_peak = crossing.timepoint_field(session, 'heading_xy', 0, 0, 'peak', 'peak')[0]
                heading_xz_peak = crossing.timepoint_field(session, 'heading_xz', 0, 0, 'peak', 'peak')[0]
                heading_xyz_peak = crossing.timepoint_field(session, 'heading_xyz', 0, 0, 'peak', 'peak')[0]

                heading_xy_exit = crossing.timepoint_field(session, 'heading_xy', 0, 0, 'exit', 'exit')[0]
                heading_xz_exit = crossing.timepoint_field(session, 'heading_xz', 0, 0, 'exit', 'exit')[0]
                heading_xyz_exit = crossing.timepoint_field(session, 'heading_xyz', 0, 0, 'exit', 'exit')[0]

                crossing.feature_set_basic = models.CrossingFeatureSetBasic(position_x_entry=position_x_entry,
                                                                            position_y_entry=position_y_entry,
                                                                            position_z_entry=position_z_entry,
                                                                            position_x_peak=position_x_peak,
                                                                            position_y_peak=position_y_peak,
                                                                            position_z_peak=position_z_peak,
                                                                            position_x_exit=position_x_exit,
                                                                            position_y_exit=position_y_exit,
                                                                            position_z_exit=position_z_exit,
                                                                            heading_xy_entry=heading_xy_entry,
                                                                            heading_xz_entry=heading_xz_entry,
                                                                            heading_xyz_entry=heading_xyz_entry,
                                                                            heading_xy_peak=heading_xy_peak,
                                                                            heading_xz_peak=heading_xz_peak,
                                                                            heading_xyz_peak=heading_xyz_peak,
                                                                            heading_xy_exit=heading_xy_exit,
                                                                            heading_xz_exit=heading_xz_exit,
                                                                            heading_xyz_exit=heading_xyz_exit)

                session.add(crossing)
                commit(session)


if __name__ == '__main__':
    main()