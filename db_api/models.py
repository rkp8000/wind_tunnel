"""
Models for wind tunnel analysis database (and test database).
"""
from __future__ import print_function, division
import numpy as np
from sqlalchemy import Column, ForeignKey
from sqlalchemy import Boolean, Integer, BigInteger, Float, String, Text, DateTime
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from connect import engine

Base = declarative_base()


class Experiment(Base):
    __tablename__ = 'experiment'

    id = Column(String(255), primary_key=True)
    file_name = Column(String(255))
    insect = Column(String(50))
    odorant = Column(String(50))
    wind_speed = Column(Float)
    visual_environment = Column(String(100))
    sampling_frequency = Column(Float)


class Timepoint(Base):
    __tablename__ = 'timepoint'

    id = Column(BigInteger, primary_key=True)
    timestep = Column(Integer)
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    velocity_x = Column(Float)
    velocity_y = Column(Float)
    velocity_z = Column(Float)
    odor = Column(Float)
    velocity_a = Column(Float)
    acceleration_x = Column(Float)
    acceleration_y = Column(Float)
    acceleration_z = Column(Float)
    acceleration_a = Column(Float)
    heading_xy = Column(Float)
    heading_xz = Column(Float)
    heading_xyz = Column(Float)
    angular_velocity_x = Column(Float)
    angular_velocity_y = Column(Float)
    angular_velocity_z = Column(Float)
    angular_velocity_a = Column(Float)
    angular_acceleration_x = Column(Float)
    angular_acceleration_y = Column(Float)
    angular_acceleration_z = Column(Float)
    angular_acceleration_a = Column(Float)
    distance_from_wall = Column(Float)


class Trajectory(Base):
    __tablename__ = 'trajectory'

    id = Column(String(100), primary_key=True)
    start_timepoint_id = Column(BigInteger)
    end_timepoint_id = Column(BigInteger)
    experiment_id = Column(String(255), ForeignKey('experiment.id'))
    raw = Column(Boolean)
    clean = Column(Boolean)
    odor_state = Column(String(50))

    experiment = relationship("Experiment", backref='trajectories')

    def timepoints(self, session):
        """Return all timepoints associated with this trajectory."""
        tps = session.query(Timepoint).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id)).all()

        return tps

    # original quantities
    def positions(self, session):
        """Get positions associated with this trajectory."""
        positions = session.query(Timepoint.position_x, Timepoint.position_y, Timepoint.position_z).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(positions.all())

    def velocities(self, session):
        """Get velocities associated with this trajectory."""
        velocities = session.query(Timepoint.velocity_x, Timepoint.velocity_y, Timepoint.velocity_z).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(velocities.all())

    def odors(self, session):
        """Get odor time-series associated with this trajectory."""
        odors = session.query(Timepoint.odor).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(odors.all()).flatten()

    # auxiliary quantities
    def velocities_a(self, session):
        velocities_a = session.query(Timepoint.velocity_a).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(velocities_a.all()).flatten()

    def accelerations(self, session):
        accelerations = session.query(Timepoint.acceleration_x, Timepoint.acceleration_y, Timepoint.acceleration_z).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(accelerations.all())

    def accelerations_a(self, session):
        accelerations_a = session.query(Timepoint.acceleration_a).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(accelerations_a.all()).flatten()

    def headings(self, session):
        headings = session.query(Timepoint.heading_xy, Timepoint.heading_xz, Timepoint.heading_xyz).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(headings.all())

    def angular_velocities(self, session):
        angular_velocities = session.query(Timepoint.angular_velocity_x,
                                           Timepoint.angular_velocity_y,
                                           Timepoint.angular_velocity_z).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(angular_velocities.all())

    def angular_velocities_a(self, session):
        angular_velocities_a = session.query(Timepoint.angular_velocity_a).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(angular_velocities_a.all()).flatten()

    def angular_accelerations(self, session):
        angular_accelerations = session.query(Timepoint.angular_acceleration_x,
                                              Timepoint.angular_acceleration_y,
                                              Timepoint.angular_acceleration_z).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(angular_accelerations.all())

    def angular_accelerations_a(self, session):
        angular_accelerations_a = session.query(Timepoint.angular_acceleration_a).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(angular_accelerations_a.all()).flatten()

    def distances_from_wall(self, session):
        distances_from_wall = session.query(Timepoint.distance_from_wall).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(distances_from_wall.all()).flatten()


class TrajectoryCleaningParameter(Base):
    __tablename__ = 'trajectory_cleaning_parameter'
    id = Column(Integer, primary_key=True)
    insect = Column(String(20))
    param = Column(String(30))
    value = Column(Float)


class TrajectoryBasicInfo(Base):
    __tablename__ = 'trajectory_basic_info'
    id = Column(Integer, primary_key=True)
    trajectory_id = Column(String(100), ForeignKey('trajectory.id'))
    start_position_x = Column(Float)
    start_position_y = Column(Float)
    start_position_z = Column(Float)
    end_position_x = Column(Float)
    end_position_y = Column(Float)
    end_position_z = Column(Float)
    distance_from_wall_start = Column(Float)
    distance_from_wall_end = Column(Float)
    start_datetime = Column(DateTime)
    duration = Column(Float)

    trajectory = relationship("Trajectory", backref=backref('basic_info', uselist=False))


if __name__ == '__main__':
    Base.metadata.create_all(engine)