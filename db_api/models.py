"""
Models for wind tunnel analysis database (and test database).
"""
from __future__ import print_function, division
import numpy as np
from sqlalchemy import Column, ForeignKey
from sqlalchemy import Boolean, Integer, BigInteger, Float, String, Text, DateTime
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base


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

    def positions(self, session):
        """Get positions associated with this trajectory."""
        positions = session.query(Timepoint.position_x, Timepoint.position_y,
                                  Timepoint.position_z).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(positions.all())

    def velocities(self, session):
        """Get velocities associated with this trajectory."""
        velocities = session.query(Timepoint.velocity_x, Timepoint.velocity_y,
                                   Timepoint.velocity_z).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(velocities.all())

    def odors(self, session):
        """Get odor time-series associated with this trajectory."""
        odors = session.query(Timepoint.odor).\
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))
        return np.array(odors.all()).flatten()