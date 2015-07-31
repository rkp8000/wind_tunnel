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


class SampleTrajectory(Base):
    __tablename__ = 'sample_trajectory'

    id = Column(Integer, primary_key=True)
    sample_group = Column(String(255))
    experiment_id = Column(String(255), ForeignKey('experiment.id'))
    odor_state = Column(String(255))
    trajectory_id = Column(String(100), ForeignKey('trajectory.id'))

    experiment = relationship("Experiment", backref='sample_trajectories')
    trajectory = relationship("Trajectory", backref=backref('sample_trajectory', uselist=False))