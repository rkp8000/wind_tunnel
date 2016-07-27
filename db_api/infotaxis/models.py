import numpy as np
from sqlalchemy import Column, ForeignKey, Sequence
from sqlalchemy import Boolean, Integer, BigInteger, Float, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

from plume import Environment3d

from db_api.infotaxis.connect import engine

Base = declarative_base()


class Simulation(Base):
    __tablename__ = 'simulation'
    _env = None

    id = Column(String(255), primary_key=True)

    description = Column(Text)
    total_trials = Column(Integer)

    xmin = Column(Float)
    xmax = Column(Float)
    nx = Column(Integer)

    ymin = Column(Float)
    ymax = Column(Float)
    ny = Column(Integer)

    zmin = Column(Float)
    zmax = Column(Float)
    nz = Column(Integer)

    dt = Column(Float)

    heading_smoothing = Column(Integer)

    geom_config_group_id = Column(String(255), ForeignKey('geom_config_group.id'))

    plume_id = Column(Integer, ForeignKey('plume.id'))
    insect_id = Column(Integer, ForeignKey('insect.id'))

    ongoing_run_id = Column(Integer, ForeignKey('ongoing_run.id'))

    trials = relationship("Trial", backref='simulation')

    @property
    def env(self):
        if not self._env:
            xbins = np.linspace(self.xmin, self.xmax, self.nx + 1)
            ybins = np.linspace(self.ymin, self.ymax, self.ny + 1)
            zbins = np.linspace(self.zmin, self.zmax, self.nz + 1)
            self._env = Environment3d(xbins, ybins, zbins)
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

        self.xmin = env.xbins[0]
        self.xmax = env.xbins[-1]
        self.nx = env.nx

        self.ymin = env.ybins[0]
        self.ymax = env.ybins[-1]
        self.ny = env.ny

        self.zmin = env.zbins[0]
        self.zmax = env.zbins[-1]
        self.nz = env.nz


class OngoingRun(Base):
    __tablename__ = 'ongoing_run'

    id = Column(Integer, primary_key=True)

    trials_completed = Column(Integer)

    simulations = relationship("Simulation", backref='ongoing_run')


class GeomConfigGroup(Base):
    __tablename__ = 'geom_config_group'

    id = Column(String(255), primary_key=True)

    description = Column(Text)

    simulations = relationship("Simulation", backref='geom_config_group')
    geom_configs = relationship("GeomConfig", backref='geom_config_group')


class GeomConfig(Base):
    __tablename__ = 'geom_config'

    id = Column(Integer, primary_key=True)

    src_xidx = Column(Integer)
    src_yidx = Column(Integer)
    src_zidx = Column(Integer)

    start_xidx = Column(Integer)
    start_yidx = Column(Integer)
    start_zidx = Column(Integer)

    duration = Column(Integer)

    geom_config_group_id = Column(String(255), ForeignKey('geom_config_group.id'))

    trials = relationship("Trial", backref='geom_config')

    @property
    def src_idx(self):
        return self.src_xidx, self.src_yidx, self.src_zidx

    @src_idx.setter
    def src_idx(self, src_idx):
        self.src_xidx, self.src_yidx, self.src_zidx = src_idx

    @property
    def start_idx(self):
        return self.start_xidx, self.start_yidx, self.start_zidx

    @start_idx.setter
    def start_idx(self, start_idx):
        self.start_xidx, self.start_yidx, self.start_zidx = start_idx


class Trial(Base):
    __tablename__ = 'trial'

    id = Column(Integer, primary_key=True)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))
    geom_config_id = Column(Integer, ForeignKey('geom_config.id'))

    trial_info = relationship("TrialInfo", uselist=False, backref='trial')

    start_timepoint_id = Column(BigInteger)
    end_timepoint_id = Column(BigInteger)

    def get_timepoints(self, session):
        """Return all timepoints for this trial."""

        timepoints = session.query(Timepoint). \
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id)). \
            order_by(Timepoint.id)

        return timepoints

    def timepoint_field(self, session, field):
        """Return one field of timepoints associated with this trajectory."""
        data = session.query(getattr(Timepoint, field)). \
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id))

        return np.array(data.all()).flatten()

    @property
    def timepoint_ids_extended(self):

        return np.arange(self.start_timepoint_id, self.end_timepoint_id + 2)


class TrialInfo(Base):
    __tablename__ = 'trial_info'

    id = Column(Integer, primary_key=True)

    trial_id = Column(Integer, ForeignKey('trial.id'))

    duration = Column(Integer)
    found_src = Column(Boolean)


class CrossingGroup(Base):
    __tablename__ = 'crossing_group'

    id = Column(String(255), primary_key=True)
    simulation_id = Column(String(255), ForeignKey('simulation.id'))
    threshold = Column(Float)
    heading_smoothing = Column(Float)

    simulation = relationship("Simulation", backref='crossing_groups')


class Crossing(Base):
    __tablename__ = 'crossing'

    id = Column(Integer, primary_key=True)
    start_timepoint_id = Column(BigInteger)
    entry_timepoint_id = Column(BigInteger)
    peak_timepoint_id = Column(BigInteger)
    exit_timepoint_id = Column(BigInteger)
    end_timepoint_id = Column(BigInteger)
    max_odor = Column(Float)
    crossing_number = Column(Integer)
    trial_id = Column(Integer, ForeignKey('trial.id'))
    crossing_group_id = Column(String(255), ForeignKey('crossing_group.id'))

    trial = relationship("Trial", backref='crossings')
    crossing_group = relationship("CrossingGroup", backref='crossings')

    def timepoint_field(self, session, field, first, last, first_rel_to, last_rel_to, nan_pad=False):
        """
        Get a column of timepoints for a specified time.
        Note: The earliest and latest timepoints that will be returned are those with
        ids equal to start_timepoint_id, and end_timepoint_id, respectively.
        :param session: session
        :param field: what timepoint field you want
        :param first: how many timepoints after first_rel_to timepoint to get
        :param last: how many timepoints after last_rel_to timpeoint to get
        :param first_rel_to: string: 'start', 'entry', 'peak', 'exit', 'end'
        :param last_rel_to: same options as "first_rel_to"
        :param nan_pad: whether or not to pad time-series with nans, should the available timepoints be too few
        :return: 1D array of timepoints
        """
        rel_tos = {'start': self.start_timepoint_id,
                   'entry': self.entry_timepoint_id,
                   'peak': self.peak_timepoint_id,
                   'exit': self.exit_timepoint_id,
                   'end': self.end_timepoint_id}

        start_tp_id = max(rel_tos[first_rel_to] + first, self.start_timepoint_id)
        end_tp_id = min(rel_tos[last_rel_to] + last, self.end_timepoint_id)

        data = np.array(session.query(getattr(Timepoint, field)).
            filter(Timepoint.id.between(start_tp_id, end_tp_id)).all()).flatten()

        if nan_pad:

            diff_start = self.start_timepoint_id - (rel_tos[first_rel_to] + first)
            diff_end = rel_tos[last_rel_to] + last - self.end_timepoint_id

            if diff_start > 0:

                data = np.concatenate([np.nan * np.ones((diff_start,)), data])

            if diff_end > 0:

                data = np.concatenate([data, np.nan * np.ones((diff_end,))])

        return data


class CrossingFeatureSetBasic(Base):
    __tablename__ = 'crossing_feature_set_basic'

    id = Column(Integer, primary_key=True)
    crossing_id = Column(Integer, ForeignKey('crossing.id'))
    position_x_entry = Column(Float)
    position_y_entry = Column(Float)
    position_z_entry = Column(Float)
    position_x_peak = Column(Float)
    position_y_peak = Column(Float)
    position_z_peak = Column(Float)
    position_x_exit = Column(Float)
    position_y_exit = Column(Float)
    position_z_exit = Column(Float)

    heading_xyz_entry = Column(Float)
    heading_xyz_peak = Column(Float)
    heading_xyz_exit = Column(Float)

    crossing = relationship("Crossing", backref=backref('feature_set_basic', uselist=False))


class Insect(Base):
    __tablename__ = 'insect'

    id = Column(Integer, primary_key=True)
    type = Column(String(255))

    simulations = relationship("Simulation", backref='insect')
    insect_params = relationship("InsectParam", backref='insect')


class InsectParam(Base):
    __tablename__ = 'insect_param'

    id = Column(Integer, primary_key=True)

    name = Column(String(50))
    value = Column(Float)

    insect_id = Column(Integer, ForeignKey('insect.id'))


class Plume(Base):
    __tablename__ = 'plume'

    id = Column(Integer, primary_key=True)
    type = Column(String(255))

    simulations = relationship("Simulation", backref='plume')
    plume_params = relationship("PlumeParam", backref='plume')


class PlumeParam(Base):
    __tablename__ = 'plume_param'

    id = Column(Integer, primary_key=True)

    name = Column(String(50))
    value = Column(Float)

    plume_id = Column(Integer, ForeignKey('plume.id'))


class Timepoint(Base):
    __tablename__ = 'timepoint'

    id = Column(BigInteger, primary_key=True)

    xidx = Column(Integer)
    yidx = Column(Integer)
    zidx = Column(Integer)

    hxyz = Column(Float)

    odor = Column(Float)
    detected_odor = Column(Float)

    src_entropy = Column(Float)


class Script(Base):
    __tablename__ = 'script'

    id = Column(String(255), primary_key=True)

    description = Column(Text)
    type = Column(String(255))

    script_executions = relationship("ScriptExecution", backref='script')


class ScriptExecution(Base):
    __tablename__ = 'script_execution'

    id = Column(Integer, primary_key=True)

    script_id = Column(String(255), ForeignKey('script.id'))
    commit = Column(String(255))
    timestamp = Column(DateTime)
    notes = Column(Text)


class GeomConfigExtensionRealTrajectory(Base):
    __tablename__ = 'geom_config_extension_real_trajectory'

    id = Column(Integer, primary_key=True)

    avg_dt = Column(Float)

    geom_config_id = Column(Integer, ForeignKey('geom_config.id'))
    real_trajectory_id = Column(String(255))

    geom_config = relationship("GeomConfig", backref=backref('extension_real_trajectory', uselist=False))


if __name__ == '__main__':

    Base.metadata.create_all(engine)