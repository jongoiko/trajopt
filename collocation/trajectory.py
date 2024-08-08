import abc


class Trajectory(abc.ABC):
    def __init__(self, t, x, u, x_dot):
        self.t_0, self.t_f = t.min(), t.max()
        self._t = t
        self._x = x
        self._u = u
        self._x_dot = x_dot

    @abc.abstractmethod
    def interpolate(self, t):
        pass

    @abc.abstractmethod
    def _approx_dynamics(self, t):
        pass
