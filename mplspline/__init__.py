"""
===============
Spline Editors
===============

Sharing events across GUIs.

"""
import pickle
import numpy as np
from matplotlib.backend_bases import MouseButton
from scipy.interpolate import UnivariateSpline
from collections import namedtuple
from functools import reduce
import scipy.interpolate as si
import numpy.fft as fft


class TooFewPointsException(Exception):
    ...


class SplineCurve:
    mode_param = namedtuple("mode_param", ["n", "x", "y"])
    abs_angle = namedtuple("abs_angle", ["abs", "angle"])

    """
    A class that wraps the scipy.interpolation objects
    """

    @classmethod
    def _get_spline(cls, points, pix_err=2, need_sort=True, **kwargs):
        """
        Returns a closed spline for the points handed in.
        Input is assumed to be a (2xN) array

        =====
        input
        =====

        :param points: the points to fit the spline to
        :type points: a 2xN ndarray or a list of len =2 tuples

        :param pix_err: the error is finding the spline in pixels
        :param need_sort: if the points need to be sorted
            or should be processed as-is

        =====
        output
        =====
        tck
           The return data from the spline fitting
        """

        if type(points) is np.ndarray:
            # make into a list
            pt_lst = list(zip(*points))
            # get center
            center = np.mean(points, axis=1).reshape(2, 1)
        else:
            # make a copy of the list
            pt_lst = list(points)
            # compute center
            center = np.array(
                reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), pt_lst)
            ).reshape(2, 1)
            center /= len(pt_lst)

        if len(pt_lst) < 5:
            raise TooFewPointsException("not enough points")

        if need_sort:
            # sort the list by angle around center
            pt_lst.sort(key=lambda x: np.arctan2(x[1] - center[1], x[0] - center[0]))

        # add first point to end because it is periodic (makes the
        # interpolation code happy)
        pt_lst.append(pt_lst[0])

        # make array for handing in to spline fitting
        pt_array = np.vstack(pt_lst).T
        # do spline fitting

        tck, u = si.splprep(pt_array, s=len(pt_lst) * (pix_err ** 2), per=True)

        return tck

    @classmethod
    def from_pts(cls, new_pts, **kwargs):
        tck = cls._get_spline(new_pts, **kwargs)
        this = cls(tck)
        this.raw_pts = new_pts
        return this

    @classmethod
    def from_pickle_dict(cls, pickle_dict):
        tck = [pickle.loads(str(pickle_dict[_tk])) for _tk in ["tck0", "tck1", "tck2"]]
        return cls(tck)

    def __init__(self, tck):
        """A really hacky way of doing different"""
        self.tck = tck
        self._cntr = None
        self._circ = None
        self._th_offset = None

    @property
    def circ(self):
        """returns a rough estimate of the circumference"""
        if self._circ is None:
            new_pts = si.splev(np.linspace(0, 1, 1000), self.tck, ext=2)
            self._circ = np.sum(np.sqrt(np.sum(np.diff(new_pts, axis=1) ** 2, axis=0)))
        return self._circ

    @property
    def cntr(self):
        """returns a rough estimate of the circumference"""
        if self._cntr is None:
            new_pts = si.splev(np.linspace(0, 1, 1000), self.tck, ext=2)
            self._cntr = np.mean(new_pts, 1)
        return self._cntr

    @property
    def th_offset(self):
        """
        The angle from the y-axis for (x, y) at `phi=0`
        """
        if self._th_offset is None:
            x, y = self.q_phi_to_xy(0, 0) - self.cntr.reshape(2, 1)
            self._th_offset = np.arctan2(y, x)
        return self._th_offset

    @property
    def tck0(self):
        return self.tck[0]

    @property
    def tck1(self):
        return self.tck[1]

    @property
    def tck2(self):
        return self.tck[2]

    @property
    def to_pickle_dict(self):
        return dict(
            (lab, pickle.dumps(getattr(self, lab))) for lab in ["tck0", "tck1", "tck2"]
        )

    def q_phi_to_xy(self, q, phi, cross=None):
        """Converts q, phi pairs -> x, y pairs.  All other code that
        does this should move to using this so that there is minimal
        breakage when we change over to using additive q instead of
        multiplicative"""
        # make sure data is arrays
        q = np.asarray(q)
        # convert real units -> interpolation units
        phi = np.mod(np.asarray(phi), 2 * np.pi) / (2 * np.pi)
        # get the shapes
        q_shape, phi_shape = [
            _.shape if (_.shape != () and len(_) > 1) else None for _ in (q, phi)
        ]

        # flatten everything
        q = q.ravel()
        phi = phi.ravel()
        # sanity checks on shapes
        if cross is False:
            if phi_shape != q_shape:
                raise ValueError(
                    "q and phi must have same" + " dimensions to broadcast"
                )
        if cross is None:
            if (
                (phi_shape is not None)
                and (q_shape is not None)
                and (phi_shape == q_shape)
            ):
                cross = False
            elif q_shape is None:
                cross = False
                q = q[0]
            else:
                cross = True

        x, y = si.splev(phi, self.tck, ext=2)
        dx, dy = si.splev(phi, self.tck, der=1, ext=2)
        norm = np.sqrt(dx ** 2 + dy ** 2)
        nx, ny = dy / norm, -dx / norm

        # if cross, then
        if cross:
            data_out = list(
                zip(
                    *[
                        (
                            (x + q_ * nx).reshape(phi_shape),
                            (y + q_ * ny).reshape(phi_shape),
                        )
                        for q_ in q
                    ]
                )
            )
        else:

            data_out = np.vstack(
                [(x + q * nx).reshape(phi_shape), (y + q * ny).reshape(phi_shape)]
            )

        return data_out

    def fft_filter(self, mode):
        if mode == 0:
            return
        sample_pow = 12
        tmp_pts = si.splev(np.linspace(0, 1, 2 ** sample_pow), self.tck)

        mask = np.zeros(2 ** sample_pow)
        mask[0] = 1
        mask[1 : (mode + 1)] = 1
        mask[-mode:] = 1

        new_pts = []
        for w in tmp_pts:
            wfft = fft.fft(w)
            new_pts.append(np.real(fft.ifft(wfft * mask)))

        new_pts = np.vstack(new_pts)

        tck = self._get_spline(new_pts, pix_err=0.05, need_sort=False)

        self.tck = tck

    def draw_to_axes(self, ax, N=1024, **kwargs):
        return ax.plot(
            *(self.q_phi_to_xy(0, np.linspace(0, 2 * np.pi, N)) + 0.5), **kwargs
        )

    def curve_shape_fft(self, N=3):
        """
        Returns the amplitude and phase of the components of the rim curve

        Parameters
        ----------
        self : SplineCurve
            The curve to extract the data from

        n : int
            The maximum mode to extract data for

        Returns
        -------
        ret : list
            [mode_param(n=n, x=abs_angle(x_amp, x_phase),
                        y=abs_angle(y_amp, y_phase)), ...]
        """
        curve_data = self.q_phi_to_xy(1, np.linspace(0, 2 * np.pi, 1000))
        curve_fft = [np.fft.fft(_d) / len(_d) for _d in curve_data]
        return [
            self.mode_param(
                n,
                *[
                    self.abs_angle(2 * np.abs(_cfft[n]), np.angle(_cfft[n]))
                    for _cfft in curve_fft
                ],
            )
            for n in range(1, N + 1)
        ]

    def cum_length(self, N=1024):
        """Returns the cumulative length at N evenly
        sampled points in parameter space

        Parameters
        ----------
        N : int
            Number of points to sample

        Returns
        -------
        ret : ndarray
            An ndarray of length N which is the cumulative distance
            around the rim
        """
        # turns out you _can_ write un-readable python
        return np.concatenate(
            (
                [0],
                np.cumsum(
                    np.sqrt(
                        np.sum(
                            np.diff(
                                si.splev(np.linspace(0, 1, N), self.tck, ext=2), axis=1
                            )
                            ** 2,
                            axis=0,
                        )
                    )
                ),
            )
        )

    def cum_length_theta(self, N=1024):
        """Returns the cumulative length evenly sampled in theta space.  Does by
        evenly sampling the rim in spline units and the interpolating to
        evenly spaced theta positions.

        Parameters
        ----------
        N : int
            Number of points to sample

        Returns
        -------
        ret : ndarray
            An ndarray of length N which is the cumulative distance
            around the rim
        """
        intep_func = si.interp1d
        cntr = self.cntr.reshape(2, 1)
        # over sample in spline space
        XY = si.splev(np.linspace(0, 1, 2 * N), self.tck, ext=2) - cntr
        theta = np.mod(np.arctan2(XY[1], XY[0]), 2 * np.pi)
        indx = np.argsort(theta)
        XY = XY[:, indx]
        theta = theta[indx]
        # pad one past the end
        theta = np.r_[theta[-1] - 2 * np.pi, theta, theta[0] + 2 * np.pi]
        XY = np.c_[XY[:, -1:], XY, XY[:, :1]]
        # the sample points
        sample_theta = np.linspace(0, 2 * np.pi, N)
        # re-sample
        XY_resample = np.vstack([intep_func(theta, _xy)(sample_theta) for _xy in XY])
        return np.concatenate(
            ([0], np.cumsum(np.sqrt(np.sum(np.diff(XY_resample, axis=1) ** 2, axis=0))))
        )


class PeriodicSplineInteractor:
    def __init__(self, ax, pix_err=1):
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self.pt_lst = []
        self.pt_plot = ax.plot([], [], marker="o", linestyle="none", zorder=5)[0]
        self.sp_plot = ax.plot([], [], lw=3, color="r")[0]
        self.pix_err = pix_err
        self.connect_sf()

    def set_visible(self, visible):
        """sets if the curves are visible """
        self.pt_plot.set_visible(visible)
        self.sp_plot.set_visible(visible)

    def clear(self):
        """Clears the points"""
        self.pt_lst = []
        self.redraw()

    def connect_sf(self):
        if self.cid is None:
            self.cid = self.canvas.mpl_connect("button_press_event", self.click_event)

    def disconnect_sf(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def click_event(self, event):
        """ Extracts locations from the user"""
        if event.key == "shift":
            self.pt_lst = []

        elif event.xdata is None or event.ydata is None:
            return
        elif event.button == 1:
            self.pt_lst.append((event.xdata, event.ydata))
        elif event.button == 3:
            self.remove_pt((event.xdata, event.ydata))

        self.redraw()

    def remove_pt(self, loc):
        if len(self.pt_lst) > 0:
            self.pt_lst.pop(
                np.argmin(
                    [
                        np.sqrt((x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2)
                        for x in self.pt_lst
                    ]
                )
            )

    def redraw(self):
        if len(self.pt_lst) > 5:
            SC = SplineCurve.from_pts(self.pt_lst, pix_err=self.pix_err)
            new_pts = SC.q_phi_to_xy(0, np.linspace(0, 2 * np.pi, 1000))
            center = SC.cntr
            self.sp_plot.set_xdata(new_pts[0])
            self.sp_plot.set_ydata(new_pts[1])
            self.pt_lst.sort(
                key=lambda x: np.arctan2(x[1] - center[1], x[0] - center[0])
            )
        else:
            self.sp_plot.set_xdata([])
            self.sp_plot.set_ydata([])
        if len(self.pt_lst) > 0:
            x, y = list(zip(*self.pt_lst))
        else:
            x, y = [], []
        self.pt_plot.set_xdata(x)
        self.pt_plot.set_ydata(y)

        self.canvas.draw()

    def return_points(self):
        """Returns the clicked points in the format the rest of the
        code expects"""
        return np.vstack(self.pt_lst).T

    def return_SplineCurve(self):
        curve = SplineCurve.from_pts(self.pt_lst, pix_err=self.pix_err)
        print(curve.circ)
        return curve


class SplineInteractor:
    """
    A Spline editor.
    Parameters
    ----------
    ax : Axes
    x, y : array-like
       The initial control points
    """

    showverts = True
    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, ax, x, y, s=None, k=3):
        self.ax = ax
        canvas = self.ax.figure.canvas
        # state to configure the spline
        self._s = s
        self._k = k
        # stash the initial data.  Make copy because we will be
        # mutating these.
        self.x, self.y = np.array(x), np.array(y)
        # line to show the control points
        (self.line,) = ax.plot(
            x,
            y,
            marker="o",
            markerfacecolor="r",
            lw=1,
            animated=True,
            color="xkcd:coral pink",
            label="selected",
        )
        # line to show the sampled results
        (self.spline_line,) = ax.plot(
            x,
            y,
            animated=True,
            lw=3,
            color="xkcd:flat green",
            label="smoothed spline",
        )
        self._ind = None  # the active vertex
        self.canvas = canvas
        self.background = None  # for blitting
        # hookup the callbacks
        canvas.mpl_connect("draw_event", self._on_draw)
        canvas.mpl_connect("button_press_event", self._on_button_press)
        canvas.mpl_connect("key_press_event", self._on_key_press)
        canvas.mpl_connect("button_release_event", self._on_button_release)
        canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def sample(self, sample_x, **kwargs):
        """
        Sample the spline at the given points.
        kwargs passed through to `scipy.interpolate.UnivariateSpline.__call__`
        Parameters
        ----------
        sample_x : array-like
            The points to given points
        Returns
        -------
        sample_y : array-like
        """
        return self.spline(sample_x, **kwargs)

    @property
    def spline(self):
        """
        The current spline
        """
        *_, interp = self._get_interp()
        return interp

    @property
    def s(self):
        """
        The smoothing factor to pass to the spline
        Parameters
        ----------
        s : float or None
        """
        return self._s

    @s.setter
    def s(self, value):
        if value is not None and value < 0:
            raise ValueError(f"s must be None or a positive number not {value}")
        self._s = value
        self._update_lines()

    @property
    def k(self):
        """
        The order of the spline
        Parameters
        ----------
        k : int 1 <= k <=5
        """
        return self._k

    @k.setter
    def k(self, value):
        if not (1 <= value <= 5):
            raise ValueError(f"Only supports k in [1, 5], you passed: {value}")
        self._k = value
        self._update_lines()

    def _get_interp(self):
        verts = np.vstack([self.x, self.y]).T
        x, y = verts[np.argsort(verts[:, 0]), :].T
        interp = UnivariateSpline(x, y, s=self.s, k=self.k)
        return x, y, interp

    def _update_lines(self):
        if self.background is None:
            # Not drawn at least once, can not blit!
            return
        x, y, interp = self._get_interp()
        sample_x = np.linspace(x[0], x[-1], 1024)
        sample_y = interp(sample_x)
        self.line.set_data(x, y)
        self.spline_line.set_data(sample_x, sample_y)
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.spline_line)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xy = np.vstack([self.x, self.y]).T
        xyt = self.line.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        ind = d.argmin()
        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def _on_draw(self, event):
        """Callback for draws."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self._update_lines()

    def _on_button_press(self, event):
        """Callback for mouse button presses."""
        if (
            event.inaxes is None
            or event.button != MouseButton.LEFT
            or not self.showverts
        ):
            return
        self._ind = self.get_ind_under_point(event)

    def _on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != MouseButton.LEFT or not self.showverts:
            return
        self._ind = None

    def _on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == "t":
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        self.canvas.draw()

    def _on_mouse_move(self, event):
        """Callback for mouse movements."""
        if (
            self._ind is None
            or event.inaxes is None
            or event.button != MouseButton.LEFT
            or not self.showverts
        ):
            return
        self.x[self._ind] = event.xdata
        self.y[self._ind] = event.ydata
        self._update_lines()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax_dict = plt.subplot_mosaic([["1D", "2D"]])
    x_init = np.linspace(-3, 3, 15)
    y_init = np.sin(x_init)
    interactor = SplineInteractor(
        ax_dict["1D"],
        x_init,
        y_init,
    )

    pi = PeriodicSplineInteractor(ax_dict["2D"], pix_err=0.001)
    plt.show()
