"""Landmark Registration of functional data module.

This module contains methods to perform the landmark registration.
"""

import numpy as np

from ... import FDataGrid
from ...representation.interpolation import SplineInterpolator

from .base import RegistrationTransformer

__author__ = "Pablo Marcos Manchón"
__email__ = "pablo.marcosm@estudiante.uam.es"

class LandmarkShift(RegistrationTransformer):
    r"""Perform a shift of the curves to align the landmarks.

        Let :math:`t^*` the time where the landmarks of the curves will be
        aligned, :math:`t_i` the location of the landmarks for each curve
        and :math:`\delta_i= t_i - t^*`.

        The registered samples will have their feature aligned.

        .. math::
            x_i^*(t^*)=x_i(t^* + \delta_i)=x_i(t_i)

    Args:
        fd (:class:`FData`): Functional data object.
        landmarks (array_like): List with the landmarks of the samples.
        location (numeric or callable, optional): Defines where
            the landmarks will be alligned. If a numeric value is passed the
            landmarks will be alligned to it. In case of a callable is
            passed the location will be the result of the the call, the
            function should be accept as an unique parameter a numpy array
            with the list of landmarks.
            By default it will be used as location :math:`\frac{1}{2}(max(
            \text{landmarks})+ min(\text{landmarks}))` wich minimizes the
            max shift.
        restrict_domain (bool, optional): If True restricts the domain to
            avoid evaluate points outside the domain using extrapolation.
            Defaults uses extrapolation.
        extrapolation (str or Extrapolation, optional): Controls the
            extrapolation mode for elements outside the domain range.
            By default uses the method defined in fd. See extrapolation to
            more information.
        eval_points (array_like, optional): Set of points where
            the functions are evaluated in :func:`shift`.
        **kwargs: Keyword arguments to be passed to :func:`shift`.

    Returns:
        :class:`FData`: Functional data object with the registered samples.

    Examples:

        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import landmark_shift

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, random_state=1)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        >>> landmarks = landmarks.squeeze()

        The function will return the sample registered

        >>> landmark_shift(fd, landmarks)
        FDataGrid(...)

    """
    def __init__(self, landmark_method=None, location=None, output_points=None,
                 restrict_domain=False, extrapolation=None):
        """

        """
            self.landmark_method = landmark_method
            self.location = location
            self.output_points = output_points
            self.restrict_domain = restrict_domain
            self.extrapolation = extrapolation

    def fit_transform(fd, landmarks, location=None, *, restrict_domain=False,
                      extrapolation=None, eval_points=None):
        """

        """
        if len(landmarks) != fd.n_samples:
            raise ValueError(f"landmark list ({len(landmarks)}) must have the "
                             f"same length than the number of samples "
                             f"({fd.n_samples})")

        landmarks = np.atleast_1d(landmarks)

        # Parses location
        if location is None:
            p = (np.max(landmarks, axis=0) + np.min(landmarks, axis=0)) / 2.
        elif callable(location):
            p = location(landmarks)
        else:
            try:
                p = np.atleast_1d(location)
            except:
                raise ValueError("Invalid location, must be None, a callable "
                                 "or a number in the domain")

        self.shifts_ = landmarks - p

        return fd.shift(self.shifts_, restrict_domain=restrict_domain,
                        extrapolation=extrapolation, eval_points=eval_points)


class LandmarkRegistration(RegistrationTransformer):
    """Perform landmark registration of the curves.

        Let :math:`t_{ij}` the time where the sample :math:`i` has the feature
        :math:`j` and :math:`t^*_j` the new time for the feature.
        The registered samples will have their features aligned, i.e.,
        :math:`x^*_i(t^*_j)=x_i(t_{ij})`.

        See [RS05-7-3]_ for a detailed explanation.

    Args:
        fd (:class:`FData`): Functional data object.
        landmarks (array_like): List containing landmarks for each samples.
        location (array_like, optional): Defines where
            the landmarks will be alligned. By default it will be used as
            location the mean of the landmarks.
        eval_points (array_like, optional): Set of points where
            the functions are evaluated to obtain a discrete
            representation of the object. In case of objects with
            multidimensional domain a list axis with points of evaluation
            for each dimension.

    Returns:
        :class:`FData`: FData with the functional data object registered.

    References:

    ..  [RS05-7-3] Ramsay, J., Silverman, B. W. (2005). Feature or landmark
        registration. In *Functional Data Analysis* (pp. 132-136). Springer.

    Examples:

        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import landmark_registration
        >>> from skfda.representation.basis import BSpline

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, n_modes=2,
        ...                              random_state=9)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
        ...                                       random_state=9)
        >>> landmarks = landmarks.squeeze()

        The function will return the registered curves

        >>> landmark_registration(fd, landmarks)
        FDataGrid(...)

        This method will work for FDataBasis as for FDataGrids

        >>> fd = fd.to_basis(BSpline(n_basis=12))
        >>> landmark_registration(fd, landmarks)
        FDataBasis(...)

    """
    def __init__(self, landmark_method=None, location=None,
                 output_points=None):
        """

        """
        self.landmark_method=landmark_method
        self.location=location
        self.output_points=output_points

    def fit_transform(fd, landmarks, *, location=None, eval_points=None):
        """


        """

        if fd.dim_domain > 1:
            raise NotImplementedError("Method only implemented for objects "
                                      "with domain dimension up to 1.")

        if len(landmarks) != fd.n_samples:
            raise ValueError("The number of list of landmarks should be equal "
                             "to the number of samples")

        landmarks = np.asarray(landmarks).reshape((fd.n_samples, -1))

        n_landmarks = landmarks.shape[-1]

        data_matrix = np.empty((fd.n_samples, n_landmarks + 2))

        data_matrix[:, 0] = fd.domain_range[0][0]
        data_matrix[:, -1] = fd.domain_range[0][1]

        data_matrix[:, 1:-1] = landmarks

        if location is None:
            sample_points = np.mean(data_matrix, axis=0)

        elif n_landmarks != len(location):

            raise ValueError(f"Number of landmark locations should be equal "
                             f"than the number of landmarks ({len(location)}) "
                             f"!= ({n_landmarks})")
        else:
            sample_points = np.empty(n_landmarks + 2)
            sample_points[0] = fd.domain_range[0][0]
            sample_points[-1] = fd.domain_range[0][1]
            sample_points[1:-1] = location

        interpolator = SplineInterpolator(interpolation_order=3, monotone=True)

        warping = FDataGrid(data_matrix=data_matrix,
                            sample_points=sample_points,
                            interpolator=interpolator,
                            extrapolation='bounds')

        try:
            warping_points = fd.sample_points
        except AttributeError:
            warping_points = [np.linspace(*domain, 201)
                              for domain in fd.domain_range]

        self.warping_ = warping.to_grid(warping_points)

        return fd.compose(self.warping_)
