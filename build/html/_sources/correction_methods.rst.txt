.. _correction_methods:

Methods For Correcting Non-Physical Regions
===========================================

Type 0 NPRs
-----------

As a matter of convention, Type 0 NPRs are never corrected in CG because the represent
a fundamental failure in the user's initialization parameters and there is no clear physical foundation for even
providing a method to fix them.

Type 1 NPRs
-----------

.. tab-set::

    .. tab-item:: Type 1a NPRs

        .. note::

            Recall that :py:class:`correction.Type1aNPR` are caused by inconsistent slopes of the profiles.

        To correct these non-physical regions, a positive interpolation scheme is used. The gravitational field :math:`\nabla \Phi`
        is computed using the original profiles and the holes are found using the Hole-Finding Algorithm (See :py:mod:`numalgs`). These
        holes correspond to the regions in which the underlying profiles are inconsistent. To fix the holes, the :math:`\nabla \Phi` profile is
        patched with :math:`C^1[a,b]` splines in such a way as to force the profile to be strictly positive. The ``correction_parameter`` kwarg can be
        passed to the correction method to determine the degree of severity in the correction. If :math:`\gamma_c = 1`, then the corresponding interpolation
        scheme will default to the monotone interpolation scheme, if :math:`\gamma_c = 0`, then the profile will be allowed to dip significantly toward zero within
        the region of the hole but never go negative.

        .. figure:: _images/numerical_algorithms/test_correction_NRP1a.h5_corrected.png

            A naive generation of the cluster A133 using the work of [ViKrF06]_ to construct best fit profiles (red) and its corrected
            version (blue). The underlying cause of this NPR is the steep slope of the temperature profile, which leads to the hole in
            the gravitational field.

        .. warning::

            There are instances in which a particularly high correction parameter will make it intractable to actually
            find a viable domain of interpolation.

Type 2 NPRs
-----------

.. tab-set::

    .. tab-item:: Type 2a NPRs

        .. note::

            Recall that :py:class:`correction.Type2aNPR` are caused by non-physical asymptotic temperature behaviors in common profiles.

        Type 2a NPRs are one of the more common classes of NPR that arise in the process of generating initial conditions. Because they are
        often tied to underlying profiles, they are generally fixable; however, the methodology has some complexities. Consider a temperature
        profile :math:`T(r)` and a complementary density profile :math:`\rho_g(r)`. Under the assumption of HSE,

        .. math::

            \nabla \Phi = \frac{-k_bT}{m_p\eta} \left[\frac{d\ln(\rho)}{dr} + \frac{d\ln(T)}{dr}\right].

        Furthermore, if we want to require that :math:`M_{dyn}` be stable at large radii, we require that :math:`\nabla \Phi \sim 1/r^2`.
        As such, at large radii, one can treat :math:`\rho(r) \sim r^\alpha` and :math:`T(r) \sim r^\beta`. Thus,

        .. math::

            \mathrm{M}_{\mathrm{dyn}}(<r) \sim (\alpha + \beta) \frac{-rk_b r^\beta}{Gm_p\eta} \implies \beta = -1.

        In many profiles, this is not the asymptotic behavior of the profile and therefore the dynamical mass becomes unstable.
        In most instances, an increasing dynamical mass profile doesn't pose much issue for simulation; however, when the dynamical mass
        decreases, it leads to significant non-physicality. As such, to repair these NPRs, the general approach is to replace (at a suitable radius)
        the temperature profile with a power law. To accomplish this, we use the following exponential extrapolation method [Diggins, 2023].

        .. admonition:: Extrapolatory Power Law Replacement

            Let :math:`f(x) \in C^n[a,b]` for some :math:`n > 1`. Let :math:`(x_0,f(x_0) = y_0)` be a point (the interpolatory point) on :math:`f(x)` and
            let :math:`(x_1,f(x_1))` be another point (called the vertex) such that :math:`x_0 < x_1`. Then there exist :math:`\gamma` and :math:`\omega` such that
            the piecewise function

            .. math::

                F(x) = \begin{cases}f(x),&x\le x_0\\\xi_\alpha(x)E_{\pm}(x),&x>x_0\end{cases},

            where

            .. math::

                \xi_\alpha(x) = y_1\left(\frac{x}{x_1}\right)^\alpha,\;\text{and}\; E_{\pm}(x;x_0) = \frac{\left(\frac{x}{x_0}\right)^{\omega}}{\left(\frac{x}{x_0}\right)^{\omega} \mp \gamma}

            is :math:`C^1[a,b]`.

        Using this approach, Type 2a NPRs are corrected by identifying the erroneous domain and using the extrapolatory power law method
        to rebuild the temperature and density profiles correctly. Because this method only guarantees that the system be :math:`C^1[a,b]`, it may produce mild jumps in
        the derived dynamical density, and halo density.
