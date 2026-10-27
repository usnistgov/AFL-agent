from typing import Callable, Optional, Tuple, List, Union
import numpy as np
import torch
import xarray as xr
from tqdm.auto import tqdm
from typing_extensions import Self
try:
    import geoopt
except ImportError as e:
    # Keep this metric optional so AgentDriver can start without it.
    _APDIST_IMPORT_ERROR = e
    geoopt = None
else:
    _APDIST_IMPORT_ERROR = None

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.spec import resolve_spec, resolve_function
from AFL.double_agent.variables import bind_dataset, resolve_variables

class SRSFHilbertSphere(geoopt.Manifold):
    """
    Hilbert Sphere manifold for functional data using SRSF representation.
    
    The Square Root Slope Function (SRSF) transformation maps functions to a
    Hilbert sphere, enabling efficient elastic shape analysis of functional data.
    
    For a function f: [0,1] -> R^d, the SRSF is defined as:
        q(t) = sign(f'(t)) * sqrt(|f'(t)|)
    
    The manifold structure is that of the unit sphere in L^2([0,1], R^d).
    
    Parameters
    ----------
    n_sampling_points : int
        Number of discrete sampling points for the functions
    ambient_dim : int, optional
        Dimension of the codomain (default: 1 for scalar functions)
    
    Attributes
    ----------
    name : str
        Name of the manifold
    reversible : bool
        Whether the exponential map is reversible
    ndim : int
        Number of dimensions
    """
    
    name = "SRSF Hilbert Sphere"
    reversible = False
    ndim = 1
    
    def __init__(self, n_sampling_points: int, ambient_dim: int = 1):
        super().__init__()
        self.n_sampling_points = n_sampling_points
        self.ambient_dim = ambient_dim
        self.dt = 1.0 / (n_sampling_points - 1)  # Time step for [0,1] discretization
        
    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5
    ) -> Tuple[bool, Optional[str]]:
        """Check if point lies on the Hilbert sphere (has unit L2 norm)."""
        try:
            self._check_shape(x, "x")
        except ValueError as e:
            return False, str(e)
        
        # Compute L2 norm using trapezoidal rule
        norm_sq = self._l2_inner(x, x)
        norm = torch.sqrt(norm_sq)
        
        ok = torch.allclose(norm, torch.ones_like(norm), atol=atol, rtol=rtol)
        if not ok:
            reason = f"L2 norm is {norm.mean().item():.6f}, expected 1.0"
            return False, reason
        return True, None
    
    def _check_vector_on_tangent(
        self, 
        x: torch.Tensor, 
        u: torch.Tensor, 
        *, 
        atol: float = 1e-5, 
        rtol: float = 1e-5
    ) -> Tuple[bool, Optional[str]]:
        """Check if vector u is tangent to the sphere at point x."""
        self._check_shape(x, "x")
        self._check_shape(u, "u")
        
        # Tangent vectors must be orthogonal to the point: <x, u> = 0
        inner_prod = self._l2_inner(x, u)
        
        ok = torch.allclose(
            inner_prod, torch.zeros_like(inner_prod), atol=atol, rtol=rtol
        )
        if not ok:
            reason = f"Inner product <x,u> = {inner_prod.mean().item():.6f}, expected 0.0"
            return False, reason
        return True, None
    
    def _l2_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 inner product using trapezoidal rule integration.
        
        Returns scalar for each batch element.
        """
        # Pointwise product
        pointwise = (x * y).sum(dim=-1)  # Sum over ambient dimensions
        
        # Trapezoidal rule: integral ≈ dt * (0.5*f[0] + f[1] + ... + f[n-2] + 0.5*f[n-1])
        integral = torch.zeros(pointwise.shape[:-1], device=x.device, dtype=x.dtype)
        integral = integral + 0.5 * pointwise[..., 0]
        integral = integral + pointwise[..., 1:-1].sum(dim=-1)
        integral = integral + 0.5 * pointwise[..., -1]
        integral = integral * self.dt
        
        return integral
    
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point onto the manifold (normalize to unit L2 norm).
        
        Parameters
        ----------
        x : torch.Tensor
            Point to project, shape (..., n_sampling_points, ambient_dim)
            
        Returns
        -------
        torch.Tensor
            Projected point on the Hilbert sphere
        """
        norm_sq = self._l2_inner(x, x)
        norm = torch.sqrt(norm_sq)
        
        # Add dimensions for broadcasting
        while norm.ndim < x.ndim:
            norm = norm.unsqueeze(-1)
        
        return x / (norm + 1e-8)
    
    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Project vector u onto the tangent space at point x.
        
        The tangent space at x consists of all vectors orthogonal to x.
        Projection: u_tangent = u - <u,x> * x
        
        Parameters
        ----------
        x : torch.Tensor
            Point on manifold, shape (..., n_sampling_points, ambient_dim)
        u : torch.Tensor
            Vector to project, shape (..., n_sampling_points, ambient_dim)
            
        Returns
        -------
        torch.Tensor
            Projected tangent vector
        """
        inner = self._l2_inner(u, x)
        
        # Add dimensions for broadcasting
        while inner.ndim < x.ndim:
            inner = inner.unsqueeze(-1)
        
        return u - inner * x
    
    def inner(
        self, 
        x: torch.Tensor, 
        u: torch.Tensor, 
        v: Optional[torch.Tensor] = None, 
        *, 
        keepdim: bool = False
    ) -> torch.Tensor:
        """
        Compute inner product in the tangent space at x.
        
        On the Hilbert sphere, the Riemannian metric is inherited from
        the ambient L2 space.
        
        Parameters
        ----------
        x : torch.Tensor
            Point on manifold
        u : torch.Tensor
            First tangent vector
        v : torch.Tensor, optional
            Second tangent vector (if None, compute norm of u)
        keepdim : bool
            Keep reduced dimensions
            
        Returns
        -------
        torch.Tensor
            Inner product value
        """
        if v is None:
            v = u
        
        result = self._l2_inner(u, v)
        
        if keepdim:
            while result.ndim < x.ndim:
                result = result.unsqueeze(-1)
        
        return result
    
    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, norm_u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Exponential map: move along geodesic from x in direction u.
        
        On the sphere, geodesics are great circles:
            exp_x(u) = cos(||u||) * x + sin(||u||) / ||u|| * u
        
        Parameters
        ----------
        x : torch.Tensor
            Base point on manifold
        u : torch.Tensor
            Tangent vector at x
        norm_u : torch.Tensor, optional
            Pre-computed norm of u
            
        Returns
        -------
        torch.Tensor
            Point on manifold reached by exponential map
        """
        if norm_u is None:
            norm_u = torch.sqrt(self._l2_inner(u, u))
        
        # Add dimensions for broadcasting
        norm_u_expanded = norm_u
        while norm_u_expanded.ndim < x.ndim:
            norm_u_expanded = norm_u_expanded.unsqueeze(-1)
        
        # Handle zero tangent vector
        mask = (norm_u > 1e-8).float()
        while mask.ndim < x.ndim:
            mask = mask.unsqueeze(-1)
        
        cos_norm = torch.cos(norm_u_expanded)
        sin_norm = torch.sin(norm_u_expanded)
        
        # When norm is small, sin(norm)/norm ≈ 1
        sinc = torch.where(
            norm_u_expanded > 1e-8,
            sin_norm / norm_u_expanded,
            torch.ones_like(norm_u_expanded)
        )
        
        return self.projx(cos_norm * x + sinc * u)
    
    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Retraction: fast approximate exponential map.
        
        Uses projection-based retraction: retr_x(u) = proj(x + u)
        
        Parameters
        ----------
        x : torch.Tensor
            Base point on manifold
        u : torch.Tensor
            Tangent vector at x
            
        Returns
        -------
        torch.Tensor
            Point on manifold
        """
        return self.projx(x + u)
    
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map: compute tangent vector pointing from x to y.
        
        On the sphere:
            log_x(y) = (y - <x,y> * x) * arccos(<x,y>) / ||y - <x,y> * x||
        
        Parameters
        ----------
        x : torch.Tensor
            Base point on manifold
        y : torch.Tensor
            Target point on manifold
            
        Returns
        -------
        torch.Tensor
            Tangent vector at x pointing toward y
        """
        inner_xy = self._l2_inner(x, y)
        
        # Add dimensions for broadcasting
        inner_xy_expanded = inner_xy
        while inner_xy_expanded.ndim < x.ndim:
            inner_xy_expanded = inner_xy_expanded.unsqueeze(-1)
        
        # Project y onto tangent space at x
        u = y - inner_xy_expanded * x
        
        # Compute norm of projection
        norm_u = torch.sqrt(self._l2_inner(u, u))
        norm_u_expanded = norm_u
        while norm_u_expanded.ndim < x.ndim:
            norm_u_expanded = norm_u_expanded.unsqueeze(-1)
        
        # Compute angle
        angle = torch.acos(torch.clamp(inner_xy, -1.0 + 1e-7, 1.0 - 1e-7))
        angle_expanded = angle
        while angle_expanded.ndim < x.ndim:
            angle_expanded = angle_expanded.unsqueeze(-1)
        
        # Scale by angle/norm
        scale = torch.where(
            norm_u_expanded > 1e-8,
            angle_expanded / norm_u_expanded,
            torch.ones_like(norm_u_expanded)
        )
        
        return u * scale
    
    def transp(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport of vector v from tangent space at x to tangent space at y.
        
        Uses the projection-based parallel transport.
        
        Parameters
        ----------
        x : torch.Tensor
            Starting point
        y : torch.Tensor
            Ending point
        v : torch.Tensor
            Tangent vector at x
            
        Returns
        -------
        torch.Tensor
            Transported tangent vector at y
        """
        return self.proju(y, v)
    
    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Convert Euclidean gradient to Riemannian gradient.
        
        The Riemannian gradient is the projection of the Euclidean gradient
        onto the tangent space at x.
        
        Parameters
        ----------
        x : torch.Tensor
            Point on manifold
        u : torch.Tensor
            Euclidean gradient at x
            
        Returns
        -------
        torch.Tensor
            Riemannian gradient (tangent vector at x)
        """
        return self.proju(x, u)
        
    def origin(
        self, 
        *size, 
        dtype: Optional[torch.dtype] = None, 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate the origin point (constant function with appropriate norm).
        
        Parameters
        ----------
        size : tuple
            Desired shape (excluding the manifold dimensions)
        dtype : torch.dtype, optional
            Data type
        device : torch.device, optional
            Device
            
        Returns
        -------
        torch.Tensor
            Origin point on the manifold
        """
        if dtype is None:
            dtype = torch.get_default_dtype()
        
        shape = size + (self.n_sampling_points, self.ambient_dim)
        x = torch.ones(shape, dtype=dtype, device=device)
        return self.projx(x)
    
    def function_to_srsf(self, f: torch.Tensor) -> torch.Tensor:
        """
        Convert function to its SRSF representation.
        
        SRSF(f)(t) = f'(t) / sqrt(|f'(t)|)
        
        Parameters
        ----------
        f : torch.Tensor
            Function values, shape (..., n_sampling_points, ambient_dim)
            
        Returns
        -------
        torch.Tensor
            SRSF representation (on the Hilbert sphere)
        """
        # Compute derivative using central differences
        df = torch.zeros_like(f)
        df[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * self.dt)
        df[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / self.dt
        df[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / self.dt
        
        # Compute SRSF
        df_norm = torch.sqrt((df ** 2).sum(dim=-1, keepdim=True))
        srsf = df / torch.sqrt(df_norm + 1e-8)
        
        # Project onto sphere
        return self.projx(srsf)
    
    def srsf_to_function(
        self, q: torch.Tensor, f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert SRSF representation back to function (approximate).
        
        Integrates: f(t) = f(0) + integral(q(s) * |q(s)|, s=0..t)
        
        Parameters
        ----------
        q : torch.Tensor
            SRSF representation
        f0 : torch.Tensor, optional
            Initial value f(0), shape (..., ambient_dim)
            
        Returns
        -------
        torch.Tensor
            Reconstructed function
        """
        if f0 is None:
            f0 = torch.zeros(
                q.shape[:-2] + (self.ambient_dim,), 
                dtype=q.dtype, 
                device=q.device
            )
        # Now recover φ(t) = normalized ∫ ψ(s)^2 ds
        dphi = q**2                                         # (N,T,1)
        phi = torch.cumsum(dphi, dim=1) * self.dt                  # integrate
        phi = phi / phi[..., -1:, :]                          # enforce φ(1)=1
        phi[..., 0, :] = 0.0

        return phi
    
    def random(
        self,
        n_samples: int,
        n_basis: int = 10,
        device=None,
        dtype=None,
        ):
        """
            Generates random points on the Hilbert Sphere S^∞ by sampling from a 
            Gaussian Process in the tangent space at the identity.

            Each of the n_samples is generated using unique, randomly sampled 
            hyperparameters (σ, β) to ensure a diverse set of functional shapes.

            Mathematical Formulation
            ------------------------
            1. Base Point (p):
            The identity element in the SRSF space is the constant function p(t) = 1,
            which corresponds to the identity warping φ(t) = t.

            2. Tangent Space Basis (e_k):
            We construct an orthonormal basis for the tangent space T_p S^∞. 
            Tangent vectors must satisfy ∫ v(t)p(t) dt = 0. We use a cosine basis:
                e_k(t) = √2 * cos(k * π * t), for k = 1, ..., n_basis

            3. Stochastic Hyperparameters:
            For each sample i ∈ {1, ..., n_samples}, we sample:
                σ_i ~ LogNormal(μ=-1.0, σ=0.5) : Controls the warping amplitude.
                β_i ~ Uniform(1.0, 3.0)       : Controls the spectral decay (smoothness).

            4. Spectral Decay (λ_{i,k}):
            The variance for each basis component k for sample i is defined by the 
            power-law decay:
                λ_{i,k} = (σ_i^2) / (k^{2 * β_i})

            5. Tangent Vector Construction (v_i):
            We sample i.i.d. coefficients c_{i,k} ~ N(0, 1) and form the vector:
                v_i(t) = ∑_{k=1}^{n_basis} (c_{i,k} * √λ_{i,k}) * e_k(t)

            6. Exponential Map:
            The final point ψ_i on the manifold is reached by moving along the 
            geodesic from p in the direction of v_i:
                ψ_i = exp_p(v_i) = cos(||v_i||) * p + sin(||v_i||) * (v_i / ||v_i||)

            Parameters
            ----------
            n_samples : int
                Number of independent random SRSF functions to generate.
            n_basis : int, optional
                The number of cosine basis functions (K). Higher values allow for 
                higher-frequency variations. Default is 10.
            device : torch.device, optional
                The torch device to host the tensors.
            dtype : torch.dtype, optional
                The data type for computations.

            Returns
            -------
            torch.Tensor
                A tensor of shape (n_samples, n_sampling_points, ambient_dim) 
                representing points (SRSF functions) on the Hilbert Sphere.
            """
        if device is None: device = torch.device("cpu")
        if dtype is None: dtype = torch.get_default_dtype()

        T = self.n_sampling_points
        p = self.origin(n_samples, dtype=dtype, device=device)

        # 1. Sample unique sigma and beta for EACH sample
        # sigma ~ LogNormal; beta ~ Uniform
        sigmas = torch.exp(torch.randn(n_samples, 1, device=device, dtype=dtype) * 0.5 - 1.0)
        betas = torch.rand(n_samples, 1, device=device, dtype=dtype) * 2.0 + 0.1 

        # 2. Build basis
        t = torch.linspace(0., 1., T, device=device, dtype=dtype)
        k = torch.arange(1, n_basis + 1, device=device, dtype=dtype).view(1, -1) # (1, K)
        
        # Compute basis functions e_k: (K, T, 1)
        e = np.sqrt(2.) * torch.cos(np.pi * k.view(-1, 1) * t)
        e = e.unsqueeze(-1)

        # Enforce tangency and normalize basis (relative to the identity p)
        for i in range(n_basis):
            inner = self._l2_inner(p[0], e[i])
            e[i] = e[i] - inner * p[0]
            norm = torch.sqrt(self._l2_inner(e[i], e[i]))
            e[i] = e[i] / (norm + 1e-8)

        # 3. Vectorized Spectral Decay: lambda_{n, k} = sigma_n^2 / k^{2 * beta_n}
        # sigmas: (N, 1), k: (1, K), betas: (N, 1) -> lam: (N, K)
        lam = (sigmas**2) / (k ** (2.0 * betas))

        # 4. Sample coefficients using the per-sample variance
        xi = torch.randn(n_samples, n_basis, device=device, dtype=dtype)
        coeffs = xi * torch.sqrt(lam) # (N, K)

        # 5. Form tangent vectors and map to sphere
        v = torch.einsum("nk,ktd->ntd", coeffs, e)
        v = self.proju(p, v)
        psi = self.expmap(p, v)

        return psi

class SRSFHilbertSphereAcquisitionFunction(PipelineOp):
    """
    Optimise a function (e.g. a temperature profile) on the SRSF Hilbert sphere.

    Maximises a user supplied score over points ``q`` of the SRSF Hilbert sphere with
    Riemannian Adam from ``n_restarts`` random starts and returns the best function
    ``f* = srsf_to_function(q*)`` on ``[0, 1]``. What is scored is entirely up to the
    callable, so the op itself holds no acquisition logic: in the manifold active
    learning method the score is the mean entropy of a phase classifier along the
    profile at the composition picked upstream (typically by
    :class:`~AFL.double_agent.AcquisitionFunction.BoTorchAcquisition`).

    The score is ordinary Python: a lambda, closure or module-level function that
    closes over whatever objects and settings it needs, and uses
    :class:`~AFL.double_agent.variables.Variable` placeholders for values that only
    exist in the dataset at calculate time (the phase model, the composition picked
    upstream). The op binds the dataset while it runs so the placeholders resolve.

    Parameters
    ----------
    manifold : SRSFHilbertSphere | dict | str | type
        The manifold to optimise on: an instance, or a spec such as
        ``{"class": SRSFHilbertSphere, "args": {"n_sampling_points": 128}}``.
    acquisition : callable | str | dict
        Score to maximise, called as ``acquisition(q)`` where ``q`` is a
        ``torch.Tensor`` of shape ``(n_restarts, n_sampling_points, ambient_dim)`` on the
        sphere. It must return one value per restart (shape ``(n_restarts,)``; extra
        trailing dimensions are averaged), higher being better, and be differentiable
        w.r.t. ``q``. ``Variable("manifold")`` resolves to this op's manifold and
        ``Variable("dataset")`` to the input dataset; any other ``Variable`` is looked
        up in the dataset (a model ref is loaded into the model, numeric data becomes a
        tensor on ``device``) and recorded in the op's ``input_variable``.

        Lambdas and closures are kept in memory and exported as ``{"ref": ...}`` in the
        pipeline JSON (see :class:`~AFL.double_agent.PipelineOp.PipelineOp`); a
        module-level function is stored by import path and may be given as a string.
        ``{"function": f, "args": {...}}`` is accepted too, with ``args`` passed as
        keyword arguments after resolving any ``Variable`` values.
    bounds_t : list[float] | None
        ``[low, high]`` the function values are mapped onto, recorded in the output
        attrs for downstream use. Defaults to ``[0, 1]``. Close over it in the score if
        the score needs it.
    output_prefix : str
        Outputs are ``{prefix}_function`` and ``{prefix}_srsf`` (dims ``{prefix}_t``,
        ``{prefix}_dim``) and ``{prefix}_loss`` (dims ``{prefix}_restart``,
        ``{prefix}_iteration``).
    n_restarts, n_iterations, learning_rate
        Riemannian Adam settings.
    device : str

    Examples
    --------
    Mean classifier entropy along the profile at the composition chosen upstream::

        from AFL.double_agent.variables import Variable

        manifold = SRSFHilbertSphere(n_sampling_points=100)
        model = Variable("phase_model")            # model ref written by the extrapolator
        composition = Variable("composition_next")  # output of the stage-1 acquisition

        def profile_score(q, composition, model, manifold, bounds_t, power=1.0):
            fx = manifold.srsf_to_function(q)  # (n_restarts, n_t, 1) in [0, 1]
            ...
            return entropy.pow(power).mean(dim=1)

        op = SRSFHilbertSphereAcquisitionFunction(
            manifold=manifold,
            acquisition=lambda q: profile_score(q, composition, model, manifold, [0.5, 45.0], power=2.0),
        )
    """

    PROVIDED_KEYS = ("manifold", "dataset")  # Variable names the op itself provides

    def __init__(
        self,
        manifold: Union["SRSFHilbertSphere", dict, str, type],
        acquisition: Union[Callable, str, dict],
        bounds_t: Optional[List[float]] = None,
        output_prefix: str = "srsf",
        n_restarts: int = 16,
        n_iterations: int = 50,
        learning_rate: float = 0.1,
        device: str = "cpu",
        name: str = "SRSFHilbertSphereAcquisitionFunction",
    ):
        super().__init__(
            name=name,
            input_variable=None,
            output_variable=f"{output_prefix}_function",
            output_prefix=output_prefix,
        )
        self.manifold = manifold
        self.acquisition = acquisition
        self.bounds_t = list(bounds_t or [0.0, 1.0])
        self.n_restarts = n_restarts
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.device = device
        # Populated by calculate(): the score built from ``self.acquisition`` (which only
        # holds the JSON spec), kept for inspection.
        self.acquisition_function = None

    def resolve_manifold(self):
        """The manifold instance: ``self.manifold`` itself or built from its spec."""
        if geoopt is not None and isinstance(self.manifold, geoopt.Manifold):
            return self.manifold
        return resolve_spec(self.manifold)

    def build_acquisition(self, dataset: xr.Dataset, manifold):
        """Return ``score(q) -> (n_restarts,)`` built from ``self.acquisition``.

        Must run inside :func:`bind_dataset` (``calculate`` takes care of this) so that
        ``Variable`` values in a ``{"function", "args"}`` spec resolve.
        """
        spec = self.acquisition
        if isinstance(spec, dict):
            if "function" not in spec:
                raise ValueError(
                    f"{self.name}: acquisition spec must be a callable, an import path or "
                    f"{{'function': ..., 'args': {{...}}}}, got keys {sorted(spec)}"
                )
            target = spec["function"]
            function = target if callable(target) else resolve_function(target)
            kwargs = resolve_variables(dict(spec.get("args") or {}))
        else:
            function, kwargs = resolve_function(spec), {}
        if isinstance(function, type):
            raise TypeError(
                f"{self.name}: acquisition must be a score function; classes are not supported."
            )

        def score(q: torch.Tensor) -> torch.Tensor:
            value = torch.as_tensor(function(q, **kwargs))
            if value.ndim > 1:
                value = value.reshape(value.shape[0], -1).mean(dim=1)
            if value.numel() != q.shape[0]:
                raise ValueError(
                    f"{self.name}: acquisition must return one value per restart ({q.shape[0]}), "
                    f"got shape {tuple(value.shape)}"
                )
            return value.reshape(-1)

        return score

    def calculate(self, dataset: xr.Dataset) -> Self:
        manifold = self.resolve_manifold()
        with bind_dataset(dataset, {"manifold": manifold}):
            self.acquisition_function = self.build_acquisition(dataset, manifold)
            q_star, f_star, traj, loss_traj, ind_star = self.optimize(self.acquisition_function, manifold)

        p = self.output_prefix
        t = np.linspace(0, 1, manifold.n_sampling_points)
        f_star = f_star.reshape(manifold.n_sampling_points, -1)
        q_star = q_star.reshape(manifold.n_sampling_points, -1)
        # function values along the optimisation path of the winning restart: (n_steps, n_t, dim)
        traj_star = traj[ind_star]  # (n_iterations + 1, n_t * dim) SRSFs
        f_traj = manifold.srsf_to_function(traj_star.reshape(traj_star.shape[0], manifold.n_sampling_points, -1))
        self.output[f"{p}_trajectory"] = xr.DataArray(
            f_traj.detach().cpu().numpy().reshape(traj_star.shape[0], manifold.n_sampling_points, -1),
            dims=[f"{p}_step", f"{p}_t", f"{p}_dim"],
            coords={f"{p}_t": t},
            attrs={"description": "Function (values in [0, 1]) at every optimiser step of the best restart"},
        )
        self.output[f"{p}_restart_star"] = xr.DataArray(int(ind_star))
        self.output[f"{p}_function"] = xr.DataArray(
            f_star.detach().cpu().numpy(),
            dims=[f"{p}_t", f"{p}_dim"],
            coords={f"{p}_t": t},
            attrs={"bounds_t": self.bounds_t, "description": "Optimal function on [0, 1] (values in [0, 1])"},
        )
        self.output[f"{p}_srsf"] = xr.DataArray(
            q_star.detach().cpu().numpy(), dims=[f"{p}_t", f"{p}_dim"], coords={f"{p}_t": t}
        )
        self.output[f"{p}_loss"] = xr.DataArray(
            loss_traj.detach().cpu().numpy().reshape(self.n_restarts, -1),
            dims=[f"{p}_restart", f"{p}_iteration"],
        )
        return self

    def optimize(self, score, manifold):
        """Maximise ``score`` over the sphere with Riemannian Adam; return the best restart."""
        q_init = manifold.random(self.n_restarts).to(torch.get_default_dtype()).to(self.device)
        q = geoopt.ManifoldParameter(q_init, manifold=manifold)
        opt = geoopt.optim.RiemannianAdam([q], lr=self.learning_rate)
        traj = [q.clone().detach()]
        loss_traj = []
        for _ in tqdm(range(self.n_iterations), desc="Optimizing q", disable=True):
            opt.zero_grad()
            loss_per_restart = -score(q)
            loss_traj.append(loss_per_restart.detach())
            loss_per_restart.sum().backward()
            opt.step()
            traj.append(q.clone().detach())

        ind_star = loss_per_restart.argmin()
        q_star = q[ind_star, ...].clone().detach()
        fx_star = manifold.srsf_to_function(q_star.reshape(1, -1, 1))
        return q_star, fx_star, torch.stack(traj, dim=1), torch.stack(loss_traj, dim=1), ind_star
