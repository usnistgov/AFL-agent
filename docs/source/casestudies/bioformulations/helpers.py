"""Settings, acquisition scores and pipeline of the manifold active learning case studies.

Shared by the notebooks *Manifold Active Learning of a Protein Formulation Phase Map*
(``manifold_active_learning.ipynb``), which runs ``pipeline`` in-process, and *Closed-Loop
Manifold Active Learning with AFL Servers* (``manifold_active_learning_server.ipynb``), which
sends ``pipeline.to_dict()["ops"]`` to a running ``DoubleAgentDriver``. The pipeline holds
lambdas, the SRSF manifold and the simulator, which have no JSON form: ``to_dict`` records them
as ``{"ref": ...}`` entries, so the server notebook passes ``environment="helpers"`` and the
agent server imports this module to obtain the live objects behind the references
(``AFL.double_agent.Pipeline.load_environment``). Launch the agent server with this directory on
``PYTHONPATH`` so the import succeeds. Plotting lives in ``visuals.py``.

The pipeline is the one of the companion notebook *Manifold Active Learning of a Protein
Formulation Phase Map*:

1. ``DirichletGPExtrapolator`` phase classifier on (protein, temperature, glycerol)
2. ``PhaseMapF1Score`` benchmark of the classifier against the simulator
3. ``BoTorchAcquisition`` over the composition with ``composition_score``
4. ``SRSFHilbertSphereAcquisitionFunction`` over the temperature profile with ``profile_score``
"""

import tempfile

import numpy as np
import torch

from AFL.double_agent.AcquisitionFunction import BoTorchAcquisition
from AFL.double_agent.benchmarking import PhaseMapF1Score
from AFL.double_agent.Generator import CartesianGrid
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.PyTorchExtrapolator import DirichletGPExtrapolator
from AFL.double_agent.representations.functions import SRSFHilbertSphere, SRSFHilbertSphereAcquisitionFunction
from AFL.double_agent.variables import Variable
from AFL.double_agent.VirtualInstruments import ProteinGlycerolTemperaturePhaseMap

SEED = 1000  # seed of the benchmark test set

# design space
LIMITS = {"protein": (0.5, 75.0), "glycerol": (0.5, 10.0), "temperature": (0.5, 45.0)}
BOUNDS_T = list(LIMITS["temperature"])  # profile values in [0, 1] are mapped onto this range
GRID_STEPS = 20  # 3D grid on which the GP posterior (phase map) is evaluated
COMPOSITION_GRID_STEPS = 10  # 2D grid on which the composition score is reported

# phase classifier
GP_PARAMS = {"method": "mll", "n_iterations": 200, "verbose": False}
MODEL_DIR = tempfile.mkdtemp(prefix="afl_manifold_gp_")  # where the fitted GP is written each iteration

# acquisition
N_SAMPLING_POINTS = 100  # discretisation of a temperature profile on [0, 1]
OPTIMIZE_XY = {"num_restarts": 8, "raw_samples": 64}  # BoTorch optimiser of the composition score
OPTIMIZE_F = {"n_restarts": 8, "n_iterations": 100}  # Riemannian optimiser of the profile score
N_MC_PROFILES = 16  # random profiles averaged in the composition score
PROFILE_SCORE_POWER = 1.0  # exponent on the entropy in profile_score
PROFILE = {"dt": 0.05, "dT": 3.0}  # sampling of the optimal profile and the degC rounding of its temperatures

# benchmark
N_TEST_PER_PHASE = 100  # stratified test points per phase of the F1 score
F1_AVERAGE = "weighted"  # sklearn f1_score averaging: weighted, macro, micro or None (per phase)

# The benchmark scores the classifier against this instance; the measurements of the campaign
# come from the instrument *server*, which runs its own instance of the same class.
sim = ProteinGlycerolTemperaturePhaseMap()
COMPONENTS = sim.components  # feature order of the instrument: protein, temperature, glycerol
XY = [c for c in COMPONENTS if c != "temperature"]  # composition components: protein, glycerol
T_INDEX = COMPONENTS.index("temperature")

MANIFOLD = SRSFHilbertSphere(n_sampling_points=N_SAMPLING_POINTS, ambient_dim=1)  # shared by both stages

phase_model = Variable("phase_model")  # written by the extrapolator (model_uri=MODEL_DIR)
composition_next = Variable("composition_next")  # written by the stage-1 acquisition


def entropy_along_profiles(xy, q, model, manifold, bounds_t):
    """Classifier entropy along temperature profiles ``q`` at composition ``xy``; shape (n_q, n_t)."""
    fx = manifold.srsf_to_function(q)  # (n_q, n_t, 1) in [0, 1]
    t_values = fx * (bounds_t[1] - bounds_t[0]) + bounds_t[0]
    xy = xy.reshape(1, 1, 2).expand(t_values.shape[0], t_values.shape[1], -1)
    cols = [xy[..., :1], xy[..., 1:]]
    cols.insert(T_INDEX, t_values)
    X = torch.cat(cols, dim=-1).reshape(-1, 3)
    _, _, entropy, _ = model.predict(X)
    return entropy.reshape(t_values.shape[:2])


def composition_score(X, model, manifold, bounds_t, n_mc_samples=16, seed=0):
    """Expected entropy at composition ``X`` over random profiles (same profiles on every call)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        q = manifold.random(n_samples=n_mc_samples).to(X)
    return torch.stack(
        [entropy_along_profiles(xy, q, model, manifold, bounds_t).mean() for xy in X.reshape(-1, 2)]
    ).mean()


def profile_score(q, composition, model, manifold, bounds_t, power=1.0):
    """Mean entropy along each profile ``q`` at ``composition``; one value per profile."""
    entropy = entropy_along_profiles(composition, q, model, manifold, bounds_t)
    return (entropy**power).mean(dim=1)


def get_temperature_profile(f_star, bounds_t, dt=0.05, dT=3.0):
    """Discretise the optimal profile ``f*`` (values in [0, 1]) into the unique temperatures to measure."""
    f = np.asarray(f_star).reshape(-1)
    tt = np.linspace(0.0, 1.0, int(np.floor(1.0 / dt)))
    temps = np.interp(tt, np.linspace(0.0, 1.0, len(f)), f) * (bounds_t[1] - bounds_t[0]) + bounds_t[0]
    temps = np.round(temps / dT) * dT
    return np.unique(np.clip(temps, *bounds_t))


with Pipeline(name="manifoldAL") as pipeline:
    CartesianGrid(
        output_variable="design_space_grid",
        sample_dim="ds_grid",
        component_dim="ds_dim",
        grid_spec={c: {"min": LIMITS[c][0], "max": LIMITS[c][1], "steps": GRID_STEPS} for c in COMPONENTS},
        name="Grid-DesignSpace",
    )
    CartesianGrid(
        output_variable="composition_grid",
        sample_dim="comp_grid",
        component_dim="component",
        grid_spec={c: {"min": LIMITS[c][0], "max": LIMITS[c][1], "steps": COMPOSITION_GRID_STEPS} for c in XY},
        name="Grid-Composition",
    )
    # phase classifier on (protein, temperature, glycerol); feature scaling lives inside the model
    DirichletGPExtrapolator(
        feature_input_variable="design_space",
        predictor_input_variable="phases",
        output_prefix="phase",
        grid_variable="design_space_grid",
        grid_dim="ds_grid",
        sample_dim="sample",
        component_dim="ds_dim",
        params=GP_PARAMS,
        standardize=True,
        bounds={c: LIMITS[c] for c in COMPONENTS},
        model_uri=MODEL_DIR,  # writes the model ref to "phase_model"
        name="DirichletGPExtrapolator-PhaseLabels",
    )
    # benchmark: F1 score of the classifier against the simulator on a phase-stratified test set
    # (drawn once, equal points per phase); writes "benchmark_f1", "benchmark_accuracy", ...
    PhaseMapF1Score(
        ground_truth=sim,
        bounds={c: LIMITS[c] for c in COMPONENTS},
        model_variable="phase_model",
        average=F1_AVERAGE,
        n_per_phase=N_TEST_PER_PHASE,
        seed=SEED,
        output_prefix="benchmark",
        name="PhaseMapF1Score",
    )
    # stage 1: composition in R^2 maximising the expected entropy over random profiles
    BoTorchAcquisition(
        feature_input_variable="composition",
        grid_variable="composition_grid",
        grid_dim="comp_grid",
        bounds={c: LIMITS[c] for c in XY},
        acquisition=lambda X: composition_score(X, phase_model, MANIFOLD, BOUNDS_T, n_mc_samples=N_MC_PROFILES),
        output_prefix="composition",
        output_variable="composition_next",
        **OPTIMIZE_XY,
        name="BoTorchAcquisition-Composition",
    )
    # stage 2: temperature profile on the SRSF Hilbert sphere at that composition
    SRSFHilbertSphereAcquisitionFunction(
        manifold=MANIFOLD,
        acquisition=lambda q: profile_score(q, composition_next, phase_model, MANIFOLD, BOUNDS_T, power=PROFILE_SCORE_POWER),
        bounds_t=BOUNDS_T,
        output_prefix="profile",
        **OPTIMIZE_F,
        name="SRSF-TemperatureProfile",
    )


def xarray_to_target(results):
    """Turn a pipeline result into the composition and temperatures to measure next.

    Stage 1 writes ``composition_next`` (over ``component``), stage 2 the optimal profile
    ``profile_function`` with values in [0, 1]; the profile is mapped onto ``BOUNDS_T`` and
    discretised into the unique temperatures of the sweep.
    """
    return {
        **{c: float(results.composition_next.sel(component=c)) for c in XY},
        "temperatures": get_temperature_profile(results.profile_function.values, BOUNDS_T, **PROFILE),
    }
