# Bioformulations Case Study

Active learning of a protein / glycerol / temperature phase map on the virtual instrument
`ProteinGlycerolTemperaturePhaseMap`.

| File | What it is |
|---|---|
| `baseline_active_learning.ipynb` | Grid-based Bayesian active learning campaign. Classifier plus grid-search acquisition, run entirely in-process. The reference the manifold notebooks compare against. |
| `manifold_active_learning.ipynb` | Same classifier and benchmark, but with the *manifold* strategy: continuous BoTorch acquisition over the composition and an SRSF Hilbert-sphere acquisition over the temperature profile. Runs the pipeline from `helpers.py` in-process. |
| `manifold_active_learning_server.ipynb` | Closed-loop version of the manifold campaign using AFL microservices: Tiled for data, a `VirtualInstrumentDriver` server for the instrument and a `DoubleAgentDriver` server for the agent. Shows how to launch the servers, register the data in Tiled and turn agent suggestions into measurements. |
| `helpers.py` | Settings (design-space limits, seeds, classifier and acquisition options), the simulator, the acquisition scores and the `pipeline` shared by the two manifold notebooks. The agent server imports it (`environment="helpers"`) to resolve the non-JSON objects in the pipeline, so start the agent server with this directory on `PYTHONPATH`. |
| `visuals.py` | Matplotlib plotting functions used by all notebooks: phase-map grids, sampled points, model entropy, acquisition surfaces, temperature history and score history. |
| `phase_map_instrument.json` | Config file for `python -m AFL.double_agent.VirtualInstrumentDriver --config phase_map_instrument.json`. Names the virtual instrument to serve and its `instrument_overrides` (noise, random seed). |
