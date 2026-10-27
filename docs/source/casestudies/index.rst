Case Studies
============

End-to-end active learning campaigns that combine the building blocks of
AFL-agent into complete workflows on virtual instruments.

Bioformulations
---------------

Phase mapping of a protein / glycerol / temperature formulation on the
``ProteinGlycerolTemperaturePhaseMap`` virtual SAS instrument. Both notebooks
run a single campaign with the same Dirichlet GP phase classifier and the same
F1 benchmark against the simulator; they differ in how the next composition and
temperatures are chosen. The third notebook runs the manifold campaign through
AFL servers: the virtual instrument and the agent run as AFL-automation
APIServers that exchange measurements through Tiled.

.. toctree::
   :maxdepth: 1

   bioformulations/baseline_active_learning
   bioformulations/manifold_active_learning
   bioformulations/manifold_active_learning_server
