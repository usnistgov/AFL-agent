Using Example Datasets
===================

AFL-agent comes with example datasets that you can use to learn and experiment with the library. These datasets are accessible through the ``AFL.double_agent.datasets`` module.

Loading Example Datasets
-----------------------

You can load the example datasets using the following code:

.. code-block:: python

    from AFL.double_agent.datasets import example_dataset1
    
    # Load the example dataset
    ds = example_dataset1()
    
    # Print information about the dataset
    print(f"Dataset dimensions: {dict(ds.sizes)}")
    print(f"Dataset variables: {list(ds.data_vars)}")
    print(f"Dataset coordinates: {list(ds.coords)}")

Available Datasets
-----------------

Currently, the following datasets are available:

- ``example_dataset1``: A synthetic dataset with compositions, measurements, and ground truth labels.

Listing Available Datasets
-------------------------

You can list all available datasets using the ``list_datasets`` function:

.. code-block:: python

    from AFL.double_agent.datasets import list_datasets
    
    # List all available datasets
    print(list_datasets())

Loading a Dataset by Name
------------------------

You can also load a dataset by name using the ``load_dataset`` function:

.. code-block:: python

    from AFL.double_agent.datasets import load_dataset
    
    # Load a dataset by name
    ds = load_dataset("example_dataset")

Dataset Location
---------------

The example datasets are stored in the ``AFL/double_agent/data`` directory within the package. The datasets module automatically locates and loads these files when you import and use the dataset functions.

Example: Using the Example Dataset with a Pipeline
-------------------------------------------------

Here's an example of how to use the example dataset with a pipeline:

.. code-block:: python

    from AFL.double_agent import Pipeline, SavgolFilter, Similarity, SpectralClustering
    from AFL.double_agent.datasets import example_dataset1
    
    # Load the example dataset
    ds = example_dataset1()
    
    # Create a pipeline
    with Pipeline() as clustering_pipeline:
        SavgolFilter(
            input_variable='measurement', 
            output_variable='derivative', 
            dim='x', 
            derivative=1
        )
        
        Similarity(
            input_variable='derivative', 
            output_variable='similarity', 
            sample_dim='sample',
        )
        
        SpectralClustering(
            input_variable='similarity',
            output_variable='labels',
            n_clusters=2
        )
    
    # Run the pipeline
    result = clustering_pipeline.calculate(ds)
    
    # Compare the predicted labels with the ground truth
    print("Predicted labels:", result.labels.values)
    print("Ground truth labels:", ds.ground_truth_labels.values) 