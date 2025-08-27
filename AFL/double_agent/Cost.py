from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp

class DesignSpaceHierarchyCost(PipelineOp):
    """
    Compute a hierarchical cost over a grid of design points.

    This class assigns a cost to each point in the design space based on a
    hierarchical ordering of coordinates, where higher-priority coordinates
    dominate the contribution. The cost is iteratively updated across calls,
    allowing dynamic penalization based on exploration history.

    Parameters
    ----------
    input_variable : str, optional
        Name of the input variable from which query points are drawn.
    grid_variable : str, optional
        Name of the variable in the dataset representing the design grid.
    grid_dim : str, optional
        Dimension label of the grid variable in the dataset.
    component_dim : str, optional
        Dimension label identifying components (coordinates) in the dataset.
    coordinates_order : list of str, optional
        Ordered list of coordinate names, from highest to lowest cost contribution.
    coordinates_offsets : list of float, optional
        Offset values associated with each coordinate in `coordinates_order`.
        Defaults to zero offsets if not provided.
    output_variable : str, optional
        Name of the variable in the dataset where the computed cost will be stored.
    name : str, default="DesignSpaceHierarchyCost"
        Name of the pipeline operation.

    Attributes
    ----------
    iteration : int
        Counter for the number of times cost evaluation has been performed.

    Methods
    -------
    calculate(dataset)
        Compute and store normalized cost over the full grid.
    evaluate_cost(query)
        Compute cost for each query point using the current iteration and offsets.
    """
    def __init__(
        self,
        input_variable: str = None,
        grid_variable: str = None,
        grid_dim:str =None,
        component_dim:str=None,
        coordinates_order : List[str] = None, # from highest to lowest cost
        coordinates_offsets : List[float] = None,
        output_variable: str = None,
        name: str = "DesignSpaceHierarchyCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.coordinates_order = coordinates_order 
        if coordinates_offsets is not None:
            assert len(self.coordinates_order)==len(coordinates_offsets) , f"Each variable should have an offset." + \
            f"Expected {len(self.coordinates_order)} but got {len(coordinates_offsets)}"
            self.coordinates_offsets = coordinates_offsets
        else:
            self.coordinates_offsets = np.zeros(len(self.coordinates_order))
        self.grid_variable = grid_variable
        self.grid_dim = grid_dim
        self.component_dim = component_dim
        self.iteration = 1

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute and store cost over the full grid.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing input query points and grid variables.

        Returns
        -------
        DesignSpaceHierarchyCost
            Updated instance with computed cost stored in `self.output`.
        """
        query = dataset[self.input_variable]
        grid = dataset[self.grid_variable]
        cost_grid = self.evaluate_cost(query=query)

        self.output[self.output_variable] = xr.DataArray(cost_grid, dims=self.grid_dim)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Cost per sample evaluated on {self.input_variable}"
        self.output[self.output_variable].attrs["domain"] = grid.values.squeeze()

        return self
    
    def evaluate_cost(self, query: xr.DataArray) -> np.ndarray:
        """
        Compute cost for each query point using previously sampled data.

        Parameters
        ----------
        query : xr.DataArray
            Query points with coordinate labels corresponding to `coordinates_order`.

        Returns
        -------
        numpy.ndarray
            Array of computed costs for each query point.
        """
        num_queries = query.shape[0]
        k = len(self.coordinates_order)
        output = np.zeros(num_queries)
        for i in range(num_queries):
            w = np.sort(np.random.dirichlet(np.ones(k),size=1))[0]
            dim_cost = []
            for j in range(k):
                ell = 1/((w[j]*self.iteration)+1)
                x = query.sel({self.component_dim: self.coordinates_order[j]}).values[i]
                f_x_t = ell*np.exp(-(np.abs(x-self.coordinates_offsets[j])*ell))
                dim_cost.append(1-f_x_t)
            output[i] = np.prod(np.array(dim_cost))

        self.iteration += 1
        return output 

class BinaryProbabilityCost(PipelineOp):
    """
    Compute a cost based on class probability from a predictive model.

    The cost is derived from the probability assigned to a specific class
    label, allowing cost-aware optimization in binary settings.

    Parameters
    ----------
    input_variable : str, optional
        Name of the dataset variable containing probability values.
    grid_variable : str, optional
        Name of the variable representing the design grid.
    cost_label : int, default=1
        Index of the class label used for cost computation.
    dim : str, default="grid"
        Dimension label of the probability array.
    output_variable : str, optional
        Name of the dataset variable where the cost will be stored.
    name : str, default="BinaryProbabilityCost"
        Name of the pipeline operation.

    Attributes
    ----------
    grid_variable : str
        Name of the dataset variable representing the grid.
    cost_label : int
        Index of the class label used for cost computation.
    dim : str
        Dimension label of the output cost variable.

    Methods
    -------
    calculate(dataset)
        Compute probability-based cost and store it in the output variable.
    """
    def __init__(
        self,
        input_variable: str = None,
        grid_variable: str = None,
        cost_label:int = 1,
        dim: str = "grid",
        output_variable: str = None,
        name: str = "BinaryProbabilityCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.dim = dim 
        self.grid_variable = grid_variable
        self.cost_label = cost_label

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute probability-based cost.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing probability values and grid variable.

        Returns
        -------
        BinaryProbabilityCost
            Updated instance with computed cost stored in `self.output`.
        """
        prob = dataset[self.input_variable]
        grid = dataset[self.grid_variable]
        cost = prob.values[:, self.cost_label] 
        # Store result
        output = xr.DataArray(cost, dims=self.dim)
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = "A probability based cost."
        self.output[self.output_variable].attrs["domain"] = grid.values.squeeze()

        return self

class MarginalCost(PipelineOp):
    """
    Compute marginal cost by averaging over selected coordinate dimensions.

    This class computes the marginal cost by integrating
    out specified coordinate dimensions of the design space.

    Parameters
    ----------
    input_variable : str, default="cost"
        Name of the dataset variable containing cost values.
    coordinate_dims : list of str, default=['temperature']
        List of coordinate dimensions to marginalize.
    component_dim : str, default="ds_dim"
        Dimension label for components in the grid.
    grid_variable : str, default="design_space_grid"
        Name of the dataset variable representing the design grid.
    output_dim : str, default="marginalized_dim"
        Dimension label for the marginalized output.
    output_variable : str, default="marginalized_utility"
        Name of the dataset variable where marginalized cost will be stored.
    name : str, default="MarginalCost"
        Name of the pipeline operation.

    Attributes
    ----------
    coordinate_dims : list of str
        List of coordinate dimensions marginalized over.
    grid_variable : str
        Dataset variable representing the design grid.
    dim : str
        Component dimension label.
    output_dim : str
        Dimension label for the marginalized output.

    Methods
    -------
    calculate(dataset)
        Compute marginal cost by averaging over specified coordinate dimensions.
    """
    def __init__(
        self,
        input_variable: str = "cost",
        coordinate_dims :  List[str]= ['temperature'],
        component_dim: str = "ds_dim",
        grid_variable:str = "design_space_grid",
        output_dim:str = "marginalized_dim",        
        output_variable: str = "marginalized_utility",
        name: str = "MarginalCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.coordinate_dims = coordinate_dims
        self.grid_variable = grid_variable 
        self.dim = component_dim 
        self.output_dim = output_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute marginal cost by averaging over selected coordinate dimensions.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing grid and cost variables.

        Returns
        -------
        MarginalCost
            Updated instance with marginalized cost stored in `self.output`.
        """
        grid = dataset[self.grid_variable]
        cost = dataset[self.input_variable] 
        
        grid_nonmarginal = grid.drop_sel({self.dim: self.coordinate_dims})
        unique_nonmarginal = grid_nonmarginal.to_pandas().drop_duplicates().reset_index(drop=True)
        x = unique_nonmarginal.values 
        num_comps = len(unique_nonmarginal)
        marginal_cost = np.zeros(num_comps)

        for i, xi in enumerate(x):
            # Find grid points for every unique non-marginal point
            squared_distances = np.sum((xi - grid_nonmarginal.values) ** 2, axis=1)
            idx = np.argwhere(squared_distances<1e-5) # indices to find the function value at xi across the coordinate_dims
            cost_xi = cost.values[idx].squeeze() # cost across the coordinate_dims
            marginal_cost[i] = cost_xi.mean() # marginal over other dims 

        output = xr.DataArray(marginal_cost, dims=self.output_dim)
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = "A probability based cost."
        self.output[self.output_variable].attrs["domain"] = x 
        
        return self

class SlicedCost(PipelineOp):
    """
    Compute sliced cost at a conditioning point along a chosen coordinate dimension.

    This class extracts the cost along a single coordinate dimension,
    conditioned on fixing all other complement dimensions to a specific
    point.

    Parameters
    ----------
    input_variable : str, default="cost"
        Name of the dataset variable containing cost values.
    conditioning_point : str, default="next_composition"
        Name of the dataset variable specifying the conditioning point.
    coordinate_dim : str, default="temperature"
        Coordinate dimension along which slicing is performed.
    grid_variable : str, default="design_space_grid"
        Name of the dataset variable representing the design grid.
    component_dim : str, default="ds_dim"
        Dimension label for components in the grid.
    output_dim : str, default="sliced_dim"
        Dimension label for the sliced output.
    output_variable : str, default="sliced_cost"
        Name of the dataset variable where sliced cost will be stored.
    name : str, default="SlicedCost"
        Name of the pipeline operation.

    Attributes
    ----------
    complement_dims : list of str
        Dimensions fixed at the conditioning point.
    slice_dim : str
        Dimension along which slicing is performed.
    conditioning_point : str
        Dataset variable specifying the conditioning point.
    grid_variable : str
        Dataset variable representing the design grid.
    dim : str
        Component dimension label.
    output_dim : str
        Dimension label for the sliced output.

    Methods
    -------
    calculate(dataset)
        Extract cost along a sliced dimension conditioned on other coordinates.
    """
    def __init__(
        self,
        input_variable: str = "cost",
        conditioning_point : str = "next_composition",
        coordinate_dim:str = "temperature",
        grid_variable:str = "design_space_grid",        
        component_dim: str = "ds_dim",
        output_dim : str = "sliced_dim",
        output_variable: str = "sliced_cost",
        name: str = "SlicedCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.slice_dim = coordinate_dim
        self.conditioning_point = conditioning_point       
        self.grid_variable = grid_variable 
        self.dim = component_dim 
        self.output_dim = output_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute sliced cost at the conditioning point.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing grid, cost, and conditioning point variables.

        Returns
        -------
        SlicedCost
            Updated instance with sliced cost stored in `self.output`.
        """
        cp = dataset[self.conditioning_point]

        grid = dataset[self.grid_variable]
        all_coordinate_dims = grid[self.dim].values.tolist()
        complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dim]
        grid_complement_dims = grid.sel({self.dim:complement_dims}).values
        grid_slice_dim = grid.sel({self.dim:self.slice_dim}).values

        squared_distances = np.sum((cp.values - grid_complement_dims) ** 2, axis=1)
        idx = np.argwhere(squared_distances<1e-5)

        cost = dataset[self.input_variable]
        # Extract cost at cp : c_{cp}(slice_dim)
        sliced_cost = cost.values[idx]

        output = xr.DataArray(sliced_cost.squeeze(), dims=self.output_dim)
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Utility calculated along the temperature axis" # type: ignore
        self.output[self.output_variable].attrs["domain"] = grid_slice_dim[idx].squeeze() 

        return self

class UtilityWithCost(PipelineOp):
    """
    Apply utility with cost weighting to a design grid.

    This class modifies an utility function by incorporating
    cost penalties, following the formulation in Equation (9) of
    "Cost-Aware Bayesian Optimization" (https://arxiv.org/pdf/1909.03600).
    The utility is scaled as::

        Q(x, t) * ∏ (1 - C_i(x, t))

    where Q is the utility function and C_i are cost terms.

    Note
    ----
    This works best when the cost functions are bound to [0,1],
    as is the case when using `DesignSpaceHierarchyCost` on [0,1]
    normalized design space coordinates.

    Parameters
    ----------
    input_variable : str, optional
        Name of the dataset variable containing the utility values.
    cost_variables : list of str, optional
        List of dataset variables containing normalized cost values.
    output_variable : str, optional
        Name of the variable in the dataset where the modified utility will be stored.
    name : str, default="UtilityWithCost"
        Name of the pipeline operation.

    Attributes
    ----------
    cost_variables : list of str
        Names of dataset variables providing normalized costs.

    Methods
    -------
    calculate(dataset)
        Apply utility with cost weighting and store the result.
    """
    def __init__(
        self,
        input_variable: str = None,
        cost_variables: List[str] = None,
        output_variable: str = None,
        name: str = "UtilityWithCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable] + cost_variables, 
            output_variable=output_variable
        )
        self.cost_variables = cost_variables

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Apply utility function with cost weighting.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing utility and cost variables.

        Returns
        -------
        UtilityWithCost
            Updated instance with cost-adjusted utility stored in `self.output`.
        """
        acqv = dataset[self.input_variable]

        # Computes acqusition to be Q(x, t)*(1-C(x,t)) following 
        # https://arxiv.org/pdf/1909.03600 Equation 9 
        acqv_cost = acqv.values.copy()
        for var in self.cost_variables:
            cost = dataset[var].values
            acqv_cost *= (1.0-cost)

        # Store result
        output = xr.DataArray(acqv_cost, dims=acqv.dims, coords=acqv.coords)
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = "utility with Cost" 
        self.output[self.output_variable].attrs["domain"] = acqv.domain

        
        return self