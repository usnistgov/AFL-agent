from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp

class DesignSpaceHierarchyCost(PipelineOp):
    """
    Compute a hierarchical cost over a grid of design points.

    Parameters
    ----------
    input_variable : str, optional
        Name of the input variable for cost evaluation.
    grid_variable : str, optional
        Name of the variable in the dataset representing the design grid.
    grid_dim : str, optional
        Dimension label of the grid variable in the dataset.
    component_dim : str, optional
        Dimension label identifying components in the dataset.
    coordinates_order : list of str, optional
        Ordered list of variable names from highest to lowest cost contribution.
    coordinates_offsets : list of float, optional
        Offset values associated with each variable in `coordinates_order`.
    output_variable : str, optional
        Name of the variable in the dataset where the computed cost will be stored.
    name : str, default="DesignSpaceHierarchyCost"
        Name of the pipeline operation.

    Attributes
    ----------
    coordinates_order : list of str
        Order of variables used to compute hierarchical costs.
    coordinates_offsets : list of float
        Offset values for each variable.
    grid_variable : str
        Dataset variable name representing the grid.
    grid_dim : str
        Dimension label of the grid variable.
    component_dim : str
        Dimension label representing components.
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
        """Compute and store cost over the full grid."""
        query = dataset[self.input_variable]
        grid = dataset[self.grid_variable]
        cost_grid = self.evaluate_cost(query=query)

        self.output[self.output_variable] = xr.DataArray(cost_grid, dims=self.grid_dim)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Cost per sample evaluated on {self.input_variable}"

        return self
    
    def evaluate_cost(self, query: xr.DataArray) -> np.ndarray:
        """Compute cost for each query point using previously sampled data."""
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
    Compute a probability-based cost for a binary outcome.

    Parameters
    ----------
    input_variable : str, optional
        Name of the dataset variable containing probabilities.
    grid_variable : str, optional
        Name of the variable in the dataset representing the design grid.
    cost_label : int, default=1
        Index of the probability corresponding to the cost label.
    dim : str, default="grid"
        Dimension label for the cost output.
    output_variable : str, optional
        Name of the variable in the dataset where the cost will be stored.
    name : str, default="BinaryProbabilityCost"
        Name of the pipeline operation.

    Attributes
    ----------
    dim : str
        Dimension label for the cost output.
    grid_variable : str
        Dataset variable name representing the grid.
    cost_label : int
        Index for the probability cost label.

    Methods
    -------
    calculate(dataset)
        Compute and store probability-based cost over the full grid.
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

        prob = dataset[self.input_variable]
        grid = dataset[self.grid_variable]
        cost = prob.values[:, self.cost_label] 
        # Store result
        output = xr.DataArray(cost, dims=self.dim)
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = "A probability based cost."

        return self

class MarginalCost(PipelineOp):
    """
    MarginalCost class

    Compute the marginal cost by averaging over specified coordinate dimensions.

    Parameters
    ----------
    input_variable : str, default="cost"
        Name of the dataset variable containing the cost values.
    coordinate_dims : list of str, default=['temperature']
        Dimensions along which the marginalization is not performed.
    component_dim : str, default="ds_dim"
        Dimension label identifying components in the dataset.
    grid_variable : str, default="design_space_grid"
        Name of the variable in the dataset representing the design grid.
    output_variable : str, default="cost"
        Name of the variable in the dataset where the marginal cost will be stored.
    name : str, default="MarginalCost"
        Name of the pipeline operation.

    Attributes
    ----------
    coordinate_dims : list of str
        Dimensions to retain while marginalizing over other dimensions.
    grid_variable : str
        Dataset variable name representing the design grid.
    dim : str
        Dimension label representing components.

    Methods
    -------
    calculate(dataset)
        Compute marginal cost and store the resulting values and domain.
    """

    def __init__(
        self,
        input_variable: str = "cost",
        coordinate_dims :  List[str]= ['temperature'],
        component_dim: str = "ds_dim",
        grid_variable:str = "design_space_grid",
        output_variable:str="cost",
        name: str = "MarginalCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable,
        )
        self.coordinate_dims = coordinate_dims
        self.grid_variable = grid_variable 
        self.dim = component_dim 

    def calculate(self, dataset: xr.Dataset) -> Self:
        grid = dataset[self.grid_variable]
        cost = dataset[self.input_variable] 

        all_coordinate_dims = grid[self.component_dim].values.tolist()
        complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dims]
        self.output_prefix = "_".join(i for i in complement_dims)

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

        output = xr.DataArray(marginal_cost, dims=self._prefix_output("n"))
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = "A probability based cost."


        domain_variable = self._prefix_output("domain")
        if not domain_variable in dataset:
            all_coordinate_dims = grid[self.dim].values.tolist()
            complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dims]
            domain = xr.DataArray(x.reshape(-1, len(complement_dims)), 
                                dims=(self._prefix_output("n"), self._prefix_output("d")),
                                coords={self._prefix_output("d"): complement_dims}
                                )
            self.output[domain_variable] = domain
            self.output[domain_variable].attrs["description"] = f"Domain of the {self.output_variable} computed." # type: ignore
        return self

class SlicedCost(PipelineOp):
    """
    Compute a slice of cost along a specified coordinate dimension conditioned on a given point.

    Parameters
    ----------
    input_variable : str, default="cost"
        Name of the dataset variable containing the cost values.
    conditioning_point : str, default="next_composition"
        Dataset variable representing the point used for conditioning.
    coordinate_dim : str, default="temperature"
        Coordinate dimension along which the slice is performed.
    grid_variable : str, default="design_space_grid"
        Name of the variable in the dataset representing the design grid.
    component_dim : str, default="ds_dim"
        Dimension label identifying components in the dataset.
    output_variable : str, default="temperature"
        Name of the variable in the dataset where the sliced cost will be stored.
    name : str, default="SlicedCost"
        Name of the pipeline operation.

    Attributes
    ----------
    coordinate_dim : str
        Dimension along which the slicing is performed.
    conditioning_point : str
        Dataset variable representing the conditioning point.
    grid_variable : str
        Dataset variable name representing the design grid.
    dim : str
        Dimension label representing components.

    Methods
    -------
    calculate(dataset)
        Compute cost slice at the conditioning point and store the values and domain.
    """

    def __init__(
        self,
        input_variable: str = "cost",
        conditioning_point : str = "next_composition",
        coordinate_dim:str = "temperature",
        grid_variable:str = "design_space_grid",        
        component_dim: str = "ds_dim",
        output_variable:str="temperature",
        name: str = "SlicedCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable,
            output_prefix = f"{coordinate_dim}"
        )
        self.coordinate_dim = coordinate_dim
        self.conditioning_point = conditioning_point       
        self.grid_variable = grid_variable 
        self.dim = component_dim 

    def calculate(self, dataset: xr.Dataset) -> Self:
        cp = dataset[self.conditioning_point]

        grid = dataset[self.grid_variable]
        all_coordinate_dims = grid[self.dim].values.tolist()
        complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dim]
        grid_complement_dims = grid.sel({self.dim:complement_dims}).values
        grid_slice_dim = grid.sel({self.dim:self.coordinate_dim}).values

        squared_distances = np.sum((cp.values - grid_complement_dims) ** 2, axis=1)
        idx = np.argwhere(squared_distances<1e-5)

        cost = dataset[self.input_variable]
        # Extract cost at cp : c_{cp}(slice_dim)
        sliced_cost = cost.values[idx]

        output = xr.DataArray(sliced_cost.squeeze(), dims=self._prefix_output("n"))
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Utility calculated along the temperature axis" # type: ignore

        domain_variable = self._prefix_output("domain")
        if not domain_variable in dataset:
            domain = xr.DataArray(grid_slice_dim[idx].squeeze(), dims=self._prefix_output("n"))
            self.output[domain_variable] = domain
            self.output[domain_variable].attrs["description"] = f"Domain of the {self.output_variable} computed." # type: ignore
        return self

class UtilityWithCost(PipelineOp):
    """
    Apply acquisition with cost weighting to a design grid.

    This class modifies an acquisition function by incorporating
    cost penalties, following the formulation in Equation (9) of
    "Cost-Aware Bayesian Optimization" (https://arxiv.org/pdf/1909.03600).
    The acquisition is scaled as::

        Q(x, t) * ∏ (1 - C_i(x, t))

    where Q is the acquisition function and C_i are cost terms.
    
    Note: 
    This works best when the cost functions are bound to [0,1] as is the case when
    using `DesignSpaceHierarchyCost` on [0,1] normalized design space.

    Parameters
    ----------
    input_variable : str, optional
        Name of the dataset variable containing the acquisition values.
    cost_variables : list of str, optional
        List of dataset variables containing normalized cost values.
    output_variable : str, optional
        Name of the variable in the dataset where the modified acquisition will be stored.
    name : str, default="AcquisitonWithCost"
        Name of the pipeline operation.

    Attributes
    ----------
    cost_variables : list of str
        Names of dataset variables providing normalized costs.

    Methods
    -------
    calculate(dataset)
        Apply acquisition with cost weighting and store the result.
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
        self.output[self.output_variable].attrs["description"] = "Acquisition with Cost" 

        return self