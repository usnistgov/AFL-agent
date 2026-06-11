from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp

class DesignSpaceHierarchyCost(PipelineOp):
    """
    Compute a hierarchical cost over a grid of design points.

    This class assigns a cost to each point in the design space based on a
    hierarchical ordering of variables, where higher-cost variables dominate
    the contribution. The cost is iteratively updated across calls, allowing
    dynamic penalization based on exploration history.

    Note
    ----
    This follows the formulation in Equation (8) of:

        Abdolshah, M., Shilton, A., Rana, S., Gupta, S., & Venkatesh, S. (2019).
        Cost-aware multi-objective Bayesian optimisation.
        arXiv preprint arXiv:1909.03600.
        https://doi.org/10.48550/arXiv.1909.03600

    Parameters
    ----------
    input_variable : str, optional
        Name of the dataset variable representing query points.
    grid_variable : str, optional
        Name of the dataset variable representing the design grid.
    grid_dim : str, optional
        Dimension label of the grid variable in the dataset.
    component_dim : str, optional
        Dimension label identifying components (variables) in the dataset.
    coordinates_order : list of str, optional
        Ordered list of variable names, from highest to lowest cost contribution.
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
        self._banned_from_attrs.extend(["iteration"])

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute and store hierarchical cost over the full grid.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing query points and grid variables.

        Returns
        -------
        self : DesignSpaceHierarchyCost
            Returns self with updated output containing hierarchical costs.
        """
        query = dataset[self.input_variable]
        cost_grid = self.evaluate_cost(query=query)

        self.output[self.output_variable] = xr.DataArray(cost_grid, dims=self.grid_dim)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Cost per sample evaluated on {self.input_variable}"

        return self
    
    def evaluate_cost(self, query: xr.DataArray) -> np.ndarray:
        """
        Compute cost for each query point using previously sampled data.

        Parameters
        ----------
        query : xr.DataArray
            Data array containing query points for which cost is computed.

        Returns
        -------
        output : np.ndarray
            Array of hierarchical costs for each query point.
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
    Compute a probability-based cost for a binary classification outcome.
    This Pipeline provides the probability of a particular outcome as a scalar
    value that can be used as a cost. 
    For example, probabiity of a particular point in the design space being feasibile.

    Note:
    -----
    Since the convention for cost in this module is to score points with
    higher costs as unfavorable (in terms of utility) when used with `UtilityWithCost`, 
    we select `cost_label` that indicates the probability of undesired outcome (e.g.: unfeasible).

    Parameters
    ----------
    input_variable : str, optional
        Name of the dataset variable containing probability values.
    grid_variable : str, optional
        Name of the dataset variable representing the design grid.
    cost_label : int, default=1
        Index of the probability class to consider as cost.
    dim : str, default="grid"
        Dimension label for the output cost array.
    output_variable : str, optional
        Name of the variable in the dataset to store the cost.
    name : str, default="BinaryProbabilityCost"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Compute the binary probability cost and store it in the dataset.
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
        Compute the binary probability cost.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing probability values and grid variable.

        Returns
        -------
        self : BinaryProbabilityCost
            Returns self with updated output containing the probability-based cost.
        """
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
    Compute marginal cost by averaging across non-selected dimensions.

    Parameters
    ----------
    input_variable : str, default="cost"
        Name of the dataset variable containing cost values.
    coordinate_dims : list of str, default=['temperature']
        Dimensions to marginalize.
    component_dim : str, default="ds_dim"
        Dimension label representing components.
    grid_variable : str, default="design_space_grid"
        Dataset variable representing the design grid.
    output_variable : str, default="cost"
        Name of the variable to store the marginal cost.
    name : str, default="MarginalCost"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Compute the marginal cost by averaging over non-selected coordinate dimensions.
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
        """
        Apply cost-weighted acquisition function to the dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing acquisition and cost values.

        Returns
        -------
        self : UtilityWithCost
            Returns self with updated output containing the cost-weighted acquisition.
        """
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
    Compute a slice of the cost along a specified coordinate dimension.

    Parameters
    ----------
    input_variable : str, default="cost"
        Name of the dataset variable containing cost values.
    conditioning_point : str, default="next_composition"
        Reference point along other dimensions for slicing.
    coordinate_dim : str, default="temperature"
        Dimension along which the slice is computed.
    grid_variable : str, default="design_space_grid"
        Dataset variable representing the design grid.
    component_dim : str, default="ds_dim"
        Dimension label representing components.
    output_variable : str, default="temperature"
        Name of the variable to store the sliced cost.
    name : str, default="SlicedCost"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Extract a slice of cost along `coordinate_dim` at the conditioning point.
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
        """
        Extract a slice of the cost along `coordinate_dim` at the conditioning point.

        The method identifies the subset of grid points that match the given 
        `conditioning_point` along all other dimensions, and extracts the cost 
        values along the specified `coordinate_dim`. Both the sliced cost and its 
        domain are stored in the output.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the design space grid, cost values, and the conditioning point.

        Returns
        -------
        self : SlicedCost
            Returns self with updated `output` containing:
            
            - `output_variable`: Array of cost values along the `coordinate_dim` slice.
            - `<output_prefix>_domain`: Array of corresponding coordinate values along the slice.
        """
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
    Apply acquisition function with cost weighting to a design grid.

    The acquisition is scaled as:
    
        Q(x, t) * ∏ (1 - C_i(x, t))
    
    where Q is the acquisition function and C_i are cost terms.

    Parameters
    ----------
    input_variable : str, optional
        Dataset variable containing acquisition values.
    cost_variables : list of str, optional
        List of dataset variables containing normalized cost values.
    output_variable : str, optional
        Name of the variable in the dataset to store the modified acquisition.
    name : str, default="UtilityWithCost"
        Name of the pipeline operation.

    Attributes
    ----------
    cost_variables : list of str
        Names of dataset variables providing normalized costs.

    Methods
    -------
    calculate(dataset)
        Apply acquisition function with cost weighting and store the result.
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
        Apply cost-weighted acquisition function to the dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing acquisition and cost values.

        Returns
        -------
        self : UtilityWithCost
            Returns self with updated output containing the cost-weighted acquisition.
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
        self.output[self.output_variable].attrs["description"] = "Acquisition with Cost" 

        return self