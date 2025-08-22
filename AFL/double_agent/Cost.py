from typing import Optional, Union, List, Tuple, Dict
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp

class DesignSpaceHierarchyCost(PipelineOp):
    """
    Compute a hierarchical cost over a grid of design points.

    This class assigns a cost to each point in the design space based on a
    hierarchical ordering of variables, where higher-cost variables dominate
    the contribution. 
    
    The cost is iteratively updated across calls, allowing
    dynamic penalization based on exploration history.

    Parameters
    ----------
    grid_variable : str, optional
        Name of the variable in the dataset representing the design grid.
    grid_dim : str, optional
        Dimension label of the grid variable in the dataset.
    component_dim : str, optional
        Dimension label identifying components (variables) in the dataset.
    variables_order : list of str, optional
        Ordered list of variable names, from highest to lowest cost contribution.
    variables_offsets : list of float, optional
        Offset values associated with each variable in `variables_order`.
    output_variable : str, optional
        Name of the variable in the dataset where the computed cost will be stored.
    name : str, default="DesignSpaceHierarchyCost"
        Name of the pipeline operation.

    Attributes
    ----------
    variables_order : list of str
        Order of variables used to compute hierarchical costs.
    variables_offsets : list of float
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
        grid_variable: str = None,
        grid_dim:str =None,
        component_dim:str=None,
        variables_order : List[str] = None, # from highest to lowest cost
        variables_offsets : List[float] = None,
        output_variable: str = None,
        name: str = "DesignSpaceHierarchyCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[grid_variable], 
            output_variable=output_variable
        )
        self.variables_order = variables_order
        self.variables_offsets = variables_offsets
        self.grid_variable = grid_variable
        self.grid_dim = grid_dim
        self.component_dim = component_dim
        self.iteration = 1

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Compute and store cost over the full grid."""
        grid = dataset[self.grid_variable]

        cost_grid = self.evaluate_cost(query=grid)

        self.output[self.output_variable] = xr.DataArray(cost_grid, dims=self.grid_dim)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Cost per sample evaluated on {self.input_variable}"

        return self
    
    def evaluate_cost(self, query: xr.DataArray) -> np.ndarray:
        """Compute cost for each query point using previously sampled data."""
        num_queries = query.shape[0]
        k = len(self.variables_order)
        output = np.zeros(num_queries)
        for i in range(num_queries):
            w = np.sort(np.random.dirichlet(np.ones(k),size=1))[0]
            dim_cost = []
            for j in range(k):
                ell = 1/((w[j]*self.iteration)+1)
                x = query.sel({self.component_dim: self.variables_order[j]}).values[i]
                f_x_t = ell*np.exp(-(np.abs(x-self.variables_offsets[j])*ell))
                dim_cost.append(1-f_x_t)
            output[i] = np.prod(np.array(dim_cost))

        self.iteration += 1
        return output 
    
class AcquisitionWithCost(PipelineOp):
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
    using `DesignSpaceHierarchyCost` on [0,1] normalized design space coordinates.

    Parameters
    ----------
    input_variable : str, optional
        Name of the dataset variable containing the acquisition values.
    cost_variables : list of str, optional
        List of dataset variables containing normalized cost values.
    grid_dim : str, default="grid"
        Dimension label for the acquisition grid.
    output_variable : str, optional
        Name of the variable in the dataset where the modified acquisition will be stored.
    name : str, default="AcquisitonWithCost"
        Name of the pipeline operation.

    Attributes
    ----------
    cost_variables : list of str
        Names of dataset variables providing normalized costs.
    grid_dim : str
        Dimension label for the acquisition grid.

    Methods
    -------
    calculate(dataset)
        Apply acquisition with cost weighting and store the result.
    """

    def __init__(
        self,
        input_variable: str = None,
        cost_variables: List[str] = None,
        grid_dim: str = "grid",
        output_variable: str = None,
        name: str = "AcquisitonWithCost",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable] + cost_variables, 
            output_variable=output_variable
        )
        self.cost_variables = cost_variables
        self.grid_dim = grid_dim

    def calculate(self, dataset: xr.Dataset) -> Self:

        acqv = dataset[self.input_variable]

        # Computes acqusition to be Q(x, t)*(1-C(x,t)) following 
        # https://arxiv.org/pdf/1909.03600 Equation 9 
        acqv_cost = acqv.values.copy()
        for var in self.cost_variables:
            cost = dataset[var].values
            acqv_cost *= (1.0-cost)

        # Store result
        output = xr.DataArray(acqv_cost, dims=self.grid_dim)
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = "Acquisition with Cost"

        return self