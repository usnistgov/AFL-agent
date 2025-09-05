from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp

class MarginalEntropyOverDimension(PipelineOp):
    """
    Compute marginal entropy by averaging probability over selected dimensions.

    This class calculates the marginal entropy of a probability distribution 
    by marginalizing over specified `coordinate_dims`. The resulting entropy 
    is used as a utility measure.

    Parameters
    ----------
    input_variable : str, default="probability"
        Name of the dataset variable containing probability values.
    coordinate_dims : list of str, default=['temperature']
        Dimensions over which the probabilities are marginalized.
    component_dim : str, default="ds_dim"
        Dimension label representing components in the design grid.
    grid_variable : str, default="design_space_grid"
        Dataset variable representing the design grid.
    output_variable : str, default="composition_utility"
        Name of the variable in the dataset to store the entropy-based utility.
    name : str, default="MarginalEntropyOverDimension"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Compute marginal entropy over specified dimensions and store the result.
    """
    def __init__(
        self,
        input_variable: str = "probability",
        coordinate_dims :  List[str]= ['temperature'],
        component_dim: str = "ds_dim",
        grid_variable:str = "design_space_grid",
        output_variable: str = "composition_utility",
        name: str = "MarginalEntropyOverDimension",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable,
        )
        self.coordinate_dims = coordinate_dims
        self.grid_variable = grid_variable 
        self.component_dim = component_dim 


    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute marginal entropy over specified dimensions.

        For each unique combination of non-marginal coordinates, this method
        averages the probabilities over the `coordinate_dims` and calculates
        the entropy.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the design grid and probability values.

        Returns
        -------
        self : MarginalEntropyOverDimension
            Returns self with `output` containing:
            
            - `output_variable`: Array of marginal entropy values per unique non-marginal point.
            - `<output_prefix>_domain`: Array of corresponding domain points for non-marginal dimensions.
        """
        grid = dataset[self.grid_variable]
        all_coordinate_dims = grid[self.component_dim].values.tolist()
        complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dims]
        self.output_prefix = "_".join(i for i in complement_dims)

        grid_nonmarginal = grid.drop_sel({self.component_dim: self.coordinate_dims})
        unique_nonmarginal = grid_nonmarginal.to_pandas().drop_duplicates().reset_index(drop=True)

        x = unique_nonmarginal.values
        num_comps = len(unique_nonmarginal)
        ux = np.zeros(num_comps)

        Pr = dataset[self.input_variable].values 
        for i, xi in enumerate(x):
            # Find grid points for every unique non-marginal point
            squared_distances = np.sum((xi - grid_nonmarginal.values) ** 2, axis=1)
            idx = np.argwhere(squared_distances<1e-5) # indices for different coordinate_dims of x 
            Pr_xi = Pr[idx,:].squeeze() # probability at xi over different coordinate_dims
            Pr_marginal = Pr_xi.mean(axis=0) # marginalization over the coordinate_dims
            ux_i =  -np.sum(np.log(Pr_marginal) * Pr_marginal, axis=-1) # Marginal entropy as the utility
            ux[i] = ux_i.item()
  
        output = xr.DataArray(ux.squeeze(), dims=self._prefix_output("n"))
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Entropy calculated by marginalizing probability over certain dimension(s)" # type: ignore 

        domain_variable = self._prefix_output("domain")
        if not domain_variable in dataset:
            domain = xr.DataArray(x.reshape(-1, len(complement_dims)), 
                                dims=(self._prefix_output("n"), self._prefix_output("d")),
                                coords={self._prefix_output("d"): complement_dims}
                                )
            self.output[domain_variable] = domain
            self.output[domain_variable].attrs["description"] = f"Domain of the {self.output_variable} computed." # type: ignore

        return self    
    
class MarginalEntropyAlongDimension(PipelineOp):
    """
    Compute marginal entropy along a specified coordinate dimension at a conditioning point.

    This class calculates the entropy of probability values along a single
    coordinate dimension, conditioned on a reference point in all other dimensions.

    Parameters
    ----------
    input_variable : str, default="probability"
        Dataset variable containing probability values.
    conditioning_point : str, default="next_composition"
        Reference point along other dimensions for slicing.
    coordinate_dim : str, default="temperature"
        Dimension along which the entropy is computed.
    grid_variable : str, default="design_space_grid"
        Dataset variable representing the design grid.
    component_dim : str, default="ds_dim"
        Dimension label representing components in the grid.
    output_variable : str, default="temperature"
        Name of the variable to store the marginal entropy along the coordinate.
    name : str, default="MarginalEntropyAlongDimension"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Compute marginal entropy along `coordinate_dim` at the conditioning point.
    """
    def __init__(
        self,
        input_variable: str = "probability",
        conditioning_point : str = "next_composition",
        coordinate_dim:str = "temperature",
        grid_variable:str = "design_space_grid",        
        component_dim: str = "ds_dim",
        output_variable:str="temperature",
        name: str = "MarginalEntropyAlongDimension",
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
        self.component_dim = component_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute marginal entropy along a single dimension at a conditioning point.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the design grid, probability values, and conditioning point.

        Returns
        -------
        self : MarginalEntropyAlongDimension
            Returns self with `output` containing:
            
            - `output_variable`: Array of entropy values along the specified coordinate dimension.
            - `<output_prefix>_domain`: Array of corresponding domain points along the coordinate dimension.
        """
        cp = dataset[self.conditioning_point]

        grid = dataset[self.grid_variable]
        all_coordinate_dims = grid[self.component_dim].values.tolist()
        complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dim]
        grid_complement_dims = grid.sel({self.component_dim:complement_dims}).values

        squared_distances = np.sum((cp.values - grid_complement_dims) ** 2, axis=1)
        idx = np.argwhere(squared_distances<1e-5)
        T = grid.sel({self.component_dim:self.coordinate_dim}).values[idx]

        Pr = dataset[self.input_variable]

        # Extract utility at cp : u_{cp}(entropy_dim)
        Pr_cp = Pr.values[idx,:].copy() # type: ignore 
        vT =  -np.sum(np.log(Pr_cp) * Pr_cp, axis=-1) # Marginal entropy as the utility

        output = xr.DataArray(vT.squeeze(), dims = self._prefix_output("n"))
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Utility calculated along the temperature axis" # type: ignore 

        domain_variable = self._prefix_output("domain")

        if not domain_variable in dataset:
            domain = xr.DataArray(T.squeeze(), dims=self._prefix_output("n"))
            self.output[domain_variable] = domain
            self.output[domain_variable].attrs["description"] = f"Domain of the {self.output_variable} computed." # type: ignore

        return self