from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp

class MarginalEntropyOverDimension(PipelineOp):
    """
    Compute marginal entropy by averaging probabilities over specified coordinate dimensions.

    This class marginalizes the probability distribution over given coordinate
    dimensions and computes the Shannon entropy at each remaining point in the
    design space. The entropy serves as a measure of uncertainty.

    Parameters
    ----------
    input_variable : str, default="probability"
        Name of the dataset variable containing probability distributions.
    coordinate_dims : list of str, default=['temperature']
        List of coordinate dimensions to marginalize out.
    component_dim : str, default="ds_dim"
        Dimension label for components in the design grid.
    grid_variable : str, default="design_space_grid"
        Name of the dataset variable representing the design grid.
    output_dim : str, default="n_comp"
        Dimension label for the marginalized entropy output.
    output_variable : str, default="composition_utility"
        Name of the dataset variable where marginalized entropy will be stored.
    name : str, default="MarginalEntropyOverDimension"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Compute marginal entropy by averaging over specified coordinate dimensions.
    """
    def __init__(
        self,
        input_variable: str = "probability",
        coordinate_dims :  List[str]= ['temperature'],
        component_dim: str = "ds_dim",
        grid_variable:str = "design_space_grid", 
        output_dim :str = "n_comp",       
        output_variable: str = "composition_utility",
        name: str = "MarginalEntropyOverDimension",
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
        Compute marginal entropy by averaging probabilities over given dimensions.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing probability distributions and grid variables.

        Returns
        -------
        MarginalEntropyOverDimension
            Updated instance with marginalized entropy stored in `self.output`.
        """
        grid = dataset[self.grid_variable]
        grid_nonmarginal = grid.drop_sel({self.dim: self.coordinate_dims})
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
  
        output = xr.DataArray(ux.squeeze(), dims=self.output_dim)
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Entropy calculated by marginalizing probability over certain dimension(s)" # type: ignore 
        self.output[self.output_variable].attrs["domain"] = x.squeeze()

        return self    
    
class MarginalEntropyAlongDimension(PipelineOp):
    """
    Compute marginal entropy along a single coordinate dimension, conditioned on a point.

    This class computes entropy of probability distributions along a chosen
    coordinate dimension (e.g., temperature), while fixing the values of all
    complementary dimensions at a specified conditioning point.

    Parameters
    ----------
    input_variable : str, default="probability"
        Name of the dataset variable containing probability distributions.
    conditioning_point : str, default="next_composition"
        Name of the dataset variable specifying the conditioning point.
    coordinate_dim : str, default="temperature"
        Coordinate dimension along which entropy is computed.
    grid_variable : str, default="design_space_grid"
        Name of the dataset variable representing the design grid.
    component_dim : str, default="ds_dim"
        Dimension label for components in the design grid.
    output_dim : str, default="n_temp"
        Dimension label for the entropy output along the chosen dimension.
    output_variable : str, default="marginal_entropy_along_dim"
        Name of the dataset variable where computed entropy will be stored.
    name : str, default="MarginalEntropyAlongDimension"
        Name of the pipeline operation.

    Attributes
    ----------
    coordinate_dim : str
        Coordinate dimension along which entropy is computed.
    conditioning_point : str
        Dataset variable specifying the conditioning point.
    grid_variable : str
        Dataset variable representing the design grid.
    dim : str
        Component dimension label.
    output_dim : str
        Dimension label for the entropy output.

    Methods
    -------
    calculate(dataset)
        Compute marginal entropy along the chosen coordinate dimension
        at the specified conditioning point.
    """
    def __init__(
        self,
        input_variable: str = "probability",
        conditioning_point : str = "next_composition",
        coordinate_dim:str = "temperature",
        grid_variable:str = "design_space_grid",        
        component_dim: str = "ds_dim",
        output_dim :str = "n_temp", 
        output_variable: str = "marginal_entropy_along_dim",
        name: str = "MarginalEntropyAlongDimension",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.coordinate_dim = coordinate_dim
        self.conditioning_point = conditioning_point       
        self.grid_variable = grid_variable 
        self.dim = component_dim
        self.output_dim = output_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute marginal entropy along a coordinate dimension at a conditioning point.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing probability distributions, grid, and conditioning point.

        Returns
        -------
        MarginalEntropyAlongDimension
            Updated instance with entropy along the chosen dimension stored in `self.output`.
        """
        cp = dataset[self.conditioning_point]

        grid = dataset[self.grid_variable]
        all_coordinate_dims = grid[self.dim].values.tolist()
        complement_dims = [d for d in all_coordinate_dims if d not in self.coordinate_dim]
        grid_complement_dims = grid.sel({self.dim:complement_dims}).values

        squared_distances = np.sum((cp.values - grid_complement_dims) ** 2, axis=1)
        idx = np.argwhere(squared_distances<1e-5)
        T = grid.sel({self.dim:self.coordinate_dim}).values[idx]

        Pr = dataset[self.input_variable]

        # Extract utility at cp : u_{cp}(entropy_dim)
        Pr_cp = Pr.values[idx,:].copy() # type: ignore 
        vT =  -np.sum(np.log(Pr_cp) * Pr_cp, axis=-1) # Marginal entropy as the utility

        output = xr.DataArray(vT.squeeze(), dims = self.output_dim)
        self.output[self.output_variable] = output # type: ignore
        self.output[self.output_variable].attrs["description"] = "Utility calculated along the temperature axis" # type: ignore 
        self.output[self.output_variable].attrs["domain"] = T.squeeze()

        return self