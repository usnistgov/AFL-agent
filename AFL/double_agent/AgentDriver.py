import pathlib
import uuid
import json
import inspect
import importlib
import pkgutil
from typing import Optional, Dict, Any, List

import xarray as xr

from AFL.automation.APIServer.Driver import Driver  # type: ignore
from AFL.automation.shared.utilities import mpl_plot_to_bytes,xarray_to_bytes
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.PipelineOp import PipelineOp


_PIPELINE_BUILDER_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset='UTF-8'>
  <title>Pipeline Builder</title>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/jsPlumb/2.15.6/js/jsplumb.min.js'></script>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; }
    #sidebar { width: 260px; height: 100vh; overflow-y: auto; border-right: 1px solid #ccc; float:left; padding:10px; box-sizing:border-box; }
    #canvas { position:relative; margin-left:260px; height:100vh; background:#f7f7f7; }
    .op-template { border:1px solid #ccc; padding:5px; margin-bottom:5px; cursor:grab; }
    .node { position:absolute; padding:10px; background:#fff; border:1px solid #333; }
    .param { display:block; margin-bottom:4px; }
    .connector { width:10px; height:10px; background:#000; border-radius:50%; display:inline-block; }
  </style>
</head>
<body>
  <div id='sidebar'>
    <h3>Pipeline Ops</h3>
    <div id='op-list'></div>
    <h3>Prefabs</h3>
    <select id='prefab-select'></select>
    <button id='load-prefab'>Load Prefab</button>
    <button id='submit'>Submit Pipeline</button>
  </div>
  <div id='canvas'></div>
  <script>
    const instance = jsPlumb.getInstance({Container: 'canvas'});
    const opList = document.getElementById('op-list');
    const canvas = document.getElementById('canvas');
    const prefabSelect = document.getElementById('prefab-select');
    let counter = 0;

    async function loadOps() {
      const res = await fetch('/pipeline_ops');
      const ops = await res.json();
      ops.forEach(op => {
        const div = document.createElement('div');
        div.className = 'op-template';
        div.textContent = op.name;
        div.draggable = true;
        div.dataset.fqcn = op.fqcn;
        div.dataset.params = JSON.stringify(op.parameters);
        div.addEventListener('dragstart', e => {
          e.dataTransfer.setData('text/plain', div.dataset.fqcn);
        });
        opList.appendChild(div);
      });
    }

    async function loadPrefabs() {
      const res = await fetch('/prefab_names');
      if (!res.ok) return;
      const names = await res.json();
      names.forEach(n => {
        const opt = document.createElement('option');
        opt.value = n;
        opt.textContent = n;
        prefabSelect.appendChild(opt);
      });
    }

    function clearCanvas() {
      instance.deleteEveryConnection();
      instance.deleteEveryEndpoint();
      canvas.innerHTML = '';
      counter = 0;
    }

    function loadPipeline(ops) {
      clearCanvas();
      let x = 20;
      let y = 20;
      const nodes = [];
      ops.forEach(op => {
        const node = addNode(op.class, op.args || {}, x, y);
        if (op.args) {
          node.querySelectorAll('input[data-param]').forEach(inp => {
            if (op.args[inp.dataset.param] !== undefined) {
              inp.value = op.args[inp.dataset.param];
            }
          });
          if (op.args.output_variable) {
            node.querySelector('input[data-output]').value = op.args.output_variable;
          }
        }
        nodes.push({node: node, op: op});
        y += 120;
      });

      nodes.forEach((n, idx) => {
        if (idx === 0) return;
        let source = null;
        if (n.op.args && n.op.args.input_variable) {
          const inVar = n.op.args.input_variable;
          source = nodes.find(m => m.op.args && m.op.args.output_variable === inVar);
        }
        if (!source && idx > 0) source = nodes[idx-1];
        if (source) {
          instance.connect({
            source: source.node.querySelector('[data-role="out"]'),
            target: n.node.querySelector('[data-role="in"]')
          });
        }
      });
    }

    async function loadCurrentPipeline() {
      const res = await fetch('/current_pipeline');
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length) {
          loadPipeline(data);
        }
      }
    }

    canvas.addEventListener('dragover', e => e.preventDefault());
    canvas.addEventListener('drop', e => {
      e.preventDefault();
      const fqcn = e.dataTransfer.getData('text/plain');
      const params = JSON.parse(document.querySelector(`[data-fqcn="${fqcn}"]`).dataset.params);
      addNode(fqcn, params, e.offsetX, e.offsetY);
    });

    function addNode(fqcn, params, x, y) {
      const node = document.createElement('div');
      node.className = 'node';
      node.id = 'node' + (counter++);
      node.style.left = x + 'px';
      node.style.top = y + 'px';
      node.dataset.fqcn = fqcn;

      const title = document.createElement('div');
      title.textContent = fqcn.split('.').pop();
      node.appendChild(title);

      Object.entries(params).forEach(([key, val]) => {
        const div = document.createElement('div');
        div.className = 'param';
        const label = document.createElement('label');
        label.textContent = key + ':';
        const input = document.createElement('input');
        input.type = 'text';
        input.dataset.param = key;
        if (val !== null) input.value = val;
        div.appendChild(label);
        div.appendChild(input);
        node.appendChild(div);
      });

      const outDiv = document.createElement('div');
      outDiv.className = 'param';
      outDiv.innerHTML = 'output_variable:<input data-output type="text">';
      node.appendChild(outDiv);

      const outAnchor = document.createElement('div');
      outAnchor.className = 'connector';
      outAnchor.dataset.role = 'out';
      outAnchor.style.position = 'absolute';
      outAnchor.style.right = '-5px';
      outAnchor.style.top = '50%';
      node.appendChild(outAnchor);

      const inAnchor = document.createElement('div');
      inAnchor.className = 'connector';
      inAnchor.dataset.role = 'in';
      inAnchor.style.position = 'absolute';
      inAnchor.style.left = '-5px';
      inAnchor.style.top = '50%';
      node.appendChild(inAnchor);

      canvas.appendChild(node);
      instance.draggable(node);
      instance.makeSource(outAnchor, {anchor:'Continuous', filter:outAnchor});
      instance.makeTarget(inAnchor, {anchor:'Continuous', allowLoopback:false});
      return node;
    }

    function buildOps() {
      const nodes = Array.from(document.querySelectorAll('.node'));
      const ops = [];
      nodes.forEach(node => {
        const args = {};
        node.querySelectorAll('input[data-param]').forEach(inp => {
          if (inp.value) args[inp.dataset.param] = inp.value;
        });
        const out = node.querySelector('input[data-output]').value;
        if (out) args['output_variable'] = out;
        const conns = instance.getConnections({target: node});
        if (conns.length) {
          const src = conns[0].source.parentElement;
          const srcOut = src.querySelector('input[data-output]').value;
          if (srcOut) args['input_variable'] = srcOut;
        }
        ops.push({class: node.dataset.fqcn, args: args});
      });
      return ops;
    }

    document.getElementById('submit').onclick = async () => {
      const ops = buildOps();
      const res = await fetch('/enqueue', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({task_name: 'initialize_pipeline', pipeline: ops})
      });
      if (res.ok) alert('Pipeline submitted');
      else alert('Submission failed');
    };

    document.getElementById('load-prefab').onclick = async () => {
      const name = prefabSelect.value;
      if (!name) return;
      const res = await fetch(`/load_prefab?name=${encodeURIComponent(name)}`);
      if (res.ok) {
        const data = await res.json();
        loadPipeline(data);
      }
    };

    loadOps();
    loadPrefabs();
    loadCurrentPipeline();
  </script>
</body>
</html>
"""


def _collect_pipeline_ops() -> List[Dict[str, Any]]:
    """Gather metadata for all available :class:`PipelineOp` subclasses."""
    ops: List[Dict[str, Any]] = []
    package = importlib.import_module("AFL.double_agent")
    for modinfo in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{modinfo.name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, PipelineOp) and obj is not PipelineOp:
                sig = inspect.signature(obj.__init__)
                params = {
                    k: (v.default if v.default is not inspect._empty else None)
                    for k, v in sig.parameters.items()
                    if k != "self"
                }
                ops.append(
                    {
                        "name": name,
                        "module": module.__name__,
                        "fqcn": f"{module.__name__}.{name}",
                        "parameters": params,
                    }
                )
    ops.sort(key=lambda o: o["name"])
    return ops


def get_pipeline_ops() -> List[Dict[str, Any]]:
    """Return metadata describing available pipeline operations."""
    return _collect_pipeline_ops()


def build_pipeline_from_ops(ops: List[Dict[str, Any]], name: str = "Pipeline") -> Dict[str, Any]:
    """Create a pipeline from a list of operation JSON dictionaries."""
    pipeline_ops = [PipelineOp.from_json(op) for op in ops]
    pipeline = Pipeline(name=name, ops=pipeline_ops)
    return {"pipeline": [op.to_json() for op in pipeline]}


def build_pipeline_from_json(ops_json: str, name: str = "Pipeline") -> Dict[str, Any]:
    """Helper that accepts a JSON string of operations."""
    try:
        ops = json.loads(ops_json)
    except json.JSONDecodeError:
        ops = []
    return build_pipeline_from_ops(ops, name)


def get_pipeline_builder_html() -> str:
    """Return the HTML for the pipeline builder UI."""
    return _PIPELINE_BUILDER_HTML


class DoubleAgentDriver(Driver):
    """
    Persistent Config
    -----------------
    save_path: str
        path to directory where data will be serialized to
    """

    defaults = {}
    defaults["save_path"] = "/home/AFL/"

    def __init__(
        self,
        name: str = "DoubleAgentDriver",
        overrides: Optional[Dict[str, Any]] = None,
    ):
        Driver.__init__(
            self, name=name, defaults=self.gather_defaults(), overrides=overrides
        )
        self.app = None
        self.name = name

        self.input: Optional[xr.Dataset] = None
        self.pipeline: Optional[Pipeline] = None
        self.results: Dict[str, xr.Dataset] = dict()

    def status(self):
        status = []
        if self.input:
            status.append(f'Input Dims: {self.input.sizes}')
        if self.pipeline:
            status.append(f'Pipeline loaded with {len(self.pipeline.ops)} operations')
        return status
        

    def initialize_input(self, db_uuid: str) -> None:
        """
        Set the initial input data to be evaluated in the `double_agent.Pipeline`

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve `xarray.Dataset` from. The Dataset should be deposited using `Client.deposit_obj`
            in interactive mode in order to obtain the uuid of the deposited item. See example below

        Example
        -------
        ```python
        from AFL.automation.APIServer.Client import Client
        import xarray as xr

        client = Client('localhost',port=5053)
        client.login('User')
        db_uuid = client.deposit_obj(obj=xr.Dataset(),interactive=True)['return_val']
        client.enqueue(task_name='initialize_input',db_uuid=db_uuid)
        ```
        """
        self.input = self.retrieve_obj(db_uuid)

    def initialize_pipeline(self, db_uuid: str) -> None:
        """
        Set the `double_agent.Pipeline` to outline

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve the `double_agent.Pipeline` from. The Dataset should be deposited using
            `Client.deposit_obj` in interactive mode in order to obtain the uuid of the deposited item. See example
            below

        Example
        -------
        ```python
        from AFL.automation.APIServer.Client import Client
        from AFL.double_agent import *

        client = Client('localhost',port=5053)
        client.login('User')
        db_uuid = client.deposit_obj(obj=Pipeline(),interactive=True)['return_val']
        client.enqueue(task_name='initialize_pipeline',db_uuid=db_uuid)
        ```
        """
        self.pipeline = self.retrieve_obj(db_uuid)

    def append(self, db_uuid: str, concat_dim: str) -> None:
        """

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve `xarray.Dataset` from. The Dataset should be deposited using `Client.deposit_obj`
            in interactive mode in order to obtain the uuid of the deposited item. See example below

        concat_dim: str
            `xarray` dimension in input dataset to concatenate to

        """
        if self.input is None:
            raise ValueError(
                'Must set "input" Dataset client.deposit_obj and then DoubleAgentDriver.initialize'
            )

        next_sample = self.retrieve_obj(db_uuid)

        if self.input is None:
            self.input = next_sample
        else:
            self.input = xr.concat(
                [self.input, next_sample], dim=concat_dim, data_vars="minimal"
            )

    @Driver.unqueued(render_hint = 'precomposed_svg')
    def plot_pipeline(self,**kwargs):
        if self.pipeline is not None:
            return mpl_plot_to_bytes(self.pipeline.draw(),format='svg')
        else:
            return None

    @Driver.unqueued(render_hint = 'html')
    def last_result(self,**kwargs):
        return self.last_results._repr_html_()

    @Driver.unqueued(render_hint = 'netcdf')
    def download_last_result(self,**kwargs):
        return xarray_to_bytes(self.last_results)
    
    @Driver.unqueued(render_hint = 'precomposed_png')
    def plot_operation(self,operation,**kwargs):
        try:
            operation = int(operation)
        except ValueError:
            pass
        if self.pipeline is not None:
            if isinstance(operation,str):
                return mpl_plot_to_bytes(self.pipeline.search(operation).plot(),format='png')
            elif isinstance(operation,int):
                return mpl_plot_to_bytes(self.pipeline[operation].plot(),format='png')
            else:
                return None
        else:
            return None

    @Driver.unqueued(render_hint='html')
    def pipeline_builder(self, **kwargs):
        """Serve the pipeline builder HTML interface."""
        return get_pipeline_builder_html()

    @Driver.unqueued()
    def pipeline_ops(self, **kwargs):
        """Return metadata for available PipelineOps."""
        return get_pipeline_ops()

    @Driver.unqueued()
    def current_pipeline(self, **kwargs):
        """Return the currently loaded pipeline as JSON."""
        if self.pipeline is None:
            return []
        return [op.to_json() for op in self.pipeline]

    @Driver.unqueued()
    def prefab_names(self, **kwargs):
        """List available prefabricated pipelines."""
        from AFL.double_agent.prefab import list_prefabs
        return list_prefabs(display_table=False)

    @Driver.unqueued()
    def load_prefab(self, name: str, **kwargs):
        """Load a prefabricated pipeline and return its JSON."""
        from AFL.double_agent.prefab import load_prefab
        pipeline = load_prefab(name)
        return [op.to_json() for op in pipeline]

    @Driver.unqueued()
    def build_pipeline(self, ops: str = "[]", name: str = "Pipeline", **kwargs):
        """Construct a pipeline from JSON and return the serialized form."""
        return build_pipeline_from_json(ops, name)

    def reset_results(self):
        self.results = dict()

    def predict(
        self,
        deposit: bool = True,
        save_to_disk: bool = True,
        sample_uuid: Optional[str] = None,
        AL_campaign_name: Optional[str] = None,
    ) -> str:
        """
        Evaluate the pipeline set with `.initialize_pipeline`.

        Parameters
        ----------
        deposit: bool
            If True, the `xarray.Dataset` resulting from the `Pipeline` calculation will be placed in this `APIServers`
            dropbox for retrieval and the `db_uuid` will be returned.

        save_to_disk: bool
            If True, the `xarray.Dataset` resulting from the `Pipeline` calculation will be serialized to disk in
            NetCDF format.

        sample_uuid: Optional[str]
            Optionally provide a sample uuid to tag the calculation with

        AL_campaign_name
            Optionally provide an AL campaign name to tag the calculation with

        """
        if (self.pipeline is None) or (self.input is None):
            raise ValueError(
                """Cannot predict without a pipeline and input loaded! Use client.set_driver_object to upload an """
                """Pipeline.Pipeline and an xr.Dataset to this APIServer. You currently have: \n"""
                f"""DoubleAgentDriver.pipeline = {self.pipeline}\n"""
                f"""DoubleAgentDriver.input = {self.input}\n"""
            )
        if sample_uuid is None:
            sample_uuid = 'SAM-'+str(uuid.uuid4())

        ag_uid = "AG-" + str(uuid.uuid4())
        self.results[ag_uid] = self.pipeline.calculate(self.input)

        self.results[ag_uid].attrs['sample_uuid'] = sample_uuid
        self.results[ag_uid].attrs['ag_uuid'] = ag_uid
        self.results[ag_uid].attrs['AL_campaign_name'] = AL_campaign_name

        if save_to_disk:
            path = (
                pathlib.Path(self.config["save_path"])
                / f"{AL_campaign_name}_SAM-{str(sample_uuid)[-6:]}_AG-{ag_uid[-6:]}.nc"
            )
            self.results[ag_uid].to_netcdf(path)

        if deposit:
            self.deposit_obj(self.results[ag_uid], uid=ag_uid)
        
        self.last_results = self.results[ag_uid]
        
        return ag_uid

_OVERRIDE_MAIN_MODULE_NAME = 'DoubleAgentDriver'
if __name__ == '__main__':
    from AFL.automation.shared.launcher import *
