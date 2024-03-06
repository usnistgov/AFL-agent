import os,sys,subprocess
from pathlib import Path
try:
        import AFL.automation
except:
        sys.path.append(os.path.abspath(Path(__file__).parent.parent))
        print(f'Could not find NistoRoboto on system path, adding {os.path.abspath(Path(__file__).parent.parent)} to PYTHONPATH')
import xarray as xr
from AFL.automation.APIServer.APIServer import APIServer
#from AFL.automation.instrument.VirtualSpec_data import VirtualSpec_data
from AFL.double_agent.SASFitter_Driver import SASModelWrapper, SASFitter_Driver
from AFL.automation.APIServer.data.DataTiled import DataTiled

data1 = DataTiled(server='http://0.0.0.0:8000', api_key = os.environ['TILED_API_KEY'], backup_path='/Users/drs18/.afl/json-backup')
server1 = APIServer('SANS_fitting_Server', data=data1)
server1.add_standard_routes()
server1.create_queue(SASFitter_Driver())
server1.init_logging()

data2 = DataTiled(server='http://0.0.0.0:8000', api_key = os.environ['TILED_API_KEY'], backup_path='/Users/drs18/.afl/json-backup')
server2 = APIServer('SAXS_fitting_Server', data=data2)
server2.add_standard_routes()
server2.create_queue(SASFitter_Driver())
server2.init_logging()

server1.run_threaded(host='0.0.0.0', port=5058)#, debug=True)
server2.run(host='0.0.0.0', port=5059)#, debug=True)
