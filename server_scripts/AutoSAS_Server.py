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
from AFL.double_agent.AutoSAS_Driver import SASModelWrapper, AutoSAS_Driver
from AFL.automation.APIServer.data.DataTiled import DataTiled

### local tiled server here
data = DataTiled(server='http://0.0.0.0:8000', api_key = os.environ['TILED_API_KEY'], backup_path='/Users/drs18/.afl/json-backup')
server = APIServer('AutoSAS_server', data=data)

server.add_standard_routes()

server.create_queue(AutoSAS_Driver())

server.init_logging()

#does this port need to be soemthing special?
server.run(host='0.0.0.0', port=5058)#, debug=True)
