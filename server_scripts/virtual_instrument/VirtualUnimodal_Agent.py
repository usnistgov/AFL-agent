import os,sys,subprocess
from pathlib import Path

try:
    import AFL.agent
except:
    sys.path.append(os.path.abspath(Path(__file__).parent.parent))
    print(f'Could not find AFL.agent on system path, adding {os.path.abspath(Path(__file__).parent.parent)} to PYTHONPATH')

from AFL.automation.APIServer.APIServer import APIServer
from AFL.agent.virtual_AL.vSAS_AgentDriver import SAS_AgentDriver
from AFL.automation.APIServer.data.DataTiled import DataTiled

data = DataTiled('http://localhost:8000',api_key = os.environ['TILED_API_KEY'],backup_path='/Users/drs18/Documents/multimodal-dev/')

server = APIServer('VirtualMultimodal_Agent',index_template="index.html",data=data)
server.add_standard_routes()
server.create_queue(SAS_AgentDriver())
server.init_logging()
server.run(host='0.0.0.0', port=5057)#, debug=True)
