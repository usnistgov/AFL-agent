import os,sys,subprocess
from pathlib import Path

try:
    import AFL.agent
except:
    sys.path.append(os.path.abspath(Path(__file__).parent.parent))
    print(f'Could not find AFL.agent on system path, adding {os.path.abspath(Path(__file__).parent.parent)} to PYTHONPATH')

from AFL.automation.APIServer.APIServer import APIServer
from AFL.agent.Multimodal_AgentDriver import Multimodal_AgentDriver

server = APIServer('SAS_Agent',index_template="index.html")
server.add_standard_routes()
server.create_queue(SAS_AgentDriver())
server.init_logging()
server.run(host='0.0.0.0', port=5053)#, debug=True)
