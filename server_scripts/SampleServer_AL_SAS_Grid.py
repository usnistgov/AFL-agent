import os,sys,subprocess
from pathlib import Path
try:
        import AFL.automation
except:
        sys.path.append(os.path.abspath(Path(__file__).parent.parent))
        print(f'Could not find NistoRoboto on system path, adding {os.path.abspath(Path(__file__).parent.parent)} to PYTHONPATH')

from AFL.automation.APIServer.APIServer import APIServer
from AFL.agent.SAS_Grid_AL_SampleDriver import SAS_Grid_AL_SampleDriver

driver =SAS_Grid_AL_SampleDriver(
        sas_url='localhost:5000',
        agent_url='localhost:5053',
        )

server = APIServer('SAS_AL_SampleDriver',index_template="index.html")
server.add_standard_routes()
server.create_queue(driver)
server.init_logging()
server.run(host='0.0.0.0', port=5050)#, debug=True)
