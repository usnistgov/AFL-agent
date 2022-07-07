import os,sys,subprocess
from pathlib import Path
try:
        import AFL.automation
except:
        sys.path.append(os.path.abspath(Path(__file__).parent.parent))
        print(f'Could not find NistoRoboto on system path, adding {os.path.abspath(Path(__file__).parent.parent)} to PYTHONPATH')

from AFL.automation.APIServer.APIServer import APIServer
# from AFL.automation.agent.SAS_AgentDriver import SAS_AgentDriver
from AFL.agent.DowShampoo_SampleDriver import SAS_AL_SampleDriver


driver =SAS_AL_SampleDriver(
        load_url='piloader:5000',
        prep_url='piot2:5000',
        sas_url='localhost:5000',
        agent_url='localhost:5053',
        camera_urls = [],
        dummy_mode=False,
        )

server = APIServer('DowShampoo_AL_SampleDriver',index_template="index.html")
server.add_standard_routes()
server.create_queue(driver)
server.init_logging()
server.run(host='0.0.0.0', port=5052)#, debug=True)
