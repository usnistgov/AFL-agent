import os,sys,subprocess
from pathlib import Path
try:
        import AFL.automation
except:
        sys.path.append(os.path.abspath(Path(__file__).parent.parent))
        print(f'Could not find NistoRoboto on system path, adding {os.path.abspath(Path(__file__).parent.parent)} to PYTHONPATH')

from AFL.automation.APIServer.APIServer import APIServer
from AFL.agent.SAS_AgentDriver import SAS_AgentDriver
from AFL.agent.SAS_AL_SampleDriver import SAS_AL_SampleDriver
from AFL.automation.APIServer.data.DataTiled import DataTiled

data = DataTiled('http://10.42.0.1:8000',api_key = os.environ['TILED_API_KEY'],backup_path='/home/afl642/.afl/json-backup')


driver =SAS_AL_SampleDriver(
        load_url ='piloader:5000',
        prep_url ='piot2:5000',
        sas_url  ='localhost:5000',
        agent_url='localhost:5053',
        spec_url ='pispectrometer:5050',
        camera_urls = [],
        )

server = APIServer('AL_SampleDriver',index_template="index.html",data=data)
server.add_standard_routes()
server.create_queue(driver)
server.init_logging()
server.run(host='0.0.0.0', port=5052)#, debug=True)
