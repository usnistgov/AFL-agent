import os,sys,subprocess
from pathlib import Path
try:
        import AFL.automation
except:
        sys.path.append(os.path.abspath(Path(__file__).parent.parent))
        print(f'Could not find NistoRoboto on system path, adding {os.path.abspath(Path(__file__).parent.parent)} to PYTHONPATH')

from AFL.automation.APIServer.APIServer import APIServer
from AFL.agent.virtual_AL.vSAS_AL_SampleDriver import SAS_AL_SampleDriver
from AFL.automation.APIServer.data.DataTiled import DataTiled

data = DataTiled('http://localhost:8000',api_key = os.environ['TILED_API_KEY'],backup_path='/Users/drs18/.afl/json-backup')


driver = SAS_AL_SampleDriver(
        load_url ='localhost:5051',
        prep_url ='localhost:5052',
        sas_url  ='localhost:5054',
        agent_url='localhost:5057',
       # spec_url ='pispectrometer:5050',
        tiled_url = 'http://localhost:8000',
        #turb_url = 'localhost:5001',
        #camera_urls = [
        #    'http://afl-video:8081/101/current',
        #    'http://afl-video:8081/103/current',
        #    'http://afl-video:8081/104/current',
        #    ],
        )

server = APIServer('Unimodal_AL_SampleDriver',index_template="index.html",data=data)
server.add_standard_routes()
server.create_queue(driver)
server.init_logging()
server.run(host='0.0.0.0', port=5053)#, debug=True)
