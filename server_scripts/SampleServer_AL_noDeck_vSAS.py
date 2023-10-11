import os,sys,subprocess
from pathlib import Path
try:
        import AFL.automation
except:
        sys.path.append(os.path.abspath(Path(__file__).parent.parent))
        print(f'Could not find NistoRoboto on system path, adding {os.path.abspath(Path(__file__).parent.parent)} to PYTHONPATH')

from AFL.automation.APIServer.APIServer import APIServer
from AFL.agent.vSAS_NoDeck_AL_SampleDriver import vSAS_NoDeck_AL_SampleDriver
from AFL.automation.APIServer.data.DataTiled import DataTiled

data = DataTiled('http://localhost:8000',api_key = os.environ['TILED_API_KEY'],backup_path='/home/afl642/.afl/json-backup')


driver = vSAS_NoDeck_AL_SampleDriver(
        load_url ='localhost:5051',
        prep_url ='localhost:5052',
        sas_url  ='localhost:5054',
        agent_url='localhost:5053',
        tiled_url='http://localhost:8000'
        )

server = APIServer('AL_SampleDriver',index_template="index.html",data=data)
server.add_standard_routes()
server.create_queue(driver)
server.init_logging()
server.run(host='0.0.0.0', port=5000)#, debug=True)
