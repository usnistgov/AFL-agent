import time
import requests

from AFL.automation.APIServer.Client import Client
from AFL.automation.shared.serialization import serialize, deserialize


class AgentClient(Client):
    """Communicate with DoubleAgent server """

    def set_mask(self, mask):
        json = {}
        json['task_name'] = 'set_mask'
        json['mask'] = serialize(mask)
        json['serialized'] = True
        self.enqueue(**json)

    def append_data(self, compositions, measurements, labels):
        json = {}
        json['task_name'] = 'append_data'
        json['compositions'] = serialize(compositions)
        json['measurements'] = serialize(measurements)
        json['labels'] = serialize(labels)
        self.enqueue(**json)

    def get_phasemap(self):
        json = {}
        json['task_name'] = 'get_object'
        json['name'] = 'phasemap'
        json['interactive'] = True
        retval = self.enqueue(**json)
        phasemap = deserialize(retval['return_val'])
        return phasemap

    def _get_next_sample(self):
        response = requests.get(self.url + '/get_next_sample', headers=self.headers)
        if response.status_code != 200:
            raise RuntimeError(
                f'API call to _get_next_sample command failed with status_code {response.status_code}\n{response.text}')
        return deserialize(response.json())

    def get_next_sample_queued(self):
        json = {}
        json['task_name'] = 'get_next_sample'
        json['interactive'] = True
        retval = self.enqueue(**json)
        obj = deserialize(retval['return_val'])
        return obj[0]

    def get_next_sample(self, wait_on_stale=True):
        json = {}
        json['task_name'] = 'get_next'
        json['interactive'] = True
        while True:
            next_sample, stale = self._get_next_sample()
            if (not wait_on_stale) or (not stale):
                break
            else:
                time.sleep(2)

        return next_sample

    def predict(self):
        kw = {}
        kw['task_name'] = 'predict'
        self.enqueue(**kw)
