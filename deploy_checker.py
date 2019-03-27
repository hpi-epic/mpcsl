import os
import pickle
from subprocess import run, CalledProcessError

import requests

TOKEN = 'a66ddccc06f682bdefee2d0964ee676c87dfc7ca'

most_recent_backend_commit = ''
if os.path.exists('most_recent_backend_commit.pkl'):
    most_recent_backend_commit = pickle.load(open('most_recent_backend_commit.pkl', 'rb'))
r = requests.get('http://api.github.com/repos/hpi-epic/mpci/commits/master', auth=('mpci-deploy-user', TOKEN))
current_backend_commit = r.json()['sha']

most_recent_ui_commit = ''
if os.path.exists('most_recent_ui_commit.pkl'):
    most_recent_ui_commit = pickle.load(open('most_recent_ui_commit.pkl', 'rb'))
r = requests.get('http://api.github.com/repos/hpi-epic/mpci-frontend/commits/master', auth=('mpci-deploy-user', TOKEN))
current_ui_commit = r.json()['sha']

if current_backend_commit != most_recent_backend_commit or current_ui_commit != most_recent_ui_commit:
    r = requests.get('https://api.github.com/repos/hpi-epic/mpci/commits/{}/check-runs'.format(current_backend_commit),
                     auth=('mpci-deploy-user', TOKEN), params={'status': 'completed'},
                     headers={'Accept': 'application/vnd.github.antiope-preview+json'})
    checks = r.json()
    if any([c['conclusion'] == 'success' for c in checks['check_runs']]):
        print('Alright folks, time to deploy!')
        print('Old backend commit was {}, new one is {}'.format(most_recent_backend_commit, current_backend_commit))
        print('Old frontend commit was {}, new one is {}'.format(most_recent_ui_commit, current_ui_commit))
        pickle.dump(current_backend_commit, open('most_recent_backend_commit.pkl', 'wb'))
        pickle.dump(current_ui_commit, open('most_recent_ui_commit.pkl', 'wb'))

        try:
            run('git pull && bash scripts/server.sh --detach',
                shell=True, check=True)
        except CalledProcessError:
            os.remove('most_recent_backend_commit.pkl')
