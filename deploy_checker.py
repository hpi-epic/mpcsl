import os
import pickle
from subprocess import run, CalledProcessError

import requests

DEPLOY_TOKEN = 'a66ddccc06f682bdefee2d0964ee676c87dfc7ca'

most_recent_commit = ''
if os.path.exists('most_recent_commit.pkl'):
    most_recent_commit = pickle.load(open('most_recent_commit.pkl', 'rb'))

r = requests.get('http://api.github.com/repos/danthe96/mpci/commits/master', auth=('mpci-deploy-user', DEPLOY_TOKEN))
current_commit = r.json()['sha']

if current_commit != most_recent_commit:
    r = requests.get('https://api.github.com/repos/danthe96/mpci/commits/{}/check-runs'.format(current_commit),
                     auth=('mpci-deploy-user', DEPLOY_TOKEN), params={'status': 'completed'},
                     headers={'Accept': 'application/vnd.github.antiope-preview+json'})
    checks = r.json()
    if any([c['conclusion'] == 'success' for c in checks['check_runs']]):
        print('Alright folks, time to deploy!')
        print('Old commit was {}, new one is {}'.format(most_recent_commit, current_commit))
        pickle.dump(current_commit, open('most_recent_commit.pkl', 'wb'))

        try:
            run('git pull && docker-compose down && docker-compose build && docker-compose up --detach',
                shell=True, check=True)
        except CalledProcessError:
            os.remove('most_recent_commit.pkl')
