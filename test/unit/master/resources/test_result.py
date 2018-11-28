from datetime import datetime

from src.db import db
from src.master.resources.results import ResultListResource
from src.models import Result, Experiment, Job, Node, Edge
from test.factories import DatasetFactory
from .base import BaseResourceTest


class DatasetTest(BaseResourceTest):

    def test_create_new_data_set(self):
        # Given
        ds = DatasetFactory()
        mock_experiment = Experiment(dataset=ds)
        db.session.add(mock_experiment)
        mock_job = Job(experiment=mock_experiment, start_time=datetime.now())
        db.session.add(mock_job)
        db.session.commit()

        data = {
            'job_id': mock_job.id,
            'meta_results': {'important_note': 'lol'},
            'edge_list': [
                {'from_node': 'X1', 'to_node': 'X2'}
            ],
            'node_list': [
                'X1', 'X2', 'X3'
            ],
            'sepset_list': [
                {'from_node': 'X2', 'to_node': 'X3', 'nodes': ['X1'], 'level': 1, 'statistic': 0.5}
            ]
        }

        # When
        result = self.post(self.api.url_for(ResultListResource), json=data)
        db_result = db.session.query(Result).first()

        # Then
        assert db_result.meta_results == data['meta_results'] == result['meta_results']
        assert db.session.query(Job).first() is None

        for node in db.session.query(Node):
            assert node.name in data['node_list']

        for edge in data['edge_list']:
            assert db.session.query(Edge).filter(
                    Edge.from_node.has(name=edge['from_node'])
                ).filter(
                    Edge.to_node.has(name=edge['to_node'])
                ).first() is not None
