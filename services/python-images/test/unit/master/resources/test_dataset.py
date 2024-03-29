import io
import os
import tempfile
from unittest.mock import patch

import factory
import networkx as nx
import numpy as np
import pandas as pd
from marshmallow.utils import from_iso
from sqlalchemy import inspect
from src.db import db
from src.master.helpers.database import add_dataset_nodes
from src.master.resources.datasets import DatasetListResource, DatasetResource, DatasetLoadResource, \
    DatasetLoadWithIdsResource, DatasetAvailableSourcesResource, DatasetExperimentResource, \
    DatasetGroundTruthResource
from src.models import Dataset, Node
from test.factories import DatasetFactory, ExperimentFactory

from .base import BaseResourceTest


def create_database_table():
    db.session.execute("""
                CREATE TABLE IF NOT EXISTS test_data (
                    a float,
                    b float,
                    c float
                );
            """)

    mean = [0, 5, 10]
    cov = [[1, 0, 0], [0, 10, 0], [0, 0, 20]]
    source = np.random.multivariate_normal(mean, cov, size=50)
    for row in source:
        db.session.execute(
            "INSERT INTO test_data VALUES ({0})".format(",".join([str(e) for e in row])))
    db.session.commit()
    return source


class DatasetTest(BaseResourceTest):
    def test_returns_all_data_sets(self):
        # Given
        ds = DatasetFactory()
        ds2 = DatasetFactory()

        # When
        result = self.get(self.url_for(DatasetListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == ds.id
        assert result[1]['id'] == ds2.id

    def test_returns_my_data_set(self):
        # Given
        ds = DatasetFactory()

        # When
        result = self.get(self.url_for(DatasetResource, dataset_id=ds.id))

        # Then
        assert result['id'] == ds.id
        assert result['load_query'] == ds.load_query
        assert result['name'] == ds.name
        assert result['description'] == ds.description
        assert from_iso(result['time_created']) == ds.time_created

    def test_create_new_data_set(self):
        # Given
        create_database_table()

        data = factory.build(dict, FACTORY_CLASS=DatasetFactory)
        data['load_query'] = 'SELECT * FROM test_data'
        assert len(db.session.query(Node).all()) == 0

        # When
        result = self.post(self.url_for(DatasetListResource), json=data)
        ds = db.session.query(Dataset).first()

        # Then
        assert ds.load_query == data['load_query'] == result['load_query']

        nodes = db.session.query(Node).filter_by(dataset=ds).all()
        for node in nodes:
            assert node.name in ['a', 'b', 'c']
        assert len(nodes) == 3

    def test_returns_the_correct_dataset(self):
        # Given
        source = create_database_table()
        ds = DatasetFactory(load_query="SELECT * FROM test_data")
        add_dataset_nodes(ds)
        nodes = ds.nodes

        # When
        result = self.test_client.get(self.url_for(DatasetLoadResource, dataset_id=ds.id))

        source = pd.DataFrame(source)
        source.columns = [n.id for n in nodes]
        result = result.data.decode('utf-8')
        result = io.StringIO(result)
        result = pd.read_csv(result)
        result.rename(columns={'a': 1, 'b': 2, 'c': 3}, inplace=True)

        # Then
        pd.testing.assert_frame_equal(source, result)

    def test_returns_the_correct_dataset_with_ids(self):
        # Given
        source = create_database_table()
        ds = DatasetFactory(load_query="SELECT * FROM test_data")
        add_dataset_nodes(ds)
        nodes = ds.nodes

        # When
        result = self.test_client.get(self.url_for(DatasetLoadWithIdsResource, dataset_id=ds.id))

        source = pd.DataFrame(source)
        source.columns = [n.id for n in nodes]
        result = result.data.decode('utf-8')
        result = io.StringIO(result)
        result = pd.read_csv(result)
        result.rename(columns=int, inplace=True)

        # Then
        pd.testing.assert_frame_equal(source, result)

    def test_delete_dataset(self):
        # Given
        create_database_table()
        dataset = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )
        add_dataset_nodes(dataset)
        assert len(db.session.query(Node).all()) == 3

        # When
        result = self.delete(self.url_for(DatasetResource, dataset_id=dataset.id))

        # Then
        assert result['id'] == dataset.id
        assert inspect(dataset).detached is True
        assert len(db.session.query(Node).all()) == 0

    def test_datasource(self):
        # Given
        data_sources = {'HANA': 'hana+pyhdb://'}

        # When
        with patch('src.master.resources.datasets.DATA_SOURCE_CONNECTIONS', data_sources):
            result = self.get(self.url_for(DatasetAvailableSourcesResource))

        # Then
        assert result['data_sources'] == ['HANA', 'postgres']

    def test_change_dataset_description(self):
        ds = DatasetFactory()
        ds.description = '1'
        result = self.put(self.url_for(DatasetResource, dataset_id=ds.id),
                          json={'description': '2'})
        assert ds.description == '2'
        result = self.put(self.url_for(DatasetResource, dataset_id=ds.id),
                          json={'asfasf': '3'},
                          parse_result=False)
        assert ds.description == '2'
        assert result.status_code == 400

    def test_dataset_experiment(self):
        ds = DatasetFactory()
        ex = ExperimentFactory()
        ex.dataset = ds
        result = self.get(self.url_for(DatasetExperimentResource, dataset_id=ds.id))
        assert (result[0]['id'] == ex.id)

    def test_dataset_ground_truth_upload(self):
        # Given
        ds = DatasetFactory()
        for name in ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"]:
            node = Node(name=name, dataset=ds)
            db.session.add(node)
        db.session.commit()
        dirname = os.path.dirname(__file__)
        fixture = os.path.join(dirname, '../../../fixtures/earthquake_groundtruth.gml')
        data = dict(
            graph_file=(open(fixture, 'rb'), "earthquake_groundtruth.gml"),
        )
        # When
        response = self.test_client.post(
            self.url_for(DatasetGroundTruthResource, dataset_id=ds.id),
            content_type='multipart/form-data',
            data=data
        )
        # Then
        assert response.status_code == 200

    def test_dataset_ground_truth_download(self):
        # Given
        ds = DatasetFactory()
        for name in ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"]:
            node = Node(name=name, dataset=ds)
            db.session.add(node)
        db.session.commit()
        dirname = os.path.dirname(__file__)
        fixture = os.path.join(dirname, '../../../fixtures/earthquake_groundtruth.gml')
        data = dict(
            graph_file=(open(fixture, 'rb'), "earthquake_groundtruth.gml"),
        )
        # When
        self.test_client.post(
            self.url_for(DatasetGroundTruthResource, dataset_id=ds.id),
            content_type='multipart/form-data',
            data=data
        )
        # Then
        res = self.test_client.get(
            self.url_for(DatasetGroundTruthResource, dataset_id=ds.id),
            query_string={'format': 'GML'}
        )

        assert res.headers['content-type'] == 'text/plain; charset=utf-8'

        downloaded_graph_path = os.path.join(tempfile.gettempdir(), 'gt-download.gml')
        with open(downloaded_graph_path, 'w') as f:
            f.write(res.data.decode('utf-8'))

        ground_truth = nx.read_gml(fixture)
        downloaded_graph = nx.read_gml(downloaded_graph_path)
        assert nx.graph_edit_distance(ground_truth, downloaded_graph) == 0.0

    def test_dataset_ground_truth_upload_igraph_generated(self):
        # Tests that the upload works with a .gml file coming from igraph package in R
        # Given
        ds = DatasetFactory()
        for name in ["1", "2", "3", "4", "5"]:
            node = Node(name=name, dataset=ds)
            db.session.add(node)
        db.session.commit()
        dirname = os.path.dirname(__file__)
        fixture = os.path.join(dirname, '../../../fixtures/graph.gml')
        data = dict(
            graph_file=(open(fixture, 'rb'), "graph.gml"),
        )
        # When
        response = self.test_client.post(
            self.url_for(DatasetGroundTruthResource, dataset_id=ds.id),
            content_type='multipart/form-data',
            data=data
        )
        # Then
        assert response.status_code == 200

    def test_dataset_ground_truth_upload_should_fail_on_invalid_file(self):
        # Given
        ds = DatasetFactory()
        db.session.commit()

        dirname = os.path.dirname(__file__)
        fixture = os.path.join(dirname, '../../../fixtures/invalid.gml')
        data = dict(
            graph_file=(open(fixture, 'rb'), "invalid.gml"),
        )
        # When
        response = self.test_client.post(
            self.url_for(DatasetGroundTruthResource, dataset_id=ds.id),
            content_type='multipart/form-data',
            data=data
        )

        assert response.status_code == 400

    def test_dataset_ground_truth_upload_should_fail_on_missing_file(self):
        # Given
        ds = DatasetFactory()
        db.session.commit()
        # When
        response = self.test_client.post(
            self.url_for(DatasetGroundTruthResource, dataset_id=ds.id),
            content_type='multipart/form-data',
            data={}
        )

        assert response.status_code == 400
