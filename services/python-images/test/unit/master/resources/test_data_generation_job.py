from .base import BaseResourceTest
from test.factories import DatasetGenerationJobFactory
from src.master.resources.dataset_generation_job import DatasetGenerationJobResource, DatasetGenerationJobListResource
import os
from src.master.resources.datasets import load_dataset_as_csv
from src.models import Dataset, DatasetGenerationJob
import pandas as pd
from src.db import db


class DatasetGenerationJobTest(BaseResourceTest):
    def test_returns_all_dataset_generation_jobs(self):
        # Given
        job = DatasetGenerationJobFactory()
        job2 = DatasetGenerationJobFactory()

        # When
        result = self.get(self.url_for(DatasetGenerationJobListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == job.id
        assert result[1]['id'] == job2.id

    def test_returns_my_dataset_generation_job(self):
        # Given
        job = DatasetGenerationJobFactory()

        # When
        result = self.get(self.url_for(DatasetGenerationJobResource, job_id=job.id))

        # Then
        assert result['id'] == job.id

    def test_put_upload_dataset(self):
        # Given
        job = DatasetGenerationJobFactory()

        dirname = os.path.dirname(__file__)
        fixture = os.path.join(dirname, '../../../fixtures/generated_dataset.csv')
        data = dict(
            file=(open(fixture, 'rb'), "generated.csv"),
        )

        # When
        response = self.test_client.put(
            self.url_for(DatasetGenerationJobResource, job_id=job.id),
            content_type='multipart/form-data',
            data=data
        )

        # Then
        assert response.status_code == 200
        assert response.json["id"] is not None

        ds = self.db.session.query(Dataset).first()
        result_buffer = load_dataset_as_csv(self.db.session, ds)
        result_dataframe = pd.read_csv(result_buffer)

        expected_dataframe = pd.read_csv(fixture)
        result_dataframe.index = expected_dataframe.index
        pd.testing.assert_frame_equal(expected_dataframe, result_dataframe)

        updated_job: DatasetGenerationJob = DatasetGenerationJob.query.get(job.id)

        assert updated_job.dataset == ds
        assert updated_job.end_time is not None

    def test_abort_after_second_upload_for_same_id(self):
        # Given
        job = DatasetGenerationJobFactory()

        dirname = os.path.dirname(__file__)
        fixture = os.path.join(dirname, '../../../fixtures/generated_dataset.csv')
        firstData = dict(
            file=(open(fixture, 'rb'), "generated.csv"),
        )
        secondData = dict(
            file=(open(fixture, 'rb'), "generated.csv"),
        )

        # When
        self.test_client.put(
            self.url_for(DatasetGenerationJobResource, job_id=job.id),
            content_type='multipart/form-data',
            data=firstData
        )

        response = self.test_client.put(
            self.url_for(DatasetGenerationJobResource, job_id=job.id),
            content_type='multipart/form-data',
            data=secondData
        )

        # Then
        assert response.status_code == 400

    def test_create_dataset_generation_job(self):
        # Given
        data = dict()
        data['parameters'] = "{'nodes': 10, 'samples':1000}"
        data['generator_type'] = 'MPCI'
        data['datasetName'] = 'creation_test_dataset'
        data['kubernetesNode'] = 'test_k8s_node'

        # When
        self.post(self.url_for(DatasetGenerationJobListResource), json=data)
        job: DatasetGenerationJob = db.session.query(DatasetGenerationJob).first()

        # Then
        assert job.dataset_id is None
        assert job.datasetName == data['datasetName']
        assert job.generator_type == data['generator_type']
        assert job.node_hostname == data['kubernetesNode']
        assert job.parameters == data['parameters']

    def test_create_dataset_generation_job_without_kubernetes_node(self):
        # Given
        data = dict()
        data['parameters'] = "{'nodes': 10, 'samples':1000}"
        data['generator_type'] = 'MPCI'
        data['datasetName'] = 'creation_test_dataset'

        # When
        self.post(self.url_for(DatasetGenerationJobListResource), json=data)
        job: DatasetGenerationJob = db.session.query(DatasetGenerationJob).first()

        # Then
        assert job.dataset_id is None
        assert job.datasetName == data['datasetName']
        assert job.generator_type == data['generator_type']
        assert job.parameters == data['parameters']

        assert job.node_hostname is None
