from .base import BaseResourceTest
from test.factories import DatasetGenerationJobFactory
from src.master.resources.dataset_generation_job import DatasetGenerationJobResource, DatasetGenerationJobListResource
import os
from src.master.resources.datasets import load_dataset_as_csv
from src.models import Dataset, DatasetGenerationJob
import pandas as pd


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

    def test_put_dataset(self):
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
        ds = self.db.session.query(Dataset).first()
        result_buffer = load_dataset_as_csv(self.db.session, ds)
        result_dataframe = pd.read_csv(result_buffer)

        expected_dataframe = pd.read_csv(fixture, index_col=0)
        result_dataframe.index = expected_dataframe.index
        pd.testing.assert_frame_equal(expected_dataframe, result_dataframe)

        updated_job = DatasetGenerationJob.query.get(job.id)

        assert updated_job.dataset == ds
