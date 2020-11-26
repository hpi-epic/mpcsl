from .base import BaseResourceTest
from test.factories import DatasetGenerationJobFactory
from src.master.resources.dataset_generation_job import DatasetGenerationJobResource, DatasetGenerationJobListResource


class DatasetGenerationJobTest(BaseResourceTest):
    def test_returns_all_dataset_generation_jobs(self):
        # Given
        job = DatasetGenerationJobFactory()
        job2 = DatasetGenerationJobFactory()

        # When
        result = self.get(self.url_for(DatasetGenerationJobListResource))
        print(result)
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

        # TODO add test for type

    # TODO test if generation is successful?
    # --> is this test useful because the job scheduler must generate the referenced dataset
    # def test_returns_job_for_dataset(self):
    #    pass
