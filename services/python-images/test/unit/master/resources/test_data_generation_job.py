from .base import BaseResourceTest
from test.factories import DatasetGenerationJobFactory
from src.master.resources.dataset_generation_job import DatasetGenerationJobResource, DatasetGenerationJobListResource


class DatasetGenerationJobTest(BaseResourceTest):
    def test_returns_all_dataset_generation_jobs(self):
        number_jobs = 5

        groundtruth_jobs = []

        for index in range(0, number_jobs):
            groundtruth_jobs.append(DatasetGenerationJobFactory())

        result = self.get(self.url_for(DatasetGenerationJobListResource))

        assert (len(result) == number_jobs)
        for (index, job) in enumerate(result):
            assert(job == groundtruth_jobs[index])

    def test_returns_my_dataset_generation_job(self):
        # Given
        job = DatasetGenerationJobFactory()

        # When
        result = self.get(self.url_for(DatasetGenerationJobResource, job_id=job.id))

        # Then
        assert result['id'] == job.id

        # TODO where does this come from?
        assert result['container_id'] == job.container_id

        # TODO add test for type

    # TODO test if generation is successful?
    # --> is this test useful because the job scheduler must generate the referenced dataset
    # def test_returns_job_for_dataset(self):
    #    pass
