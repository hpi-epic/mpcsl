import pytest
import requests

from src.db import db
from src.master.resources import ExecutorResource
from test.factories import ExperimentFactory
from .base import BaseIntegrationTest


class ExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-15)
    def test_r_execution_db_change_detection_alter_table(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_gauss(), algorithm__script_filename='pcalg.r')
        # Change underlying database table to modify content_hash
        db.session.execute("ALTER TABLE test_data ADD new_column integer")

        # When
        job_r = requests.post(self.url_for(ExecutorResource, experiment_id=ex.id))

        # Then
        assert job_r.status_code == 409

    @pytest.mark.run(order=-14)
    def test_r_execution_db_change_detection_insert(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_gauss(), algorithm__script_filename='pcalg.r')
        # Change underlying database table to modify content_hash
        db.session.execute("INSERT INTO test_data VALUES (1.2,2.3,3.4)")

        # When
        job_r = requests.post(self.url_for(ExecutorResource, experiment_id=ex.id))

        # Then
        assert job_r.status_code == 409

    @pytest.mark.run(order=-13)
    def test_r_execution_sql_error(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_gauss(), algorithm__script_filename='pcalg.r')
        ex.dataset.load_query = 'SELECT * FROM non_existing_table'
        db.session.commit()

        # When
        job_r = requests.post(self.url_for(ExecutorResource, experiment_id=ex.id))

        # Then
        assert job_r.status_code == 400
