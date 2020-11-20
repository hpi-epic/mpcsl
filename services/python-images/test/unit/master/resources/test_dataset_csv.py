import os

from src.master.resources import DatasetCsvUploadResource
from src.master.resources.datasets import load_dataset_as_csv
from src.models import Dataset
from test.unit.master.resources.base import BaseResourceTest
import pandas as pd


class DatasetCsvUploadResourceTest(BaseResourceTest):

    def test_post_success(self):
        dirname = os.path.dirname(__file__)
        fixture = os.path.join(dirname, '../../../fixtures/generated_dataset.csv')
        data = dict(
            file=(open(fixture, 'rb'), "generated.csv"),
        )

        response = self.test_client.post(
            self.url_for(DatasetCsvUploadResource),
            content_type='multipart/form-data',
            data=data
        )

        assert response.status_code == 200
        ds = self.db.session.query(Dataset).first()
        result_buffer = load_dataset_as_csv(self.db.session, ds)
        result_dataframe = pd.read_csv(result_buffer)

        expected_dataframe = pd.read_csv(fixture, index_col=0)
        result_dataframe.index = expected_dataframe.index
        pd.testing.assert_frame_equal(expected_dataframe, result_dataframe)
