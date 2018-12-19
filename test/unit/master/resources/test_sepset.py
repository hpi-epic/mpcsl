from src.master.resources import SepsetResource, ResultSepsetListResource
from test.factories import ResultFactory, NodeFactory, SepsetFactory
from .base import BaseResourceTest


class SepsetTest(BaseResourceTest):
    def test_returns_all_sepsets_for_result(self):
        # Given
        result = ResultFactory()
        nodes = [NodeFactory(result=result) for _ in range(3)]
        sepsets = [SepsetFactory(result=result, from_node=nodes[0], to_node=nodes[2], node_names=[nodes[1].name])]

        # When
        results = self.get(self.api.url_for(ResultSepsetListResource, result_id=result.id))

        # Then
        assert len(results) == len(sepsets)
        sepset_ids = {s.id for s in sepsets}
        for sepset in results:
            assert sepset['result_id'] == result.id
            assert sepset['id'] in sepset_ids
            sepset_ids.remove(sepset['id'])
        assert len(sepset_ids) == 0

    def test_returns_my_sepset(self):
        # Given
        sepset = SepsetFactory()

        # When
        result = self.get(self.api.url_for(SepsetResource, sepset_id=sepset.id))

        # Then
        assert result['id'] == sepset.id
        assert result['result_id'] == sepset.result_id
