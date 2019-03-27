from src.master.resources import EdgeInformationListResource, EdgeInformationResource
from test.factories import ResultFactory, NodeFactory, EdgeInformationFactory
from .base import BaseResourceTest


class EdgeTest(BaseResourceTest):
    def test_returns_all_edge_information_for_result(self):
        # Given
        result = ResultFactory()
        nodes = [NodeFactory(dataset=result.job.experiment.dataset) for _ in range(3)]
        edges = [EdgeInformationFactory(result=result, from_node=nodes[i], to_node=nodes[j]) for i, j in [(0, 1),
                                                                                                          (1, 2)]]

        # When
        results = self.get(self.url_for(EdgeInformationListResource, result_id=result.id))

        # Then
        assert len(results) == len(edges)
        edge_ids = {e.id: e for e in edges}
        for edge in results:
            assert edge['result_id'] == result.id
            assert edge['id'] in edge_ids.keys()
            del edge_ids[edge['id']]
        assert len(edge_ids) == 0

    def test_delete_edge_information(self):
        # Given
        edge = EdgeInformationFactory()

        # When
        result = self.delete(self.url_for(EdgeInformationResource, edge_information_id=edge.id))

        # Then
        assert result['id'] == edge.id
        assert result['result_id'] == edge.result_id
